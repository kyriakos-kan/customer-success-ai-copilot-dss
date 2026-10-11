"""Pipeline container entrypoint: monitoring -> analyst -> RAG -> NBA -> writer.

Each run is idempotent and tagged with a run_id. Agents hand off through
Postgres, so the API and frontend never call the agents directly.

    python -m copilot.pipeline.orchestrator           # loop on schedule
    python -m copilot.pipeline.orchestrator --once    # single run (cron / demo)
    python -m copilot.pipeline.orchestrator --retrain # force model retraining
"""
import argparse
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from copilot.agents.analyst import AnalystAgent
from copilot.agents.monitoring import MonitoringAgent
from copilot.agents.nba import recommend_next_best_action
from copilot.agents.writer import WriterAgent
from copilot.config import get_settings
from copilot.data import FEATURES, load_gold
from copilot.db import Account, Draft, ReviewEvent, RiskScore, SessionLocal, init_db
from copilot.feedback import writer_feedback
from copilot.rag import ContextRetriever, synthetic_context
from copilot.trust import in_control_group

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pipeline")


def run_once(force_retrain: bool = False) -> dict:
    s = get_settings()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-") + uuid.uuid4().hex[:6]
    log.info("Run %s started", run_id)
    init_db()

    gold = load_gold()
    analyst = AnalystAgent(s.model_dir, s.risk_high, s.risk_medium)
    if force_retrain or not analyst.load():
        log.info("Training models: %s", analyst.train(gold))

    retriever = ContextRetriever()
    retriever.index(synthetic_context(gold))

    with SessionLocal() as session:
        for row in gold.itertuples():                         # upsert Gold snapshot
            session.merge(Account(account_id=row.account_id, name=row.name, segment=row.segment,
                                  supplier_origin=row.supplier_origin,
                                  features={f: getattr(row, f) for f in FEATURES} |
                                           {"avg_order_value": row.avg_order_value}))
        session.commit()

        flagged = MonitoringAgent().run(gold)
        log.info("Monitoring flagged %d / %d accounts", len(flagged.account_ids), len(gold))
        assessments = analyst.assess(gold, flagged.account_ids)
        # Prioritise by revenue at risk so the review queue starts with what matters
        assessments.sort(key=lambda a: -a.revenue_at_risk)

        writer = WriterAgent(s.anthropic_api_key, s.writer_model, s.min_context_docs)
        feedback_cache: dict[str, object] = {}       # one feedback lookup per action per run
        pending = set(session.scalars(select(Draft.account_id)
                                      .where(Draft.status == "pending_review")).all())
        since = datetime.now(timezone.utc) - timedelta(days=s.recontact_cooldown_days)
        recently_approved = set(session.scalars(
            select(Draft.account_id).join(ReviewEvent, ReviewEvent.draft_id == Draft.id)
            .where(ReviewEvent.decision == "approved", ReviewEvent.created_at >= since)).all())
        lookup = gold.set_index("account_id")
        counts = {"scored": 0, "drafted": 0, "insufficient": 0, "no_action": 0, "skipped": 0,
                  "recently_contacted": 0, "control": 0}
        llm_used = 0
        log.info("Writer: %s (LLM drafts per run: %s)", writer.label, s.llm_max_drafts or "no limit")
        fb = writer_feedback(session, "")
        log.info("Feedback log: %d approved examples, %d reviewer corrections available",
                 len(fb.examples), len(fb.lessons))

        for a in assessments:
            control = in_control_group(a.account_id, s.control_group_pct)
            session.add(RiskScore(run_id=run_id, account_id=a.account_id, clv_score=a.clv_score,
                                  churn_probability=a.churn_probability, risk_level=a.risk_level,
                                  top_factors=a.top_factors,
                                  monitor_signals=flagged.flagged[a.account_id],
                                  cohort="control" if control else "copilot"))
            counts["scored"] += 1
            if control:                                       # pilot control group: handled as today
                counts["control"] += 1
                continue
            if a.account_id in pending:                       # don't stack duplicate drafts
                counts["skipped"] += 1
                continue
            if a.account_id in recently_approved:             # outreach already approved
                counts["recently_contacted"] += 1
                continue
            acct = lookup.loc[a.account_id]
            rec = recommend_next_best_action(a.risk_level, acct.segment, a.top_factors, s.playbook_path)
            query = f"{rec.action} {' '.join(f['label'] for f in a.top_factors[:3])}"
            context = retriever.retrieve(a.account_id, query)
            use_llm = s.llm_max_drafts == 0 or llm_used < s.llm_max_drafts
            if rec.action not in feedback_cache:
                feedback_cache[rec.action] = writer_feedback(session, rec.action)
            d = writer.draft(acct["name"], a, rec, context, use_llm=use_llm,
                             feedback=feedback_cache[rec.action])
            if use_llm and writer.backend != "template" and context and d.status != "no_action":
                llm_used += 1
            if d.status == "no_action":
                counts["no_action"] += 1
                continue
            # An "insufficient data" placeholder is never reviewed; replace it instead of stacking.
            session.query(Draft).filter(Draft.account_id == a.account_id,
                                        Draft.status == "insufficient_data").delete()
            session.add(Draft(run_id=run_id, account_id=a.account_id, action=rec.action,
                              action_rationale=rec.rationale, subject=d.subject, body=d.body,
                              context_used=[{"text": c["text"], **c["metadata"]} for c in d.context_used],
                              status=d.status, feedback_used=d.feedback_used))
            counts["drafted" if d.status == "pending_review" else "insufficient"] += 1
        session.commit()

    log.info("Run %s finished: %s", run_id, counts)
    return {"run_id": run_id, **counts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()
    interval = get_settings().pipeline_interval_seconds
    retrain = args.retrain
    while True:
        try:
            run_once(force_retrain=retrain)
            retrain = False
        except Exception:
            log.exception("Pipeline run failed; will retry next cycle")
            if args.once:
                raise
        if args.once:
            break
        time.sleep(interval)


if __name__ == "__main__":
    main()
