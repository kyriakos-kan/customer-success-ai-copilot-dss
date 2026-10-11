"""FastAPI service. Reads what the pipeline wrote; the only write path is
the human review decision. No ML libraries are imported here, so the API
image stays small and starts in about a second."""
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from copilot.agents.nba import recommend_next_best_action
from copilot.config import get_settings
from copilot.db import Account, Draft, ReviewEvent, RiskScore, SessionLocal, init_db
from copilot.feedback import (REASONS, FeedbackError, prompt_block, record_review,
                              training_records, writer_feedback)



@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="Customer Success AI Copilot", version="1.0.0", lifespan=lifespan)


def db():
    with SessionLocal() as session:
        yield session


def _latest_score(session: Session, account_id: str) -> RiskScore:
    score = session.scalars(select(RiskScore).where(RiskScore.account_id == account_id)
                            .order_by(RiskScore.created_at.desc(), RiskScore.id.desc())
                            .limit(1)).first()
    if score is None:
        raise HTTPException(404, f"No risk score for {account_id} (not flagged yet, or unknown).")
    return score


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/accounts")
def list_accounts(risk_level: str | None = None, limit: int = Query(100, le=500),
                  session: Session = Depends(db)):
    """Latest score per flagged account, sorted by revenue at risk."""
    latest = (select(RiskScore.account_id, func.max(RiskScore.id).label("id"))
              .group_by(RiskScore.account_id).subquery())
    q = (select(RiskScore, Account).join(latest, RiskScore.id == latest.c.id)
         .join(Account, Account.account_id == RiskScore.account_id))
    if risk_level:
        q = q.where(RiskScore.risk_level == risk_level)
    rows = session.execute(q).all()
    out = [{"account_id": a.account_id, "name": a.name, "segment": a.segment,
            "supplier_origin": a.supplier_origin, "risk_level": r.risk_level,
            "churn_probability": r.churn_probability, "clv_score": r.clv_score,
            "revenue_at_risk": round(r.churn_probability * r.clv_score, 2),
            "signals": r.monitor_signals} for r, a in rows]
    return sorted(out, key=lambda x: -x["revenue_at_risk"])[:limit]


@app.get("/accounts/{account_id}/risk")
def account_risk(account_id: str, session: Session = Depends(db)):
    s = _latest_score(session, account_id)
    return {"account_id": account_id, "risk_level": s.risk_level,
            "churn_probability": s.churn_probability, "clv_score": s.clv_score,
            "run_id": s.run_id, "scored_at": s.created_at}


@app.get("/accounts/{account_id}/explanation")
def account_explanation(account_id: str, session: Session = Depends(db)):
    s = _latest_score(session, account_id)
    return {"account_id": account_id, "top_factors": s.top_factors, "signals": s.monitor_signals}


@app.get("/accounts/{account_id}/next-best-action")
def next_best_action(account_id: str, session: Session = Depends(db)):
    s = _latest_score(session, account_id)
    acct = session.get(Account, account_id)
    rec = recommend_next_best_action(s.risk_level, acct.segment, s.top_factors,
                                     get_settings().playbook_path)
    return rec.__dict__


@app.get("/drafts")
def list_drafts(status: str = "pending_review", session: Session = Depends(db)):
    rows = session.execute(select(Draft, Account).join(Account)
                           .where(Draft.status == status)
                           .order_by(Draft.created_at.desc())).all()
    return [{"id": d.id, "account_id": d.account_id, "account_name": a.name,
             "action": d.action, "action_rationale": d.action_rationale,
             "subject": d.subject, "body": d.body, "context_used": d.context_used,
             "status": d.status, "feedback_used": d.feedback_used,
             "created_at": d.created_at} for d, a in rows]


class ReviewIn(BaseModel):
    reviewer: str = Field(min_length=1)
    decision: str = Field(pattern="^(approved|rejected|needs_edit)$")
    edited_body: str | None = None
    # Required for rejected / needs_edit (see GET /feedback/reasons)
    reason: str | None = None
    explanation: str | None = None
    comment: str | None = None          # older name for explanation, still accepted


@app.post("/drafts/{draft_id}/review")
def review_draft(draft_id: int, body: ReviewIn, session: Session = Depends(db)):
    try:
        d = record_review(session, draft_id, body.reviewer, body.decision,
                          reason=body.reason, explanation=body.explanation or body.comment,
                          edited_body=body.edited_body)
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except FeedbackError as exc:
        status = 409 if "already" in str(exc) else 422
        raise HTTPException(status, str(exc))
    # Sending is deliberately out of scope: an approved draft is handed to
    # the account owner's mailbox / CRM by a separate integration.
    return {"id": d.id, "status": d.status}


@app.get("/feedback/reasons")
def feedback_reasons():
    return REASONS


@app.get("/feedback")
def feedback_log(action: str = "", session: Session = Depends(db)):
    """What the writer will learn from on its next run, plus the reason breakdown."""
    fb = writer_feedback(session, action)
    reasons = dict(session.execute(select(ReviewEvent.reason, func.count())
                                   .where(ReviewEvent.reason.is_not(None))
                                   .group_by(ReviewEvent.reason)).all())
    return {"approved_examples": fb.examples, "corrections": fb.lessons,
            "reason_counts": reasons, "prompt_preview": prompt_block(fb)}


@app.get("/feedback/export")
def feedback_export(session: Session = Depends(db)):
    """Training dataset: one record per human decision (JSON)."""
    return training_records(session)


@app.get("/trust")
def trust(session: Session = Depends(db)):
    """Trust metrics: review quality, flagged vs. churned, contacted vs. control, model card."""
    from copilot.trust import trust_summary
    return trust_summary(session, get_settings().model_dir)


@app.get("/metrics")
def metrics(session: Session = Depends(db)):
    model_file = get_settings().model_dir / "metrics.json"
    decisions = dict(session.execute(select(ReviewEvent.decision, func.count())
                                     .group_by(ReviewEvent.decision)).all())
    reviewed = sum(decisions.values())
    avg_secs = session.scalar(select(func.avg(ReviewEvent.seconds_to_decision)))
    return {
        "model": json.loads(model_file.read_text()) if model_file.exists() else None,
        "review": {"decisions": decisions,
                   "approval_rate": round(decisions.get("approved", 0) / reviewed, 3) if reviewed else None,
                   "avg_hours_flag_to_decision": round(avg_secs / 3600, 2) if avg_secs else None},
        "queue": dict(session.execute(select(Draft.status, func.count()).group_by(Draft.status)).all()),
    }
