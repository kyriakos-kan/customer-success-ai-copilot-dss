"""Run with:  PYTHONPATH=src pytest -q   (uses SQLite + in-memory RAG, no Docker needed)"""
import itertools
import os
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp())
os.environ.update(DATABASE_URL=f"sqlite:///{TMP/'t.db'}", MODEL_DIR=str(TMP / "models"))
os.environ.pop("CHROMA_HOST", None)
os.environ.pop("ANTHROPIC_API_KEY", None)

from fastapi.testclient import TestClient  # noqa: E402

from copilot.agents.analyst import AccountAssessment  # noqa: E402
from copilot.agents.nba import Recommendation, recommend_next_best_action  # noqa: E402
from copilot.agents.writer import WriterAgent  # noqa: E402
from copilot.api.main import app  # noqa: E402
from copilot.pipeline.orchestrator import run_once  # noqa: E402

PLAYBOOK = Path("playbook/nba_playbook.yaml")


def test_playbook_covers_every_combination():
    for risk, seg in itertools.product(["high", "medium", "low"],
                                       ["at_risk", "needs_attention", "loyal", "champion"]):
        assert recommend_next_best_action(risk, seg, [], PLAYBOOK).action


def test_factor_rule_beats_segment_rule():
    factors = [{"feature": "quality_complaints_12m", "direction": "increases risk"}]
    assert recommend_next_best_action("high", "champion", factors, PLAYBOOK).action == "escalate_quality_review"


def test_writer_refuses_without_context():
    a = AccountAssessment("X", 1000, 0.8, 800, "high", [])
    rec = Recommendation("r", "schedule_call", "call", "am", "why")
    d = WriterAgent(None, "m").draft("Acme (DE)", a, rec, context=[])
    assert d.status == "insufficient_data" and d.context_used == []


def test_end_to_end_and_review_flow():
    result = run_once(force_retrain=True)
    assert result["scored"] > 0 and result["drafted"] > 0

    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok"}
    accounts = c.get("/accounts").json()
    aid = accounts[0]["account_id"]
    assert accounts[0]["revenue_at_risk"] >= accounts[-1]["revenue_at_risk"]
    assert c.get(f"/accounts/{aid}/risk").status_code == 200
    assert len(c.get(f"/accounts/{aid}/explanation").json()["top_factors"]) == 5
    assert c.get(f"/accounts/{aid}/next-best-action").json()["action"]

    queue = c.get("/drafts").json()
    draft, second = queue[0], queue[1]
    assert draft["context_used"], "drafts must be grounded in retrieved context"

    # Feedback rule: a rejection needs a reason AND an explanation
    no_reason = c.post(f"/drafts/{draft['id']}/review",
                       json={"reviewer": "test", "decision": "rejected", "explanation": "Too formal for them"})
    assert no_reason.status_code == 422
    no_expl = c.post(f"/drafts/{draft['id']}/review",
                     json={"reviewer": "test", "decision": "rejected", "reason": "tone"})
    assert no_expl.status_code == 422
    r = c.post(f"/drafts/{draft['id']}/review",
               json={"reviewer": "test", "decision": "rejected", "reason": "tone",
                     "explanation": "Too formal for this buyer; lead with their shrinkage issue."})
    assert r.json()["status"] == "rejected"
    assert c.post(f"/drafts/{draft['id']}/review",
                  json={"reviewer": "test", "decision": "approved"}).status_code == 409

    # Approval with an edit: the edited text becomes the approved example
    edited = second["body"] + "\n\nP.S. Happy to visit you in person."
    assert c.post(f"/drafts/{second['id']}/review",
                  json={"reviewer": "test", "decision": "approved", "edited_body": edited}).status_code == 200

    m = c.get("/metrics").json()
    assert m["review"]["decisions"] == {"rejected": 1, "approved": 1} and m["model"]["churn_roc_auc"] > 0.6

    fb = c.get("/feedback", params={"action": draft["action"]}).json()
    assert fb["approved_examples"][0]["body"] == edited
    assert fb["corrections"][0]["reason"] == "tone" and fb["reason_counts"] == {"tone": 1}
    assert "APPROVED" in fb["prompt_preview"] and "lead with their shrinkage issue" in fb["prompt_preview"]

    export = c.get("/feedback/export").json()
    pos = next(x for x in export if x["label"] == "positive")
    neg = next(x for x in export if x["label"] == "negative")
    assert pos["was_edited"] and pos["final_body"] == edited and pos["original_body"] == second["body"]
    assert neg["reason"] == "tone" and neg["original_body"] == draft["body"]

    # Next run: pending drafts are skipped, the approved account is not contacted again,
    # and the rejected account gets a new draft
    nxt = run_once()
    assert nxt["skipped"] > 0 and nxt["recently_contacted"] == 1
    redrafted = [d for d in c.get("/drafts").json() if d["account_id"] == draft["account_id"]]
    assert len(redrafted) == 1 and redrafted[0]["id"] != draft["id"]


def test_writer_prompt_contains_feedback():
    """The LLM prompt carries approved examples and reviewer corrections."""
    from copilot.feedback import WriterFeedback
    captured = {}

    class Spy(WriterAgent):
        def _complete(self, system, prompt):
            captured["prompt"] = prompt
            doc = prompt.split('<doc id="')[1].split('"')[0]
            import json
            return json.dumps({"status": "OK", "subject": "S", "body": "B", "cited_ids": [doc]})

    w = Spy(None, "m", backend="ollama")
    fb = WriterFeedback(
        examples=[{"id": 7, "action": "escalate_quality_review", "subject": "Your last batches",
                   "body": "Dear team, we checked the lab reports..."}],
        lessons=[{"id": 3, "action": "escalate_quality_review", "reason": "tone",
                  "label": "Tone not right", "explanation": "Too formal; be warmer."}])
    a = AccountAssessment("X", 1000, 0.8, 800, "high", [])
    rec = Recommendation("r", "escalate_quality_review", "call", "am", "why")
    ctx = [{"text": "Complaint about shrinkage.", "metadata": {"doc_id": "X-0", "doc_type": "ticket"}}]
    d = w.draft("Acme (DE)", a, rec, ctx, feedback=fb)
    assert "we checked the lab reports" in captured["prompt"]
    assert "Too formal; be warmer." in captured["prompt"]
    assert d.feedback_used == {"approved_examples": [7], "lessons": [3]}


def test_trust_metrics_control_group_and_outcomes():
    """Control accounts get no draft; outcomes feed flagged-vs-churned and retention."""
    from copilot.config import get_settings
    from copilot.db import Draft, Outcome, ReviewEvent, RiskScore, SessionLocal
    from copilot.trust import (in_control_group, model_card, prediction_check,
                               retention_comparison, review_metrics)

    assert in_control_group("ACC-0001", 20) == in_control_group("ACC-0001", 20)   # fixed per account
    assert not in_control_group("ACC-0001", 0)
    share = sum(in_control_group(f"ACC-{i:04d}", 20) for i in range(2000)) / 2000
    assert 0.15 < share < 0.25

    settings = get_settings()
    settings.control_group_pct = 20
    try:
        run_once()
    finally:
        settings.control_group_pct = 0
    with SessionLocal() as s:
        control = set(s.scalars(RiskScore.__table__.select().with_only_columns(RiskScore.account_id)
                                .where(RiskScore.cohort == "control")).all())
        assert control
        run_id = s.scalar(RiskScore.__table__.select().with_only_columns(RiskScore.run_id)
                          .where(RiskScore.cohort == "control"))
        new_drafts = set(s.scalars(Draft.__table__.select().with_only_columns(Draft.account_id)
                                   .where(Draft.run_id == run_id)).all())
        assert not (control & new_drafts)                       # nobody in control was drafted

        # one approved (contacted) account and one control account, with outcomes
        contacted = s.scalar(Draft.__table__.select().with_only_columns(Draft.account_id)
                             .join(ReviewEvent, ReviewEvent.draft_id == Draft.id)
                             .where(ReviewEvent.decision == "approved"))
        ctrl = sorted(control - {contacted})[0]
        s.add_all([Outcome(account_id=contacted, period_start="2026-04-01", period_end="2026-10-01", churned=0),
                   Outcome(account_id=ctrl, period_start="2026-04-01", period_end="2026-10-01", churned=1)])
        s.commit()

        rv = review_metrics(s)
        assert rv["decisions"] >= 1 and 0 <= rv["approval_rate"] <= 1
        pc = prediction_check(s)
        assert pc["source"] == "outcomes" and sum(r["accounts"] for r in pc["rows"]) == 2
        rc = retention_comparison(s)
        by = {c["cohort"]: c for c in rc["cohorts"]}
        assert by["contacted"]["retention_rate"] == 1.0 and by["control"]["retention_rate"] == 0.0
        assert rc["uplift"] == 1.0

    card = model_card(settings.model_dir)
    assert card["model_version"].startswith("xgb-") and card["age_days"] == 0
    assert client_get_trust()["model"]["model_version"] == card["model_version"]


def client_get_trust():
    r = TestClient(app).get("/trust")
    assert r.status_code == 200
    return r.json()
