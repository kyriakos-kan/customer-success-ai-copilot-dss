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

    assert c.get("/feedback", params={"action": draft["action"]}).status_code == 422
    fb = c.get("/feedback", params={"account_id": second["account_id"],
                                    "action": second["action"]}).json()
    assert fb["approved_examples"][0]["body"] == edited
    assert fb["corrections"] == [] and fb["reason_counts"] == {}
    rejected_fb = c.get("/feedback", params={"account_id": draft["account_id"],
                                             "action": draft["action"]}).json()
    assert rejected_fb["approved_examples"] == []
    assert rejected_fb["corrections"][0]["reason"] == "tone"
    assert rejected_fb["reason_counts"] == {"tone": 1}
    assert "APPROVED" in fb["prompt_preview"]
    assert "lead with their shrinkage issue" not in fb["prompt_preview"]
    assert "lead with their shrinkage issue" in rejected_fb["prompt_preview"]

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
            import json
            if "strict evidence verifier" in system:
                return json.dumps({"draft_supported": True,
                                   "claim_checks": [{"claim_index": 0, "supported": True}]})
            captured["prompt"] = prompt
            doc = prompt.split('<doc id="')[1].split('"')[0]
            return json.dumps({
                "status": "OK", "subject": "S", "body": "Complaint about shrinkage.",
                "claims": [{"text": "Complaint about shrinkage.",
                            "evidence": [{"doc_id": doc, "quote": "Complaint about shrinkage."}]}],
            })

    w = Spy(None, "m", backend="ollama")
    fb = WriterFeedback(
        account_id="X",
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
    assert "Required JSON schema" in captured["prompt"]
    assert d.feedback_used == {"approved_examples": [7], "lessons": [3]}


def test_writer_ignores_feedback_from_another_account():
    import json
    from copilot.feedback import WriterFeedback

    captured = {}

    class Spy(WriterAgent):
        def _complete(self, system, prompt):
            if "strict evidence verifier" in system:
                return json.dumps({"draft_supported": True,
                                   "claim_checks": [{"claim_index": 0, "supported": True}]})
            captured["prompt"] = prompt
            doc = prompt.split('<doc id="')[1].split('"')[0]
            return json.dumps({
                "status": "OK", "subject": "S", "body": "Complaint about shrinkage.",
                "claims": [{"text": "Complaint about shrinkage.",
                            "evidence": [{"doc_id": doc, "quote": "Complaint about shrinkage."}]}],
            })

    feedback = WriterFeedback(
        account_id="OTHER",
        examples=[{"id": 88, "action": "escalate_quality_review", "subject": "Private subject",
                   "body": "Other customer's confidential detail."}],
        lessons=[{"id": 89, "action": "escalate_quality_review", "reason": "other",
                  "label": "Other", "explanation": "Other customer's private comment."}],
    )
    assessment = AccountAssessment("X", 1000, 0.8, 800, "high", [])
    recommendation = Recommendation("r", "escalate_quality_review", "call", "am", "why")
    context = [{"text": "Complaint about shrinkage.",
                "metadata": {"doc_id": "X-0", "doc_type": "ticket"}}]

    draft = Spy(None, "m", backend="ollama").draft(
        "Acme (DE)", assessment, recommendation, context, feedback=feedback)

    assert "Other customer's confidential detail" not in captured["prompt"]
    assert "Other customer's private comment" not in captured["prompt"]
    assert draft.feedback_used is None


def test_writer_feedback_only_returns_rows_for_requested_account():
    from uuid import uuid4

    from copilot.db import Account, Draft, ReviewEvent, SessionLocal, init_db
    from copilot.feedback import writer_feedback

    init_db()
    account_id = f"SCOPE-{uuid4().hex[:12]}"
    other_account_id = f"SCOPE-{uuid4().hex[:12]}"
    action = "escalate_quality_review"

    with SessionLocal() as session:
        session.add_all([
            Account(account_id=account_id, name="Scope account", segment="loyal",
                    supplier_origin="GR", features={}),
            Account(account_id=other_account_id, name="Other account", segment="loyal",
                    supplier_origin="GR", features={}),
        ])
        session.flush()
        own_approved = Draft(run_id="feedback-scope", account_id=account_id, action=action,
                             action_rationale="test", subject="Own approved", body="Own example",
                             status="approved")
        own_rejected = Draft(run_id="feedback-scope", account_id=account_id, action=action,
                             action_rationale="test", subject="Own rejected", body="Own original",
                             status="rejected")
        other_approved = Draft(run_id="feedback-scope", account_id=other_account_id, action=action,
                               action_rationale="test", subject="Other approved", body="SECRET EXAMPLE",
                               status="approved")
        other_rejected = Draft(run_id="feedback-scope", account_id=other_account_id, action=action,
                               action_rationale="test", subject="Other rejected", body="Other original",
                               status="rejected")
        session.add_all([own_approved, own_rejected, other_approved, other_rejected])
        session.flush()
        session.add_all([
            ReviewEvent(draft_id=own_approved.id, reviewer="test", decision="approved"),
            ReviewEvent(draft_id=own_rejected.id, reviewer="test", decision="rejected",
                        reason="tone", comment="Own account comment."),
            ReviewEvent(draft_id=other_approved.id, reviewer="test", decision="approved"),
            ReviewEvent(draft_id=other_rejected.id, reviewer="test", decision="rejected",
                        reason="other", comment="OTHER ACCOUNT SECRET COMMENT."),
        ])
        session.commit()

        feedback = writer_feedback(session, action, account_id)

    assert feedback.account_id == account_id
    assert [example["body"] for example in feedback.examples] == ["Own example"]
    assert [lesson["explanation"] for lesson in feedback.lessons] == ["Own account comment."]


def test_writer_rejects_claim_not_supported_by_cited_document():
    import json

    class UnsupportedSpy(WriterAgent):
        def _complete(self, system, prompt):
            if "strict evidence verifier" in system:
                return json.dumps({"draft_supported": False,
                                   "claim_checks": [{"claim_index": 0, "supported": False}]})
            return json.dumps({
                "status": "OK", "subject": "Refund request", "body": "You asked us for a refund.",
                "claims": [{"text": "You asked us for a refund.",
                            "evidence": [{"doc_id": "X-0", "quote": "Complaint about shrinkage."}]}],
            })

    a = AccountAssessment("X", 1000, 0.8, 800, "high", [])
    rec = Recommendation("r", "escalate_quality_review", "call", "am", "why")
    ctx = [{"text": "Complaint about shrinkage.",
            "metadata": {"doc_id": "X-0", "doc_type": "ticket"}}]

    draft = UnsupportedSpy(None, "m", backend="ollama").draft("Acme (DE)", a, rec, ctx)

    assert draft.status == "pending_review"
    assert "refund" not in draft.body.lower()
    assert draft.context_used[0]["id"] == "X-0"


def test_writer_rejects_evidence_quote_not_in_retrieved_document():
    import json

    class InvalidQuoteSpy(WriterAgent):
        calls = 0

        def _complete(self, system, prompt):
            self.calls += 1
            return json.dumps({
                "status": "OK", "subject": "Refund request", "body": "You asked us for a refund.",
                "claims": [{"text": "You asked us for a refund.",
                            "evidence": [{"doc_id": "X-0", "quote": "They requested a refund."}]}],
            })

    a = AccountAssessment("X", 1000, 0.8, 800, "high", [])
    rec = Recommendation("r", "escalate_quality_review", "call", "am", "why")
    ctx = [{"text": "Complaint about shrinkage.",
            "metadata": {"doc_id": "X-0", "doc_type": "ticket"}}]
    writer = InvalidQuoteSpy(None, "m", backend="ollama")

    draft = writer.draft("Acme (DE)", a, rec, ctx)

    assert writer.calls == 1
    assert draft.status == "pending_review"
    assert "refund" not in draft.body.lower()


def test_writer_matches_normalized_claim_in_subject_and_exempts_body_conventions():
    import json

    class NormalizedSpy(WriterAgent):
        def _complete(self, system, prompt):
            if "strict evidence verifier" in system:
                return json.dumps({"draft_supported": True,
                                   "claim_checks": [{"claim_index": 0, "supported": True}]})
            return json.dumps({
                "status": "OK",
                "subject": "THE CUSTOMER SAID “THE BATCH SHRANK” AFTER DELIVERY",
                "body": "Dear Acme team,\n\nWould you have 20 minutes for a call?\n\n"
                        "Kind regards,\n[Account owner]\n[Company name]",
                "claims": [{
                    "text": 'The customer said "the batch shrank" after delivery',
                    "evidence": [{
                        "doc_id": "X-0",
                        "quote": 'THE   CUSTOMER SAID "THE BATCH SHRANK" AFTER DELIVERY.',
                    }],
                }],
            })

    a = AccountAssessment("X", 1000, 0.8, 800, "high", [])
    rec = Recommendation("r", "escalate_quality_review", "call", "am", "why")
    ctx = [{"text": "The customer said “the batch shrank” after delivery.",
            "metadata": {"doc_id": "X-0", "doc_type": "ticket"}}]

    draft = NormalizedSpy(None, "m", backend="ollama").draft("Acme (DE)", a, rec, ctx)

    assert draft.status == "pending_review"
    assert draft.subject.startswith("THE CUSTOMER")
    assert draft.body.startswith("Dear Acme team")


def test_writer_rejects_body_sentence_with_claim_missing_from_claim_list():
    import json

    class UnlistedClaimSpy(WriterAgent):
        verifier_calls = 0

        def _complete(self, system, prompt):
            if "strict evidence verifier" in system:
                self.verifier_calls += 1
                return json.dumps({"draft_supported": True,
                                   "claim_checks": [{"claim_index": 0, "supported": True}]})
            return json.dumps({
                "status": "OK",
                "subject": "Quality follow-up",
                "body": "Complaint about shrinkage. Please call us, as the customer requested a refund.",
                "claims": [{"text": "Complaint about shrinkage.",
                            "evidence": [{"doc_id": "X-0", "quote": "Complaint about shrinkage."}]}],
            })

    a = AccountAssessment("X", 1000, 0.8, 800, "high", [])
    rec = Recommendation("r", "escalate_quality_review", "call", "am", "why")
    ctx = [{"text": "Complaint about shrinkage.",
            "metadata": {"doc_id": "X-0", "doc_type": "ticket"}}]
    writer = UnlistedClaimSpy(None, "m", backend="ollama")

    draft = writer.draft("Acme (DE)", a, rec, ctx)

    assert writer.verifier_calls == 0
    assert draft.status == "pending_review"
    assert "refund" not in draft.body.lower()


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
