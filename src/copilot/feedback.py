"""Feedback log: how human review decisions train the writer agent.

1. Approved drafts become examples. The final text a reviewer approved (including their
   edits) is shown to the writer as a model of a good draft for the same kind of action.
2. Rejected / needs-edit drafts must come with a reason code and an explanation. These are
   shown to the writer as corrections to avoid, and exported as a training dataset.

Everything goes through `record_review()`, so the API, the laptop demo and the notebooks
apply exactly the same rules.

    python -m copilot.feedback export feedback.jsonl     # training dataset (JSON Lines)
"""
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from copilot.db import Draft, ReviewEvent

# Reason codes for rejected / needs-edit drafts: code -> label shown to reviewers
REASONS: dict[str, str] = {
    "factual_error": "Factually wrong or unsupported claim",
    "wrong_action": "Wrong action for this customer",
    "tone": "Tone not right (too formal / too casual / too pushy)",
    "too_long": "Too long or unclear",
    "missing_context": "Misses something important about the customer",
    "internal_info": "Reveals internal information (risk score, models, notes)",
    "other": "Other (explain below)",
}
DECISIONS = ("approved", "needs_edit", "rejected")
MIN_EXPLANATION = 10          # characters; a reason needs a real explanation


class FeedbackError(ValueError):
    """A review that breaks the feedback rules (e.g. a rejection without a reason)."""


def validate_review(decision: str, reason: str | None, explanation: str | None) -> None:
    if decision not in DECISIONS:
        raise FeedbackError(f"Unknown decision: {decision}")
    if decision in ("rejected", "needs_edit"):
        if reason not in REASONS:
            raise FeedbackError("Please choose a reason for rejecting or sending back this draft "
                                f"(valid codes: {', '.join(REASONS)}).")
        if not explanation or len(explanation.strip()) < MIN_EXPLANATION:
            raise FeedbackError(f"Please explain what was wrong (at least {MIN_EXPLANATION} "
                                "characters); the explanation trains the writer.")


def record_review(session: Session, draft_id: int, reviewer: str, decision: str,
                  reason: str | None = None, explanation: str | None = None,
                  edited_body: str | None = None) -> Draft:
    """Validate and store one human decision. Raises LookupError / FeedbackError."""
    validate_review(decision, reason, explanation)
    d = session.get(Draft, draft_id)
    if d is None:
        raise LookupError("Draft not found")
    if d.status != "pending_review":
        raise FeedbackError(f"Draft already {d.status}")
    created = d.created_at if d.created_at.tzinfo else d.created_at.replace(tzinfo=timezone.utc)
    original = d.body
    edited = edited_body if edited_body and edited_body.strip() != original.strip() else None
    if decision == "approved" and edited:
        d.body = edited                       # the approved, edited text is what gets sent
    d.status = decision
    session.add(ReviewEvent(
        draft_id=d.id, reviewer=reviewer, decision=decision,
        reason=reason if decision != "approved" else None,
        original_body=original, edited_body=edited,
        comment=(explanation or "").strip() or None,
        seconds_to_decision=(datetime.now(timezone.utc) - created).total_seconds()))
    session.commit()
    return d


# --------------------------------------------------------------- what the writer learns
@dataclass
class WriterFeedback:
    account_id: str
    examples: list[dict]          # [{"id", "action", "subject", "body"}]
    lessons: list[dict]           # [{"id", "action", "reason", "label", "explanation"}]

    @property
    def used(self) -> dict:
        return {"approved_examples": [e["id"] for e in self.examples],
                "lessons": [l["id"] for l in self.lessons]}

    def is_empty(self) -> bool:
        return not self.examples and not self.lessons


def writer_feedback(session: Session, action: str, account_id: str,
                    n_examples: int = 2, n_lessons: int = 5) -> WriterFeedback:
    """Return only this account's approved drafts and reviewer corrections."""
    approved = session.execute(
        select(Draft, ReviewEvent).join(ReviewEvent, ReviewEvent.draft_id == Draft.id)
        .where(ReviewEvent.decision == "approved", Draft.account_id == account_id)
        .order_by((Draft.action == action).desc(), ReviewEvent.created_at.desc())
        .limit(n_examples)).all()
    examples = [{"id": d.id, "action": d.action, "subject": d.subject, "body": d.body}
                for d, _ in approved]

    corrections = session.execute(
        select(ReviewEvent, Draft).join(Draft, ReviewEvent.draft_id == Draft.id)
        .where(ReviewEvent.decision.in_(["rejected", "needs_edit"]),
             ReviewEvent.comment.is_not(None), Draft.account_id == account_id)
        .order_by((Draft.action == action).desc(), ReviewEvent.created_at.desc())
        .limit(n_lessons)).all()
    lessons = [{"id": e.id, "action": d.action, "reason": e.reason,
                "label": REASONS.get(e.reason or "other", "Other"), "explanation": e.comment}
               for e, d in corrections]
    return WriterFeedback(account_id, examples, lessons)


def prompt_block(fb: WriterFeedback) -> str:
    """The text added to the writer's prompt. Also shown in the demo, so it's transparent."""
    if fb.is_empty():
        return ""
    parts = []
    if fb.examples:
        parts.append("Drafts that reviewers APPROVED for similar cases. Match their tone, length and "
                     "structure, but never copy their facts; use only the facts in <context>:")
        for i, e in enumerate(fb.examples, 1):
            parts.append(f'<approved_example n="{i}" action="{e["action"]}">\n'
                         f'Subject: {e["subject"]}\n{e["body"]}\n</approved_example>')
    if fb.lessons:
        parts.append("Reviewer CORRECTIONS on past drafts. Do not repeat these mistakes:")
        parts += [f'- [{l["label"]}] {l["explanation"]}' for l in fb.lessons]
    return "\n".join(parts)


# --------------------------------------------------------------- training dataset
def training_records(session: Session) -> list[dict]:
    """One record per decision, ready for evaluation or fine-tuning the writer later.
    `label` is "positive" for approved drafts (use final_body as the target) and
    "negative" for rejected / needs-edit drafts (with reason and explanation)."""
    rows = session.execute(select(ReviewEvent, Draft).join(Draft, ReviewEvent.draft_id == Draft.id)
                           .order_by(ReviewEvent.created_at)).all()
    out = []
    for e, d in rows:
        original = e.original_body or (d.body if e.decision != "approved" or not e.edited_body else None)
        out.append({
            "event_id": e.id, "draft_id": d.id, "account_id": d.account_id,
            "decided_at": e.created_at.isoformat() if e.created_at else None,
            "reviewer": e.reviewer, "decision": e.decision,
            "label": "positive" if e.decision == "approved" else "negative",
            "reason": e.reason, "reason_label": REASONS.get(e.reason) if e.reason else None,
            "explanation": e.comment,
            "action": d.action, "action_rationale": d.action_rationale,
            "context": [c.get("text") for c in (d.context_used or [])],
            "subject": d.subject,
            "original_body": original,
            "final_body": (e.edited_body or original) if e.decision == "approved" else e.edited_body,
            "was_edited": bool(e.edited_body),
        })
    return out


def export_jsonl(session: Session, path: str | Path) -> int:
    records = training_records(session)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "export":
        from copilot.db import SessionLocal, init_db
        init_db()
        with SessionLocal() as s:
            print(f"Exported {export_jsonl(s, sys.argv[2])} decisions to {sys.argv[2]}")
    else:
        print(__doc__)
