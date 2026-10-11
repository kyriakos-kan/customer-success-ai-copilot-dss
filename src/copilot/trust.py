"""Trust metrics: how the copilot proves itself over time, not just on day one.

Four checks, shared by the API (/trust) and the laptop demo:

1. Human review   - approval rate and the most common rejection reasons.
2. Predictions    - accounts the model flagged vs. accounts that actually churned.
3. Impact         - retention of contacted accounts vs. the control group.
4. Model card     - which model version is live and when it was last retrained.

Checks 2 and 3 need outcomes, which only exist after the fact. Import them from the ERP:

    python -m copilot.trust import-outcomes outcomes.csv
    # columns: account_id, period_start, period_end, churned (1/0), revenue (optional)
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from copilot.db import Draft, Outcome, ReviewEvent, RiskScore
from copilot.feedback import REASONS

RETRAIN_WARNING_DAYS = 90       # model card turns amber when it is older than this


# --------------------------------------------------------------- control group
def in_control_group(account_id: str, pct: int) -> bool:
    """Random but fixed per account: the same account always lands in the same group."""
    if pct <= 0:
        return False
    return int(hashlib.sha256(account_id.encode()).hexdigest(), 16) % 100 < pct


# --------------------------------------------------------------- 1. human review
def review_metrics(session: Session, top_n: int = 3) -> dict:
    decisions = dict(session.execute(select(ReviewEvent.decision, func.count())
                                     .group_by(ReviewEvent.decision)).all())
    total = sum(decisions.values())
    reasons = session.execute(
        select(ReviewEvent.reason, func.count()).where(ReviewEvent.decision != "approved")
        .group_by(ReviewEvent.reason).order_by(func.count().desc())).all()
    return {
        "decisions": total,
        "approved": decisions.get("approved", 0),
        "needs_edit": decisions.get("needs_edit", 0),
        "rejected": decisions.get("rejected", 0),
        "approval_rate": decisions.get("approved", 0) / total if total else None,
        "top_reasons": [{"reason": r or "other", "label": REASONS.get(r or "other", "Other"), "count": n}
                        for r, n in reasons[:top_n]],
    }


# --------------------------------------------------------------- helpers
def _latest_scores(session: Session) -> dict[str, RiskScore]:
    latest = (select(RiskScore.account_id, func.max(RiskScore.id).label("id"))
              .group_by(RiskScore.account_id).subquery())
    return {r.account_id: r for r in
            session.scalars(select(RiskScore).join(latest, RiskScore.id == latest.c.id))}


def _outcomes(session: Session) -> dict[str, int]:
    """Latest known outcome per account (1 = churned)."""
    latest = (select(Outcome.account_id, func.max(Outcome.id).label("id"))
              .group_by(Outcome.account_id).subquery())
    return {o.account_id: int(o.churned) for o in
            session.scalars(select(Outcome).join(latest, Outcome.id == latest.c.id))}


# --------------------------------------------------------------- 2. predictions
def prediction_check(session: Session, all_account_ids: list[str] | None = None,
                     labels: dict[str, int] | None = None) -> dict:
    """Flagged vs. actually churned, per risk level, plus 'not flagged' as the baseline.

    Uses imported outcomes when there are any. `labels` is a fallback (e.g. the synthetic
    dataset's own churn labels in the demo); the result says which source was used."""
    outcomes = _outcomes(session)
    source = "outcomes"
    if not outcomes and labels:
        # Labels the model may have been trained on: only count accounts listed in
        # all_account_ids (pass the held-out accounts), or the check grades itself.
        outcomes, source = labels, "synthetic_labels"
        if all_account_ids is not None:
            keep = set(all_account_ids)
            outcomes = {a: v for a, v in outcomes.items() if a in keep}
    if not outcomes:
        return {"source": None, "rows": [], "precision_high": None, "recall_high": None}

    scores = _latest_scores(session)
    groups: dict[str, list[int]] = {"high": [], "medium": [], "low": [], "not_flagged": []}
    for acc, churned in outcomes.items():
        if acc in scores:
            groups[scores[acc].risk_level].append(churned)
        elif all_account_ids is None or acc in set(all_account_ids):
            groups["not_flagged"].append(churned)
    rows = [{"group": g, "accounts": len(v), "churned": sum(v),
             "churn_rate": sum(v) / len(v) if v else None} for g, v in groups.items()]
    total_churned = sum(r["churned"] for r in rows)
    high = groups["high"]
    return {
        "source": source, "rows": rows,
        "precision_high": sum(high) / len(high) if high else None,
        "recall_high": sum(high) / total_churned if total_churned else None,
    }


# --------------------------------------------------------------- 3. impact
def retention_comparison(session: Session) -> dict:
    """Contacted (an approved outreach) vs. control group (flagged, no draft).

    Retention is only reported where outcomes exist; until then the cohort sizes show
    the experiment is running."""
    scores = _latest_scores(session)
    contacted = set(session.scalars(
        select(Draft.account_id).join(ReviewEvent, ReviewEvent.draft_id == Draft.id)
        .where(ReviewEvent.decision == "approved")).all())
    control = {a for a, s in scores.items() if s.cohort == "control"} - contacted
    outcomes = _outcomes(session)

    def cohort(name: str, ids: set[str]) -> dict:
        known = [outcomes[a] for a in ids if a in outcomes]
        return {"cohort": name, "accounts": len(ids), "with_outcome": len(known),
                "retained": len(known) - sum(known),
                "retention_rate": (len(known) - sum(known)) / len(known) if known else None}

    c, k = cohort("contacted", contacted), cohort("control", control)
    uplift = (c["retention_rate"] - k["retention_rate"]
              if c["retention_rate"] is not None and k["retention_rate"] is not None else None)
    return {"cohorts": [c, k], "uplift": uplift, "has_outcomes": bool(outcomes)}


# --------------------------------------------------------------- 4. model card
def holdout_accounts(model_dir: Path) -> list[str] | None:
    """Accounts the current model never saw in training (saved at training time)."""
    f = Path(model_dir) / "holdout_accounts.json"
    return json.loads(f.read_text()) if f.exists() else None


def model_card(model_dir: Path) -> dict | None:
    f = Path(model_dir) / "metrics.json"
    if not f.exists():
        return None
    m = json.loads(f.read_text())
    trained = m.get("trained_at") or datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat(
        timespec="seconds")
    age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(trained)).days
    return {
        "model_version": m.get("model_version", "unversioned (retrain to stamp a version)"),
        "trained_at": trained, "age_days": age_days,
        "needs_retraining": age_days > RETRAIN_WARNING_DAYS,
        "n_train": m.get("n_train"), "n_test": m.get("n_test"),
        "churn_roc_auc": m.get("churn_roc_auc"),
        "churn_precision@high": m.get("churn_precision@high"),
        "churn_recall@high": m.get("churn_recall@high"),
    }


def trust_summary(session: Session, model_dir: Path) -> dict:
    return {"review": review_metrics(session), "predictions": prediction_check(session),
            "impact": retention_comparison(session), "model": model_card(model_dir)}


# --------------------------------------------------------------- outcomes import
def import_outcomes(session: Session, path: str | Path, source: str = "erp") -> int:
    """CSV with account_id, period_start, period_end, churned (1/0), revenue (optional)."""
    n = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            session.add(Outcome(account_id=row["account_id"].strip(),
                                period_start=row["period_start"], period_end=row["period_end"],
                                churned=int(str(row["churned"]).strip() in ("1", "true", "True", "yes")),
                                revenue=float(row["revenue"]) if row.get("revenue") else None,
                                source=source))
            n += 1
    session.commit()
    return n


def _cli() -> None:
    from copilot.db import SessionLocal, init_db
    ap = argparse.ArgumentParser(description="Trust metrics utilities")
    sub = ap.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import-outcomes", help="Load observed outcomes from a CSV export")
    imp.add_argument("csv")
    args = ap.parse_args()
    init_db()
    with SessionLocal() as s:
        print(f"Imported {import_outcomes(s, args.csv)} outcomes")


if __name__ == "__main__":
    _cli()
