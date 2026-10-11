"""Customer Success AI Copilot — laptop demo.

One process, no Docker, no API server: this app imports the same `copilot`
package the containers run and stores everything in a local SQLite file.

    streamlit run demo/app.py        (from the project root, or use run_demo.ps1)
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# --- Point at the project package and a local demo database BEFORE importing copilot
def _project_root() -> Path:
    """Project folder, whether started with `streamlit run` (has __file__) or pasted elsewhere."""
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for folder in (start, *start.parents):
        if (folder / "src" / "copilot").exists():
            return folder
    raise RuntimeError("Run this from inside the cs_ai_copilot_dss folder: streamlit run demo/app.py")


ROOT = _project_root()
DATA = ROOT / "demo" / "_data"
DATA.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)                                   # playbook path is relative to the root
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{(DATA / 'demo.db').as_posix()}")
os.environ.setdefault("MODEL_DIR", str(DATA / "models"))
os.environ.pop("CHROMA_HOST", None)              # in-memory RAG for the demo
os.environ.setdefault("CONTROL_GROUP_PCT", "20")   # demo shows the pilot design: 20% held back

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from sqlalchemy import func, select  # noqa: E402

from copilot.config import get_settings  # noqa: E402
from copilot.db import (Account, Draft, ReviewEvent, RiskScore,  # noqa: E402
                        SessionLocal, init_db)
from copilot.feedback import (REASONS, FeedbackError, prompt_block,  # noqa: E402
                              training_records, writer_feedback)
from copilot.feedback import record_review as _record_review  # noqa: E402
from copilot.pipeline.orchestrator import run_once  # noqa: E402
from copilot.trust import (holdout_accounts, model_card,  # noqa: E402
                           prediction_check, retention_comparison, review_metrics)

st.set_page_config(page_title="CS Copilot · B2B textile exporter (demo)", page_icon="🧵", layout="wide")

ACTION_LABEL = {
    "escalate_quality_review": "Escalate quality review", "offer_payment_plan_review": "Review payment terms",
    "propose_stock_programme": "Propose stock programme", "executive_call": "Executive call",
    "schedule_call": "Schedule a call", "send_reorder_incentive": "Send reorder incentive",
    "renewal_offer": "Renewal offer", "check_in_email": "Check-in email", "monitor": "Keep monitoring",
}
RISK_COLOR = {"high": "#c0392b", "medium": "#d68910", "low": "#1e8449"}


# ----------------------------------------------------------------- data access
@st.cache_resource(show_spinner="Running the copilot pipeline for the first time…")
def bootstrap() -> dict:
    """Run monitoring → analyst → RAG → playbook → writer once, if not done yet."""
    init_db()
    with SessionLocal() as s:
        if s.scalar(select(func.count()).select_from(RiskScore)):
            return {"status": "existing"}
    return run_once(force_retrain=True)


def latest_scores() -> pd.DataFrame:
    with SessionLocal() as s:
        latest = (select(RiskScore.account_id, func.max(RiskScore.id).label("id"))
                  .group_by(RiskScore.account_id).subquery())
        rows = s.execute(select(RiskScore, Account).join(latest, RiskScore.id == latest.c.id)
                         .join(Account, Account.account_id == RiskScore.account_id)).all()
    df = pd.DataFrame([{
        "account_id": r.account_id, "customer": a.name, "segment": a.segment,
        "origin": a.supplier_origin, "risk": r.risk_level, "churn": r.churn_probability,
        "clv": r.clv_score, "at_risk": r.churn_probability * r.clv_score,
        "signals": r.monitor_signals, "factors": r.top_factors,
    } for r, a in rows])
    return df.sort_values("at_risk", ascending=False).reset_index(drop=True) if len(df) else df


def drafts(status: str | None = None) -> pd.DataFrame:
    with SessionLocal() as s:
        q = select(Draft)
        if status:
            q = q.where(Draft.status == status)
        rows = s.scalars(q.order_by(Draft.id)).all()
    return pd.DataFrame([{c: getattr(d, c) for c in
                          ["id", "account_id", "action", "action_rationale", "subject", "body",
                           "context_used", "status", "feedback_used", "created_at"]} for d in rows])


def record_review(draft_id: int, decision: str, reviewer: str, body: str,
                  reason: str | None, explanation: str) -> str | None:
    """Store a decision through the shared feedback rules. Returns an error message or None."""
    try:
        with SessionLocal() as s:
            _record_review(s, draft_id, reviewer, decision, reason=reason,
                           explanation=explanation, edited_body=body)
    except (FeedbackError, LookupError) as exc:
        return str(exc)
    return None


def learned_caption(fb_used) -> str | None:
    if not isinstance(fb_used, dict):
        return None
    n_ex, n_l = len(fb_used.get("approved_examples", [])), len(fb_used.get("lessons", []))
    if not (n_ex or n_l):
        return None
    return f"✍️ Written with {n_ex} approved example(s) and {n_l} reviewer correction(s) from the feedback log."


def reset_demo() -> None:
    # Clear rows rather than deleting the file: Windows locks an open SQLite file.
    with SessionLocal() as s:
        for model in (ReviewEvent, Draft, RiskScore):
            s.query(model).delete()
        s.commit()
    st.cache_resource.clear()


@st.cache_data
def synthetic_labels() -> dict[str, int]:
    """The synthetic dataset's own churn labels: a stand-in until real outcomes are imported."""
    from copilot.data import load_gold
    g = load_gold()
    return dict(zip(g.account_id, g.churned_next_6m.astype(int)))


# ----------------------------------------------------------------- UI helpers
def euro(x: float) -> str:
    return f"€{x:,.0f}"


def risk_badge(level: str) -> str:
    return f"<span style='background:{RISK_COLOR[level]};color:white;padding:2px 10px;" \
           f"border-radius:10px;font-size:0.8em;font-weight:600'>{level.upper()}</span>"


def factor_chart(factors: list[dict]):
    f = pd.DataFrame(factors)
    def fmt(v):
        if not isinstance(v, (int, float)):
            return str(v)
        if float(v).is_integer() or abs(v) >= 100:
            return f"{v:,.0f}"
        return f"{v:.3g}"
    f["label"] = [f"{l}: {fmt(v)}" for l, v in zip(f["label"], f["value"])]
    f["effect"] = f["direction"].map({"increases risk": "Increases risk", "reduces risk": "Reduces risk"})
    return (alt.Chart(f).mark_bar(cornerRadius=3).encode(
        x=alt.X("shap_log_odds:Q", title="Contribution to churn risk"),
        y=alt.Y("label:N", sort=alt.EncodingSortField("shap_log_odds", op="sum", order="descending"),
                title=None, axis=alt.Axis(labelLimit=320, labelFontSize=12)),
        color=alt.Color("effect:N", scale=alt.Scale(domain=["Increases risk", "Reduces risk"],
                                                    range=["#c0392b", "#1e8449"]), legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["label", "direction", alt.Tooltip("shap_log_odds:Q", format=".3f")],
    ).properties(height=40 * len(f) + 30))


def demo_stories(scores: pd.DataFrame, d: pd.DataFrame) -> dict[str, str]:
    """Pick three accounts that tell the system's story well."""
    stories = {}
    pend = d[d.status == "pending_review"].merge(scores[["account_id", "at_risk"]], on="account_id")
    for action, title in [("escalate_quality_review", "Quality problems on a valuable account"),
                          ("offer_payment_plan_review", "Payment pressure building up")]:
        hit = pend[pend.action == action].sort_values("at_risk", ascending=False)
        if len(hit):
            stories[title] = hit.iloc[0].account_id
    held = d[d.status == "insufficient_data"].merge(scores[["account_id", "at_risk"]], on="account_id")
    if len(held):
        stories["Guardrail: not enough information"] = held.sort_values("at_risk", ascending=False).iloc[0].account_id
    return stories


# ----------------------------------------------------------------- app
result = bootstrap()
scores, all_drafts = latest_scores(), drafts()
settings = get_settings()

with st.sidebar:
    st.markdown("### 🧵 CS Copilot")
    st.caption("B2B textile exporter · demo")
    page = st.radio("View", ["Overview", "Account story", "Review queue", "Feedback log", "Results",
                             "Trust metrics"],
                    label_visibility="collapsed")
    st.divider()
    reviewer = st.text_input("Reviewer", "Reviewer")
    try:
        from copilot.agents.writer import WriterAgent
        _writer_label = WriterAgent(settings.anthropic_api_key, settings.writer_model).label
    except Exception as exc:                        # misconfigured backend: show it, don't crash
        _writer_label = f"misconfigured ({exc})"
    st.caption("Writer: " + _writer_label)
    st.caption("Data: synthetic, 2,000 B2B customers")
    if settings.control_group_pct:
        st.caption(f"Control group: {settings.control_group_pct}% of flagged accounts get no draft")
    if st.button("Run next cycle", help="Runs the pipeline again, as the nightly job would. "
                 "Rejected drafts are rewritten using the feedback log; approved accounts are skipped."):
        with st.spinner("Running monitoring → analyst → writer…"):
            st.session_state["last_cycle"] = run_once()
        st.rerun()
    if "last_cycle" in st.session_state:
        c = st.session_state["last_cycle"]
        st.caption(f"Last cycle: {c['drafted']} new drafts, {c['skipped']} still pending, "
                   f"{c.get('recently_contacted', 0)} recently approved.")
    if st.button("Reset demo", help="Clears decisions, the feedback log and drafts, then reruns the pipeline"):
        reset_demo()
        st.rerun()

# ============================================================== OVERVIEW
if page == "Overview":
    st.title("Customers at risk, before they leave")
    st.caption("Every night the copilot scans all accounts, scores the ones whose behaviour changed, "
               "explains why, and drafts outreach for the account manager to approve.")

    pending = (all_drafts.status == "pending_review").sum() if len(all_drafts) else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accounts monitored", "2,000")
    c2.metric("Flagged this cycle", f"{len(scores):,}")
    c3.metric("High risk", f"{(scores.risk == 'high').sum():,}")
    c4.metric("Revenue at risk (high)", euro(scores.loc[scores.risk == "high", "at_risk"].sum()))
    st.metric("Drafts waiting for review", int(pending))

    st.subheader("Top accounts by revenue at risk")
    top = scores.head(15).copy()
    top["churn"] = top["churn"] * 100
    top["clv"] = top["clv"].map(euro)
    top["at_risk"] = top["at_risk"].map(euro)
    top["top driver"] = top.factors.apply(
        lambda fs: next((f["label"] for f in fs if f["direction"] == "increases risk"), "—"))
    st.dataframe(
        top[["customer", "risk", "churn", "clv", "at_risk", "top driver", "origin"]],
        hide_index=True, use_container_width=True,
        column_config={
            "churn": st.column_config.ProgressColumn("Churn probability", min_value=0, max_value=100, format="%.0f%%"),
            "clv": "12-month CLV", "at_risk": "Revenue at risk", "top driver": "Top risk driver",
            "risk": "Risk", "customer": "Customer", "origin": "Supplier origin"})
    st.caption("Revenue at risk = churn probability × predicted 12-month value. "
               "This decides the order of the review queue.")

    st.subheader("How it works")
    cols = st.columns(5)
    for col, (title, text) in zip(cols, [
        ("1 · Watch", "Monitoring rules flag accounts whose behaviour shifted."),
        ("2 · Understand", "Models score churn risk and value, and explain the drivers."),
        ("3 · Recommend", "The playbook turns risk + drivers into a next best action."),
        ("4 · Draft", "The writer drafts outreach grounded in the customer's own history."),
        ("5 · Decide", "A person approves, edits or rejects. Nothing is sent automatically."),
    ]):
        col.markdown(f"**{title}**")
        col.caption(text)

# ============================================================== ACCOUNT STORY
elif page == "Account story":
    st.title("One account, end to end")
    stories = demo_stories(scores, all_drafts)
    options = {f"★ {t}": a for t, a in stories.items()}
    options |= {f"{r.customer} · {r.risk}": r.account_id for r in scores.head(40).itertuples()}
    choice = st.selectbox("Account", list(options), help="★ = suggested demo stories")
    aid = options[choice]
    row = scores.set_index("account_id").loc[aid]
    d = all_drafts[all_drafts.account_id == aid].tail(1)

    st.markdown(f"## {row.customer} &nbsp; {risk_badge(row.risk)}", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns([1, 1.2, 1.2, 1.1, 0.8])
    c1.metric("Churn probability", f"{row.churn:.0%}")
    c2.metric("Predicted 12-month value", euro(row.clv))
    c3.metric("Revenue at risk", euro(row.at_risk))
    c4.metric("RFM segment", row.segment.replace("_", " ").title())
    c5.metric("Supplier origin", row.origin)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### 1 · Why it was flagged")
        for s in row.signals:
            st.markdown(f"- {s}")
        st.markdown("#### 2 · What drives the risk")
        st.altair_chart(factor_chart(row.factors), use_container_width=True)
        st.caption("Each bar is the factor's contribution to this customer's churn score (SHAP).")
    with right:
        st.markdown("#### 3 · Recommended action")
        if len(d):
            dr = d.iloc[0]
            st.info(f"**{ACTION_LABEL.get(dr.action, dr.action)}**  \n{dr.action_rationale}")
            st.markdown("#### 4 · Draft for the account manager")
            if dr.status == "insufficient_data":
                st.warning(f"**No draft generated.** {dr.body}")
                st.caption("Guardrail: the writer refuses to invent details when there is no customer history to ground the message.")
            else:
                st.markdown(f"**Subject:** {dr.subject}")
                st.text_area("Draft", dr.body, height=230, disabled=True, label_visibility="collapsed")
                with st.expander("Customer history used to write this draft"):
                    for c in dr.context_used:
                        st.markdown(f"- **{c['doc_type'].replace('_', ' ')}**: {c['text']}")
                st.caption(f"Status: **{dr.status.replace('_', ' ')}**. Approve or edit it in the Review queue.")
                if learned_caption(dr.feedback_used):
                    st.caption(learned_caption(dr.feedback_used))
        else:
            st.info("**Keep monitoring.** No outreach is recommended this cycle.")

# ============================================================== REVIEW QUEUE
elif page == "Review queue":
    queue = drafts("pending_review").merge(scores, on="account_id").sort_values("at_risk", ascending=False)
    st.title(f"Review queue · {len(queue)} pending")
    st.caption("Highest revenue at risk first. Nothing leaves the system without a decision here.")
    if queue.empty:
        st.success("Queue is empty. See the Results page.")
    for q in queue.head(10).itertuples():
        with st.expander(f"{q.customer} · {ACTION_LABEL.get(q.action, q.action)} · {euro(q.at_risk)} at risk"):
            a, b = st.columns([3, 2])
            with b:
                st.markdown(risk_badge(q.risk), unsafe_allow_html=True)
                st.write(f"Churn {q.churn:.0%} · CLV {euro(q.clv)}")
                st.caption("Why flagged: " + "; ".join(q.signals))
                st.caption("Playbook: " + q.action_rationale)
                for c in q.context_used:
                    st.caption(f"📎 {c['doc_type'].replace('_', ' ')}: {c['text']}")
                if learned_caption(q.feedback_used):
                    st.caption(learned_caption(q.feedback_used))
            with a:
                st.markdown(f"**Subject:** {q.subject}")
                body = st.text_area("Draft", q.body, height=220, key=f"b{q.id}", label_visibility="collapsed")
                st.caption("Approve: your final text becomes an example for future drafts. "
                           "Needs edit / Reject: choose a reason and explain; this trains the writer.")
                reason = st.selectbox("Reason (required for Needs edit / Reject)",
                                      [None, *REASONS], key=f"r{q.id}",
                                      format_func=lambda k: "— choose a reason —" if k is None else REASONS[k])
                explanation = st.text_area("Explanation (required for Needs edit / Reject)", key=f"c{q.id}",
                                           height=70, placeholder="e.g. Too formal for this buyer; "
                                           "open with the shrinkage issue, not the contract.")
                x, y, z = st.columns(3)
                for col, (dec, label, kind) in zip((x, y, z), [("approved", "Approve", "primary"),
                                                              ("needs_edit", "Needs edit", "secondary"),
                                                              ("rejected", "Reject", "secondary")]):
                    if col.button(label, key=f"{dec}{q.id}", type=kind, use_container_width=True):
                        error = record_review(q.id, dec, reviewer, body, reason, explanation)
                        if error:
                            st.error(error)
                        else:
                            st.toast(f"{label}: {q.customer}")
                            st.rerun()
    if len(queue) > 10:
        st.caption(f"Showing the top 10 of {len(queue)}.")

# ============================================================== FEEDBACK LOG
elif page == "Feedback log":
    st.title("Feedback log")
    st.caption("How human review trains the writer: approved drafts become examples to imitate; "
               "rejections and edit requests, with their reasons, become corrections to avoid.")
    with SessionLocal() as s:
        records = training_records(s)
        log_df = pd.DataFrame(records)
    names = scores.set_index("account_id")["customer"].to_dict() if len(scores) else {}

    approved = [r for r in records if r["decision"] == "approved"]
    negative = [r for r in records if r["decision"] != "approved"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Approved examples", len(approved))
    c2.metric("…of which edited by reviewer", sum(r["was_edited"] for r in approved))
    c3.metric("Corrections (reject / needs edit)", len(negative))
    c4.metric("Decisions in training set", len(records))

    if settings.writer_backend == "template" or (settings.writer_backend == "auto" and not settings.anthropic_api_key):
        st.info("**Template mode:** feedback is logged and exported, but drafts come from fixed templates. "
                "Set `WRITER_BACKEND=ollama` in `.env` to have the writer apply it on the next cycle.")

    if not records:
        st.info("No decisions yet. Review a few drafts in the Review queue, including at least one rejection.")
    else:
        st.subheader("Review outcomes")
        OUTCOMES = [("Approved", "#2e8b57"), ("Approved with edits", "#7fbf7f"),
                    ("Sent back for edits", "#e0a030"), ("Rejected", "#c0392b")]
        def _outcome(r):
            if r["decision"] == "approved":
                return "Approved with edits" if r["was_edited"] else "Approved"
            return "Rejected" if r["decision"] == "rejected" else "Sent back for edits"
        oc = pd.Series([_outcome(r) for r in records]).value_counts()
        oc = pd.DataFrame({"outcome": [o for o, _ in OUTCOMES],
                           "drafts": [int(oc.get(o, 0)) for o, _ in OUTCOMES]})
        top = int(oc.drafts.max())
        st.altair_chart(alt.Chart(oc).mark_bar(cornerRadius=3, size=22).encode(
            x=alt.X("drafts:Q", title="Drafts", scale=alt.Scale(domain=[0, top + 1]),
                    axis=alt.Axis(values=list(range(0, top + 2)), format="d")),
            y=alt.Y("outcome:N", title=None, sort=[o for o, _ in OUTCOMES],
                    scale=alt.Scale(paddingInner=0.35, paddingOuter=0.4),
                    axis=alt.Axis(labelLimit=380, labelFontSize=12)),
            color=alt.Color("outcome:N", legend=None,
                            scale=alt.Scale(domain=[o for o, _ in OUTCOMES], range=[c for _, c in OUTCOMES])),
            tooltip=["outcome", "drafts"],
        ).properties(height=56 * len(OUTCOMES) + 60), use_container_width=True)

        if negative:
            st.subheader("Why drafts were rejected or sent back")
            neg = pd.DataFrame(negative)
            neg["reason_label"] = neg["reason"].map(lambda r: REASONS.get(r, "Not recorded"))
            counts = neg.groupby("reason_label").size().reset_index(name="drafts")
            st.altair_chart(alt.Chart(counts).mark_bar(cornerRadius=3, color="#c0392b", size=22).encode(
                x=alt.X("drafts:Q", title="Drafts", scale=alt.Scale(domain=[0, int(counts.drafts.max()) + 1]),
                        axis=alt.Axis(values=list(range(0, int(counts.drafts.max()) + 2)), format="d")),
                y=alt.Y("reason_label:N", title=None, sort="-x",
                        scale=alt.Scale(paddingInner=0.35, paddingOuter=0.4),
                        axis=alt.Axis(labelLimit=380, labelFontSize=12)),
                tooltip=["reason_label", "drafts"],
            ).properties(height=56 * len(counts) + 60), use_container_width=True)
            neg["customer"] = neg["account_id"].map(names)
            neg["decided_at"] = pd.to_datetime(neg["decided_at"]).dt.strftime("%Y-%m-%d %H:%M UTC")
            st.dataframe(neg[["decided_at", "customer", "action", "decision", "reason_label", "explanation", "reviewer"]]
                         .rename(columns={"reason_label": "reason"}).sort_values("decided_at", ascending=False),
                         hide_index=True, use_container_width=True)

        if approved:
            st.subheader("Approved drafts used as examples")
            for r in sorted(approved, key=lambda r: r["decided_at"] or "", reverse=True)[:5]:
                tag = " · edited by reviewer" if r["was_edited"] else ""
                with st.expander(f"{names.get(r['account_id'], r['account_id'])} · "
                                 f"{ACTION_LABEL.get(r['action'], r['action'])}{tag}"):
                    if r["was_edited"]:
                        o, f = st.columns(2)
                        o.markdown("**Writer's draft**"); o.text(r["original_body"] or "")
                        f.markdown("**Approved version**"); f.text(r["final_body"] or "")
                    else:
                        st.text(r["final_body"] or "")

    st.subheader("What the writer will see on the next cycle")
    actions = sorted(all_drafts.action.unique()) if len(all_drafts) else []
    if actions:
        act = st.selectbox("For drafts of type", actions, format_func=lambda a: ACTION_LABEL.get(a, a))
        with SessionLocal() as s:
            preview = prompt_block(writer_feedback(s, act))
        if preview:
            st.code(preview, language=None, wrap_lines=True)
            st.caption("This block is added to the writer's prompt, below the customer context. "
                       "Same-type examples and corrections come first.")
        else:
            st.caption("Nothing to learn from yet.")

    st.download_button("Download training dataset (JSON Lines)",
                       data="\n".join(json.dumps(r, ensure_ascii=False, default=str)
                                      for r in records),
                       file_name="writer_feedback.jsonl", mime="application/json", disabled=not records,
                       help="One record per decision: positive (approved, with final text) or negative "
                            "(with reason and explanation). For evaluating or fine-tuning the writer later.")

# ============================================================== RESULTS
elif page == "Results":
    st.title("Results")
    with SessionLocal() as s:
        ev = pd.read_sql(select(ReviewEvent.decision, ReviewEvent.reason, ReviewEvent.reviewer,
                                ReviewEvent.comment.label("explanation"),
                                ReviewEvent.seconds_to_decision, ReviewEvent.created_at), s.bind)
    decided = len(ev)
    c1, c2, c3 = st.columns(3)
    c1.metric("Decisions made", decided)
    c2.metric("Approval rate", f"{(ev.decision == 'approved').mean():.0%}" if decided else "—")
    held = (all_drafts.status == "insufficient_data").sum() if len(all_drafts) else 0
    c3.metric("Held back by guardrail", int(held))

    if decided:
        st.altair_chart(alt.Chart(ev).mark_bar(cornerRadius=3).encode(
            x=alt.X("count():Q", title="Drafts"), y=alt.Y("decision:N", title=None),
            color=alt.Color("decision:N", legend=None, scale=alt.Scale(
                domain=["approved", "needs_edit", "rejected"], range=["#1e8449", "#d68910", "#c0392b"])),
        ).properties(height=140), use_container_width=True)
        st.markdown("**Audit log**")
        st.dataframe(ev.sort_values("created_at", ascending=False), hide_index=True, use_container_width=True)
    else:
        st.info("No decisions yet. Approve or reject a few drafts in the Review queue.")

    st.subheader("Model quality (held-out test set)")
    mfile = settings.model_dir / "metrics.json"
    if mfile.exists():
        import json
        m = json.loads(mfile.read_text())
        a, b, c, d_ = st.columns(4)
        a.metric("Churn ROC-AUC", f"{m['churn_roc_auc']:.2f}")
        b.metric("Precision at high risk", f"{m['churn_precision@high']:.0%}")
        c.metric("Recall at high risk", f"{m['churn_recall@high']:.0%}")
        d_.metric("Base churn rate", f"{m['churn_base_rate']:.0%}")
    st.caption("Measured on synthetic data: this validates the pipeline, not real-world accuracy. "
               "The pilot measures retention, response time and revenue recovered against a control group.")

# ============================================================== TRUST METRICS
else:
    st.title("Trust metrics")
    st.caption("How the copilot proves itself over time, not just on day one. Each panel says "
               "what it measures and where the numbers come from.")
    with SessionLocal() as s:
        rv = review_metrics(s)
        holdout = holdout_accounts(settings.model_dir)
        pc = prediction_check(s, all_account_ids=holdout, labels=synthetic_labels() if holdout else None)
        rc = retention_comparison(s)
    card = model_card(settings.model_dir)

    # ---- 1. Human review
    st.subheader("1 · Do our people trust the drafts?")
    a, b, c = st.columns(3)
    a.metric("Approval rate", f"{rv['approval_rate']:.0%}" if rv["approval_rate"] is not None else "—",
             help="Approved ÷ all decisions (approved, sent back, rejected)")
    b.metric("Decisions", rv["decisions"])
    c.metric("Sent back / rejected", f"{rv['needs_edit']} / {rv['rejected']}")
    if rv["top_reasons"]:
        st.markdown("**Top rejection reasons**")
        for i, r in enumerate(rv["top_reasons"], 1):
            st.markdown(f"{i}. {r['label']} · **{r['count']}**")
    else:
        st.caption("No rejections yet.")

    # ---- 2. Predictions
    st.subheader("2 · Do flagged accounts really churn?")
    if pc["source"]:
        tbl = pd.DataFrame(pc["rows"])
        tbl["group"] = tbl["group"].map({"high": "Flagged: high risk", "medium": "Flagged: medium risk",
                                         "low": "Flagged: low risk", "not_flagged": "Not flagged"})
        tbl = tbl[tbl.accounts > 0]
        tbl["group"] = [f"{g} ({n:,} accounts)" for g, n in zip(tbl.group, tbl.accounts)]
        a, b = st.columns(2)
        a.metric("High-risk accounts that churned", f"{pc['precision_high']:.0%}" if pc["precision_high"] is not None else "—",
                 help="Precision: of the accounts we flag as high risk, how many really churn")
        b.metric("Churners we flagged as high risk", f"{pc['recall_high']:.0%}" if pc["recall_high"] is not None else "—",
                 help="Recall: of all accounts that churned, how many we caught as high risk")
        st.altair_chart(alt.Chart(tbl).mark_bar(cornerRadius=3, size=22, color="#c0392b").encode(
            x=alt.X("churn_rate:Q", title="Share that churned", axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("group:N", title=None, sort=list(tbl.group),
                    axis=alt.Axis(labelLimit=300, labelFontSize=12)),
            tooltip=["group", "accounts", "churned", alt.Tooltip("churn_rate:Q", format=".0%")],
        ).properties(height=50 * len(tbl) + 50), use_container_width=True)
        st.caption("This checks the whole pipeline: monitoring first (at least two warning signs), then the "
                   "model. That's why it can differ from the model-only numbers in panel 4: the extra filter "
                   "makes the high-risk list more accurate but misses some churners that show fewer signs.")
        if pc["source"] == "synthetic_labels":
            st.caption(f"Source: the synthetic dataset's own churn labels, for the {len(holdout):,} held-out "
                       "accounts the model never saw in training, because no real outcomes have been "
                       "imported yet. In the pilot this uses real ERP outcomes "
                       "(`python -m copilot.trust import-outcomes outcomes.csv`).")
        else:
            st.caption("Source: outcomes imported from the ERP.")
    else:
        st.info("Waiting for outcomes. Import them with `python -m copilot.trust import-outcomes outcomes.csv`. "
                "(In the demo, **Reset demo** retrains the model so it can be checked on held-out accounts.)")

    # ---- 3. Impact
    st.subheader("3 · Does outreach keep customers? Contacted vs. control group")
    cohorts = {c_["cohort"]: c_ for c_ in rc["cohorts"]}
    a, b, c = st.columns(3)
    a.metric("Contacted accounts", cohorts["contacted"]["accounts"],
             help="Flagged accounts whose outreach a reviewer approved")
    b.metric("Control group", cohorts["control"]["accounts"],
             help="Flagged accounts randomly held back: no draft, handled as today")
    if rc["has_outcomes"] and rc["uplift"] is not None:
        c.metric("Retention uplift", f"{rc['uplift']:+.0%}",
                 help="Retention of contacted minus retention of control")
        st.dataframe(pd.DataFrame(rc["cohorts"]), hide_index=True, use_container_width=True)
    else:
        c.metric("Retention uplift", "—")
        st.info("**Measured in the pilot.** Retention needs real outcomes, roughly 6 months after contact, "
                "once a full reorder cycle has passed. The cohorts above show the experiment is set up: "
                f"{settings.control_group_pct or 0}% of flagged accounts are held back at random, so the comparison is fair.")
        if not settings.control_group_pct:
            st.caption("The control group is off. Set `CONTROL_GROUP_PCT=20` in `.env` for the pilot.")

    # ---- 4. Model card
    st.subheader("4 · Which model is live, and is it up to date?")
    if card:
        trained = datetime.fromisoformat(card["trained_at"])
        a, b = st.columns(2)
        a.metric("Last retrained", f"{trained:%d %b %Y}", help=f"{card['trained_at']} UTC")
        b.metric("Days since retraining", card["age_days"])
        st.markdown(f"**Model version** `{card['model_version']}`")
        st.caption("Version = training date and time plus a fingerprint of the training data, so any "
                   "score can be traced to the exact model that produced it.")
        if card["needs_retraining"]:
            st.warning("The model is over 90 days old. Retrain it on recent outcomes: "
                       "`python -m copilot.pipeline.orchestrator --once --retrain`.")
        st.caption(f"Trained on {card['n_train']:,} accounts, tested on {card['n_test']:,} held-out accounts · "
                   f"ROC-AUC {card['churn_roc_auc']:.2f} · precision {card['churn_precision@high']:.0%} · "
                   f"recall {card['churn_recall@high']:.0%} at high risk.")
    else:
        st.info("No trained model yet.")
