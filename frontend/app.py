"""Streamlit human-review console. Talks only to the API (never to the DB),
so it could be replaced by React without touching the backend."""
import json
import os

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000")
st.set_page_config(page_title="CS Copilot", layout="wide")


def get(path, **params):
    r = requests.get(f"{API}{path}", params=params, timeout=10)
    r.raise_for_status()
    return r.json()


st.sidebar.title("CS Copilot")
reviewer = st.sidebar.text_input("Reviewer", value="Reviewer")
page = st.sidebar.radio("View", ["Review queue", "At-risk accounts", "Feedback log", "Metrics"])
REASONS = get("/feedback/reasons")

if page == "Review queue":
    drafts = get("/drafts", status="pending_review")
    st.header(f"Review queue · {len(drafts)} pending")
    if not drafts:
        st.info("Nothing to review. The pipeline adds drafts after each monitoring run.")
    for d in drafts:
        with st.expander(f"{d['account_name']} — {d['action'].replace('_', ' ')}", expanded=False):
            left, right = st.columns([3, 2])
            with right:
                risk = get(f"/accounts/{d['account_id']}/risk")
                expl = get(f"/accounts/{d['account_id']}/explanation")
                st.metric("Churn probability", f"{risk['churn_probability']:.0%}", risk["risk_level"])
                st.metric("Predicted 12-month CLV", f"€{risk['clv_score']:,.0f}")
                st.caption("Why flagged")
                for s in expl["signals"]:
                    st.write(f"• {s}")
                st.caption("Top model factors")
                st.dataframe(pd.DataFrame(expl["top_factors"])[["label", "value", "direction"]],
                             hide_index=True, use_container_width=True)
                st.caption(f"Playbook: {d['action_rationale']}")
            with left:
                st.text_input("Subject", d["subject"], key=f"s{d['id']}", disabled=True)
                body = st.text_area("Draft", d["body"], height=260, key=f"b{d['id']}")
                with st.popover("Grounding context"):
                    for c in d["context_used"]:
                        st.write(f"**{c['doc_type']}** — {c['text']}")
                fb = d.get("feedback_used") or {}
                if fb.get("approved_examples") or fb.get("lessons"):
                    st.caption(f"✍️ Written with {len(fb.get('approved_examples', []))} approved example(s) "
                               f"and {len(fb.get('lessons', []))} reviewer correction(s).")
                reason = st.selectbox("Reason (required for Needs edit / Reject)", [None, *REASONS],
                                      key=f"rs{d['id']}",
                                      format_func=lambda k: "— choose a reason —" if k is None else REASONS[k])
                explanation = st.text_area("Explanation (required for Needs edit / Reject; trains the writer)",
                                           key=f"c{d['id']}", height=70)
                a, e, r = st.columns(3)
                decision = None
                if a.button("Approve", key=f"a{d['id']}", type="primary"):
                    decision = "approved"
                if e.button("Needs edit", key=f"e{d['id']}"):
                    decision = "needs_edit"
                if r.button("Reject", key=f"r{d['id']}"):
                    decision = "rejected"
                if decision:
                    resp = requests.post(f"{API}/drafts/{d['id']}/review", timeout=10, json={
                        "reviewer": reviewer, "decision": decision, "reason": reason,
                        "explanation": explanation or None,
                        "edited_body": body if body != d["body"] else None})
                    if resp.ok:
                        st.toast(decision)
                        st.rerun()
                    else:
                        st.error(resp.json().get("detail", resp.text))

    with st.expander("Accounts held back for insufficient data"):
        for d in get("/drafts", status="insufficient_data"):
            st.write(f"**{d['account_name']}** — {d['body']}")

elif page == "Feedback log":
    st.header("Feedback log")
    st.caption("Approved drafts become examples for the writer; rejections with reasons become corrections.")
    fb = get("/feedback")
    c1, c2 = st.columns(2)
    c1.metric("Approved examples in use", len(fb["approved_examples"]))
    c2.metric("Corrections in use", len(fb["corrections"]))
    if fb["reason_counts"]:
        st.bar_chart(pd.Series({REASONS.get(k, k): v for k, v in fb["reason_counts"].items()}))
    if fb["corrections"]:
        st.dataframe(pd.DataFrame(fb["corrections"])[["action", "label", "explanation"]],
                     hide_index=True, use_container_width=True)
    st.subheader("What the writer will see next run")
    st.code(fb["prompt_preview"] or "Nothing to learn from yet.", language=None, wrap_lines=True)
    records = get("/feedback/export")
    st.download_button("Download training dataset (JSON Lines)",
                       "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
                       file_name="writer_feedback.jsonl", disabled=not records)

elif page == "At-risk accounts":
    level = st.selectbox("Risk level", ["all", "high", "medium", "low"])
    rows = get("/accounts", **({} if level == "all" else {"risk_level": level}))
    st.header(f"Flagged accounts · {len(rows)}")
    if rows:
        df = pd.DataFrame(rows)
        df["signals"] = df["signals"].apply(lambda s: "; ".join(s))
        st.dataframe(df, hide_index=True, use_container_width=True,
                     column_config={"churn_probability": st.column_config.ProgressColumn(
                         "Churn prob.", min_value=0, max_value=1, format="%.2f"),
                         "revenue_at_risk": st.column_config.NumberColumn(format="€%.0f"),
                         "clv_score": st.column_config.NumberColumn("CLV", format="€%.0f")})

else:
    m = get("/metrics")
    st.header("Evaluation")
    c1, c2, c3 = st.columns(3)
    rv = m["review"]
    c1.metric("Draft approval rate", f"{rv['approval_rate']:.0%}" if rv["approval_rate"] is not None else "—")
    c2.metric("Avg hours to decision", rv["avg_hours_flag_to_decision"] or "—")
    c3.metric("Pending drafts", m["queue"].get("pending_review", 0))
    st.subheader("Model (held-out test set)")
    st.json(m["model"] or {"status": "not trained yet"})
    st.caption("Business metrics (retention, revenue recovered) need outcome data "
               "from the pilot's control group and are added once a quarter has elapsed.")
