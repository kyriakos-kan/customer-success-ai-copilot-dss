"""Writer agent: turns assessment + recommendation + retrieved context into a
draft for human review.

Guardrails (hard, enforced in code — not just asked for in the prompt):
  1. No retrieved context for the account -> status `insufficient_data`,
     no draft text is invented.
  2. The LLM must return JSON citing the context ids it relied on; drafts
     citing ids that were not retrieved are rejected as ungrounded.
  3. The LLM can itself answer INSUFFICIENT_DATA, which is respected.
  4. Nothing is ever sent from here — every draft lands as `pending_review`.

Feedback loop (copilot.feedback): approved drafts are passed in as examples to imitate,
and rejection reasons + explanations as corrections to avoid. Each draft records which
examples and corrections it was written with (DraftResult.feedback_used).
"""
import json
import logging
import re
from dataclasses import dataclass, field

from copilot.agents.analyst import AccountAssessment
from copilot.agents.nba import Recommendation

log = logging.getLogger(__name__)

SYSTEM = """You draft B2B customer-success outreach for a B2B textile exporter.
Rules you must follow:
- Use ONLY facts in <context>. Never invent orders, prices, dates, people or promises.
- Never mention churn scores, risk levels, models or "AI" to the customer.
- Tone: warm, professional, concise (under 160 words), written by the account owner.
- If the context does not support a specific, helpful message for the recommended action,
  return {"status": "INSUFFICIENT_DATA", "reason": "..."}.
Otherwise return ONLY JSON: {"status": "OK", "subject": "...", "body": "...", "cited_ids": ["..."]}"""


@dataclass
class DraftResult:
    status: str                       # pending_review | insufficient_data
    subject: str
    body: str
    context_used: list[dict] = field(default_factory=list)
    feedback_used: dict | None = None      # set only when an LLM wrote the draft


class WriterAgent:
    """Backends (WRITER_BACKEND in .env):
      template  - deterministic, grounded templates; no LLM, nothing leaves the machine
      ollama    - local open model via Ollama; nothing leaves the machine
      anthropic - Claude via API; context documents are sent to the provider
      auto      - anthropic if ANTHROPIC_API_KEY is set, otherwise template (default)
    """

    def __init__(self, api_key: str | None, model: str, min_context_docs: int = 1,
                 backend: str | None = None, ollama_url: str | None = None,
                 ollama_model: str | None = None):
        from copilot.config import get_settings
        s = get_settings()
        self.min_context_docs = min_context_docs
        self.model = model
        self.backend = (backend or s.writer_backend).lower()
        if self.backend == "auto":
            self.backend = "anthropic" if api_key else "template"
        self.ollama_url = (ollama_url or s.ollama_url).rstrip("/")
        self.ollama_model = ollama_model or s.ollama_model
        self.ollama_timeout = s.ollama_timeout_seconds
        self.client = None
        if self.backend == "anthropic":
            if not api_key:
                raise ValueError("WRITER_BACKEND=anthropic needs ANTHROPIC_API_KEY")
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        elif self.backend not in ("template", "ollama"):
            raise ValueError(f"Unknown WRITER_BACKEND: {self.backend}")

    @property
    def label(self) -> str:
        return {"template": "template mode (no LLM)",
                "ollama": f"local LLM via Ollama ({self.ollama_model})",
                "anthropic": f"Claude ({self.model})"}[self.backend]

    def draft(self, account_name: str, assessment: AccountAssessment, rec: Recommendation,
              context: list[dict], style_notes: list[str] | None = None,
              use_llm: bool = True, feedback=None) -> DraftResult:
        """feedback: copilot.feedback.WriterFeedback (approved examples + corrections).
        style_notes is the older plain-text form, still accepted."""
        if rec.action == "monitor":
            return DraftResult("no_action", "", "", [])
        if len(context) < self.min_context_docs:
            return self._insufficient(f"No customer context found for {account_name}; "
                                      "add call notes or emails before outreach.")
        ctx = [{"id": c["metadata"].get("doc_id", f"ctx{i}"), **c} for i, c in enumerate(context)]
        if self.backend == "template" or not use_llm:
            return self._template(account_name, assessment, rec, ctx)
        return self._llm(account_name, assessment, rec, ctx, style_notes or [], feedback)

    # ------------------------------------------------------------- LLM calls
    def _complete(self, system: str, prompt: str) -> str:
        """One place that talks to an LLM; returns the raw text reply."""
        if self.backend == "anthropic":
            msg = self.client.messages.create(model=self.model, max_tokens=700, system=system,
                                              messages=[{"role": "user", "content": prompt}])
            return msg.content[0].text
        # Ollama's local REST API, stdlib only (no extra dependency)
        import urllib.request
        payload = json.dumps({
            "model": self.ollama_model, "stream": False, "format": "json",
            "options": {"temperature": 0.3},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(f"{self.ollama_url}/api/chat", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.ollama_timeout) as resp:  # slow on CPU
            return json.loads(resp.read())["message"]["content"]

    # ------------------------------------------------------------------ paths
    @staticmethod
    def _insufficient(reason: str) -> DraftResult:
        return DraftResult("insufficient_data", "Insufficient data — no draft generated", reason, [])

    def _llm(self, name, a, rec, ctx, style_notes, feedback=None) -> DraftResult:
        context_xml = "\n".join(
            f'<doc id="{c["id"]}" type="{c["metadata"]["doc_type"]}">{c["text"]}</doc>' for c in ctx)
        drivers = ", ".join(f'{f["label"]} ({f["value"]})' for f in a.top_factors
                            if f["direction"] == "increases risk") or "none"
        if feedback is not None and not feedback.is_empty():
            from copilot.feedback import prompt_block
            learned = prompt_block(feedback)
        else:
            notes = "\n".join(f"- {n}" for n in style_notes[-5:]) or "- none yet"
            learned = f"Reviewer feedback on past drafts:\n{notes}"
        prompt = (f"Customer: {name}\nRecommended action: {rec.action} via {rec.channel} "
                  f"(owner: {rec.owner})\nWhy: {rec.rationale}\n"
                  f"Internal risk drivers (do not disclose): {drivers}\n\n"
                  f"{learned}\n\n<context>\n{context_xml}\n</context>")
        try:
            raw = self._complete(SYSTEM, prompt)
            data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        except Exception as exc:                                   # network, timeout, parse, ...
            log.warning("Writer LLM (%s) failed (%s); falling back to template", self.backend, exc)
            return self._template(name, a, rec, ctx)

        if data.get("status") != "OK":
            return self._insufficient(data.get("reason", "Model reported insufficient context."))
        valid_ids = {c["id"] for c in ctx}
        cited = {str(i) for i in data.get("cited_ids", [])}
        if not cited or not cited <= valid_ids or not data.get("subject") or not data.get("body"):
            # Ungrounded or malformed LLM output is never shown; use the grounded template instead.
            log.info("LLM draft failed the citation guardrail for %s; using template", name)
            return self._template(name, a, rec, ctx)
        used = [c for c in ctx if c["id"] in cited]
        return DraftResult("pending_review", data["subject"], data["body"], used,
                           feedback.used if feedback is not None else None)

    # Template mode: (keywords that must appear in a retrieved document, customer-facing
    # paragraph, subject). A paragraph is only used when a document supports it, so the
    # draft stays grounded; internal notes are cited for the reviewer, never pasted.
    TEMPLATES = [
        (("complaint", "shrinkage", "colour"),
         "I know the recent batches did not meet the standard you expect. I'd like to go through "
         "the lab reports with you and agree together how we make sure the next deliveries are right.",
         "Following up on the recent quality issues"),
        (("payment terms", "cash-flow"),
         "I'd welcome a short conversation about whether our current payment terms still work "
         "well for you this season, so we can find an arrangement that suits both sides.",
         "A quick review of our terms"),
        (("lead time",),
         "I understand our lead times have made your seasonal planning harder. We could hold stock "
         "of your core articles, or offer Greek-made alternatives with shorter delivery.",
         "Making lead times easier for your planning"),
        (("second supplier", "trialling"),
         "I'd value hearing how our offer compares for your basic qualities, so we can see "
         "where we can do better on price, service or delivery.",
         "How can we serve you better?"),
        (("renews",),
         "Your framework agreement comes up for renewal soon, and I'd like to make sure it still "
         "fits your plans, including the volume rebate tier.",
         "Your framework agreement with us"),
    ]
    ACTION_KEYWORDS = {
        "escalate_quality_review": ("complaint", "shrinkage", "colour"),
        "offer_payment_plan_review": ("payment terms", "cash-flow"),
        "propose_stock_programme": ("lead time",),
        "send_reorder_incentive": ("second supplier", "trialling", "renews"),
        "renewal_offer": ("renews",),
    }

    @classmethod
    def _template(cls, name, a, rec, ctx) -> DraftResult:
        """Deterministic draft — used when no API key is set, or as LLM fallback.
        Every paragraph is backed by a retrieved document, so it cannot invent facts."""
        def matches(doc, kws):
            return any(k in doc["text"].lower() for k in kws)

        # Prefer the document that supports the recommended action, then any supported topic.
        wanted = cls.ACTION_KEYWORDS.get(rec.action, ())
        ordered = sorted(cls.TEMPLATES, key=lambda t: 0 if set(t[0]) & set(wanted) else 1)
        for kws, paragraph, subject in ordered:
            doc = next((c for c in ctx if matches(c, kws)), None)
            if doc:
                break
        else:
            return DraftResult("insufficient_data", "Insufficient data — no draft generated",
                               "Retrieved context does not support a specific message for this action.", [])

        body = (f"Dear {name.split(' (')[0]} team,\n\n{paragraph}\n\n"
                "Would you have 20 minutes this week for a call? I'd like to understand your "
                "priorities for the coming season and how we can support them.\n\n"
                "Kind regards,\n[Account owner]\n[Company name]")
        return DraftResult("pending_review", subject, body, [doc])
