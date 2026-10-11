"""Monitoring agent: cheap, rule-based triage. It flags accounts whose
behaviour shifted; it does NOT judge risk (that is the analyst's job)."""
from dataclasses import dataclass, field

import pandas as pd

# (column, operator, threshold, human-readable signal). Tune freely.
RULES = [
    ("order_volume_change_pct", "<", -20, "Order volume dropped >20%"),
    ("late_payments_12m", ">=", 3, "3+ late payments in 12 months"),
    ("support_tickets_90d", ">=", 4, "Support ticket spike (4+ in 90 days)"),
    ("quality_complaints_12m", ">=", 2, "Repeated quality complaints"),
    ("engagement_score", "<", 0.45, "Low engagement"),
    ("recency_days", ">", 90, "No order in 90+ days"),
]
_OPS = {"<": pd.Series.lt, "<=": pd.Series.le, ">": pd.Series.gt, ">=": pd.Series.ge}


@dataclass
class MonitoringResult:
    flagged: dict[str, list[str]] = field(default_factory=dict)   # account_id -> signals

    @property
    def account_ids(self) -> list[str]:
        return list(self.flagged)


class MonitoringAgent:
    def __init__(self, rules=RULES, min_signals: int = 2):
        self.rules = rules
        self.min_signals = min_signals

    def run(self, gold: pd.DataFrame) -> MonitoringResult:
        hits = pd.DataFrame(index=gold.index)
        for col, op, thr, label in self.rules:
            hits[label] = _OPS[op](gold[col], thr)
        result = MonitoringResult()
        for idx, row in hits.iterrows():
            signals = [label for label, hit in row.items() if hit]
            if len(signals) >= self.min_signals:
                result.flagged[gold.at[idx, "account_id"]] = signals
        return result
