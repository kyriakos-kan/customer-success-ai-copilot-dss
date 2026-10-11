"""Next-best-action resolution from the editable YAML playbook.

Most specific matching rule wins (factor match > exact segment > exact
risk level), with file order breaking ties. The playbook is re-read on
every call so CS leadership edits take effect without a restart."""
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Recommendation:
    rule_id: str
    action: str
    channel: str
    owner: str
    rationale: str


def load_rules(path: Path) -> list[dict]:
    return yaml.safe_load(Path(path).read_text())["rules"]


def _specificity(rule: dict) -> int:
    return (4 if rule.get("when_factor") else 0) + \
           (2 if rule["segment"] != "*" else 0) + (1 if rule["risk_level"] != "*" else 0)


def recommend_next_best_action(risk_level: str, segment: str, top_factors: list[dict],
                               playbook_path: Path) -> Recommendation:
    risk_drivers = {f["feature"] for f in top_factors if f["direction"] == "increases risk"}
    candidates = []
    for order, rule in enumerate(load_rules(playbook_path)):
        if rule["risk_level"] not in ("*", risk_level):
            continue
        if rule["segment"] not in ("*", segment):
            continue
        if rule.get("when_factor") and rule["when_factor"] not in risk_drivers:
            continue
        candidates.append((-_specificity(rule), order, rule))
    if not candidates:
        raise LookupError(f"Playbook has no rule for ({risk_level}, {segment}); add a catch-all.")
    rule = sorted(candidates, key=lambda c: (c[0], c[1]))[0][2]
    return Recommendation(rule["id"], rule["action"], rule["channel"], rule["owner"], rule["rationale"])
