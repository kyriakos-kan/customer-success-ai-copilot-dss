"""Gold-layer loader. In production this reads the Gold Delta table
(Databricks) or a Postgres view; here it generates the synthetic B2B
textile dataset so the stack runs end to end out of the box.

Swap `load_gold()` for your real loader — every downstream stage only
depends on the column contract in FEATURES."""
import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "recency_days", "order_frequency_12m", "log_avg_order_value",
    "order_volume_change_pct", "avg_lead_time_days", "late_payments_12m",
    "support_tickets_90d", "quality_complaints_12m", "reorder_interval_days",
    "engagement_score",
]
CATEGORICAL_FEATURES = ["supplier_origin"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

COUNTRIES = ["DE", "IT", "FR", "NL", "UK", "PL", "SE", "ES", "US", "AT"]


def load_gold(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    origin = rng.choice(["GR", "CN", "TR"], size=n, p=[0.5, 0.3, 0.2])
    lead = np.where(origin == "GR", rng.normal(14, 3, n),
            np.where(origin == "CN", rng.normal(45, 8, n), rng.normal(22, 5, n))).clip(5)
    aov = rng.lognormal(mean=8.5, sigma=0.7, size=n)
    # customer_origin = where the customer is based (export market).
    # supplier_origin = where the fabric is produced (GR / CN / TR only).
    customer_origin = rng.choice(COUNTRIES, n)
    df = pd.DataFrame({
        "account_id": [f"ACC-{i:04d}" for i in range(n)],
        "name": [f"Customer {i:04d} ({c})" for i, c in enumerate(customer_origin)],
        "customer_origin": customer_origin,
        "supplier_origin": origin,
        "recency_days": rng.gamma(2.0, 25, n).round(),
        "order_frequency_12m": rng.poisson(8, n),
        "avg_order_value": aov.round(2),
        # FIX: raw AOV dominated SHAP because of its scale/skew. Models see
        # the log only; the raw value is kept for CLV targets and display.
        "log_avg_order_value": np.log1p(aov),
        "order_volume_change_pct": rng.normal(0, 25, n).round(1),
        "avg_lead_time_days": lead.round(1),
        "late_payments_12m": rng.poisson(1.0, n),
        "support_tickets_90d": rng.poisson(1.5, n),
        "quality_complaints_12m": rng.poisson(0.8, n),
        "reorder_interval_days": rng.normal(45, 15, n).clip(7).round(),
        "engagement_score": rng.beta(5, 2, n).round(3),
    })

    # Targets -------------------------------------------------------------
    # FIX (leakage): churn is a *future* outcome simulated from a latent
    # composite of several drivers plus noise — not a threshold on a single
    # feature the model can also see, which made the old classifier trivial.
    latent = (
        0.025 * df.recency_days - 0.03 * df.order_volume_change_pct
        + 0.45 * df.late_payments_12m + 0.35 * df.quality_complaints_12m
        + 0.02 * (df.avg_lead_time_days - 20) - 2.5 * df.engagement_score
        + rng.normal(0, 1.0, n)
    )
    z = (latent - latent.mean()) / latent.std()
    p = 1 / (1 + np.exp(-(1.8 * z - 1.6)))          # ~20-25% base churn rate
    df["churned_next_6m"] = rng.binomial(1, p)
    df["clv_next_12m"] = (df.avg_order_value * df.order_frequency_12m
                          * (1 - 0.7 * p) * rng.normal(1, 0.1, n)).round(2)

    # RFM segment (quintile scores) ----------------------------------------
    r = pd.qcut(-df.recency_days, 5, labels=False, duplicates="drop") + 1
    f = pd.qcut(df.order_frequency_12m.rank(method="first"), 5, labels=False) + 1
    m = pd.qcut(df.avg_order_value, 5, labels=False) + 1
    score = r + f + m
    df["segment"] = pd.cut(score, bins=[0, 6, 9, 12, 15],
                           labels=["at_risk", "needs_attention", "loyal", "champion"]).astype(str)
    return df
