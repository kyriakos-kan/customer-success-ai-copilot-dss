"""Analyst agent: CLV regression + churn classification + per-account
explanations.

Fine-tuning notes vs. the earlier notebook version:
  * Explanations use XGBoost's native TreeSHAP (`pred_contribs=True`). Same
    Shapley values as `shap.TreeExplainer`, no `shap` dependency, no mock
    module needed in sandboxes, and ~10x faster at scoring time.
  * One-hot columns for supplier_origin are folded back into a single
    "supplier_origin" factor, so explanations read like business language.
  * Models are fitted on a train split and evaluated on a held-out split;
    metrics are saved next to the model so the API can show them.
  * `revenue_at_risk = churn_probability * clv` is used for prioritisation,
    matching the pilot rule "top at-risk accounts by CLV".
"""
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score, mean_absolute_error,
                             precision_score, recall_score, roc_auc_score,
                             root_mean_squared_error)
from sklearn.model_selection import train_test_split

from copilot.data import NUMERIC_FEATURES

ORIGINS = ["GR", "CN", "TR"]

FRIENDLY = {
    "recency_days": "Days since last order",
    "order_frequency_12m": "Orders in last 12 months",
    "log_avg_order_value": "Average order value",
    "order_volume_change_pct": "Order volume trend",
    "avg_lead_time_days": "Average lead time",
    "late_payments_12m": "Late payments",
    "support_tickets_90d": "Recent support tickets",
    "quality_complaints_12m": "Quality complaints",
    "reorder_interval_days": "Reorder interval",
    "engagement_score": "Engagement",
    "supplier_origin": "Supplier origin",
}


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic encoding so train and serve always produce the same columns."""
    X = df[NUMERIC_FEATURES].astype(float).copy()
    for o in ORIGINS:
        X[f"supplier_origin={o}"] = (df["supplier_origin"] == o).astype(float)
    return X


@dataclass
class AccountAssessment:
    account_id: str
    clv_score: float
    churn_probability: float
    revenue_at_risk: float
    risk_level: str
    top_factors: list[dict]


class AnalystAgent:
    CLV_FILE, CHURN_FILE, METRICS_FILE = "clv_model.joblib", "churn_model.joblib", "metrics.json"
    HOLDOUT_FILE = "holdout_accounts.json"

    def __init__(self, model_dir: Path, risk_high: float = 0.65, risk_medium: float = 0.35):
        self.model_dir = Path(model_dir)
        self.risk_high, self.risk_medium = risk_high, risk_medium
        self.clv_model: xgb.XGBRegressor | None = None
        self.churn_model: xgb.XGBClassifier | None = None

    # ---------------------------------------------------------------- train
    def train(self, gold: pd.DataFrame, seed: int = 42) -> dict:
        X, y_clv, y_churn = encode(gold), gold["clv_next_12m"], gold["churned_next_6m"]
        X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
            X, y_clv, y_churn, test_size=0.25, random_state=seed, stratify=y_churn)

        common = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=2)
        self.clv_model = xgb.XGBRegressor(objective="reg:squarederror", **common)
        self.clv_model.fit(X_tr, yc_tr)

        pos_weight = float((yr_tr == 0).sum() / max((yr_tr == 1).sum(), 1))
        self.churn_model = xgb.XGBClassifier(objective="binary:logistic",
                                             eval_metric="aucpr",
                                             scale_pos_weight=pos_weight, **common)
        self.churn_model.fit(X_tr, yr_tr)

        p = self.churn_model.predict_proba(X_te)[:, 1]
        pred_clv = self.clv_model.predict(X_te)
        metrics = {
            "clv_rmse": float(root_mean_squared_error(yc_te, pred_clv)),
            "clv_mae": float(mean_absolute_error(yc_te, pred_clv)),
            "churn_roc_auc": float(roc_auc_score(yr_te, p)),
            "churn_pr_auc": float(average_precision_score(yr_te, p)),
            "churn_precision@high": float(precision_score(yr_te, p >= self.risk_high, zero_division=0)),
            "churn_recall@high": float(recall_score(yr_te, p >= self.risk_high, zero_division=0)),
            "churn_base_rate": float(yr_te.mean()),
            "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        }
        # Model card: which model is live and when it was trained (shown on Trust metrics)
        trained_at = datetime.now(timezone.utc)
        data_hash = hashlib.sha256(pd.util.hash_pandas_object(X, index=False).values.tobytes()
                                   ).hexdigest()[:8]
        metrics |= {
            "model_version": f"xgb-{trained_at:%Y%m%d-%H%M}-{data_hash}",
            "trained_at": trained_at.isoformat(timespec="seconds"),
            "training_data_hash": data_hash,
            "n_features": int(X.shape[1]),
        }
        self.save(metrics)
        # Held-out accounts: the only ones it is fair to check predictions against later
        (self.model_dir / self.HOLDOUT_FILE).write_text(
            json.dumps(gold.loc[X_te.index, "account_id"].tolist()))
        return metrics

    # ----------------------------------------------------------- persistence
    def save(self, metrics: dict) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clv_model, self.model_dir / self.CLV_FILE)
        joblib.dump(self.churn_model, self.model_dir / self.CHURN_FILE)
        (self.model_dir / self.METRICS_FILE).write_text(json.dumps(metrics, indent=2))

    def load(self) -> bool:
        clv, churn = self.model_dir / self.CLV_FILE, self.model_dir / self.CHURN_FILE
        if not (clv.exists() and churn.exists()):
            return False
        self.clv_model, self.churn_model = joblib.load(clv), joblib.load(churn)
        return True

    # ----------------------------------------------------------------- score
    def _risk_level(self, p: float) -> str:
        return "high" if p >= self.risk_high else "medium" if p >= self.risk_medium else "low"

    def _explain(self, X: pd.DataFrame, raw: pd.DataFrame, k: int) -> list[list[dict]]:
        contribs = self.churn_model.get_booster().predict(
            xgb.DMatrix(X), pred_contribs=True)[:, :-1]          # drop bias column
        contrib_df = pd.DataFrame(contribs, columns=X.columns, index=X.index)
        origin_cols = [c for c in X.columns if c.startswith("supplier_origin=")]
        contrib_df["supplier_origin"] = contrib_df[origin_cols].sum(axis=1)
        contrib_df = contrib_df.drop(columns=origin_cols)

        out = []
        for idx, row in contrib_df.iterrows():
            top = row.reindex(row.abs().sort_values(ascending=False).index)[:k]
            factors = []
            for feat, val in top.items():
                shown = raw.at[idx, "avg_order_value"] if feat == "log_avg_order_value" else raw.at[idx, feat]
                factors.append({
                    "feature": feat,
                    "label": FRIENDLY.get(feat, feat),
                    "value": shown.item() if hasattr(shown, "item") else shown,
                    "shap_log_odds": round(float(val), 4),
                    "direction": "increases risk" if val > 0 else "reduces risk",
                })
            out.append(factors)
        return out

    def assess(self, gold: pd.DataFrame, account_ids: list[str] | None = None,
               top_k: int = 5) -> list[AccountAssessment]:
        if self.churn_model is None and not self.load():
            raise RuntimeError("Models not trained yet — run train() first.")
        df = gold if account_ids is None else gold[gold.account_id.isin(account_ids)]
        if df.empty:
            return []
        X = encode(df)
        p = self.churn_model.predict_proba(X)[:, 1]
        clv = np.clip(self.clv_model.predict(X), 0, None)
        factors = self._explain(X, df, top_k)
        return [
            AccountAssessment(
                account_id=aid, clv_score=round(float(c), 2),
                churn_probability=round(float(pi), 4),
                revenue_at_risk=round(float(pi * c), 2),
                risk_level=self._risk_level(pi), top_factors=f)
            for aid, c, pi, f in zip(df.account_id, clv, p, factors)
        ]


def as_dict(a: AccountAssessment) -> dict:
    return asdict(a)
