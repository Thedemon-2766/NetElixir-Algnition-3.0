"""
Forecasting Model Module
=========================
Probabilistic aggregate-period revenue & ROAS forecasts.

forecast_series() returns ONLY the two primary business metrics:
  revenue_p10 / revenue_p50 / revenue_p90
  roas_p10    / roas_p50    / roas_p90

Internal derivations (spend_total, historical_roas, n_obs) stay inside the
function and are never surfaced in the output dict.
"""

import logging
import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

log = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────────────
@dataclass
class SeriesModel:
    key:                str
    level:              str
    daily_revenue_mean: float = 0.0
    daily_revenue_std:  float = 0.0   # 0 for single-row series → safe parametric fallback
    daily_spend_mean:   float = 0.0
    trend_slope:        float = 0.0
    seasonal_dow:       dict  = field(default_factory=dict)
    roas_p10:           float = 0.0
    roas_p50:           float = 0.0
    roas_p90:           float = 0.0
    residual_pool:      list  = field(default_factory=list)
    n_obs:              int   = 0
    last_date:          str   = ""
    campaign_type:      str   = ""   # populated for campaign-level models


@dataclass
class ForecastBundle:
    seed:                 int
    trained_at:           str
    channel_models:       dict
    campaign_type_models: dict
    campaign_models:      dict
    gbr_model:            object = None
    gbr_feature_cols:     list   = field(default_factory=list)
    gbr_mae:              float  = 0.0
    global_roas_p10:      float  = 0.0
    global_roas_p50:      float  = 0.0
    global_roas_p90:      float  = 0.0
    metadata:             dict   = field(default_factory=dict)


# ── Statistical fitting ──────────────────────────────────────────────────────────
def _fit_series(dates: pd.Series, revenue: pd.Series, spend: pd.Series,
                key: str, level: str, campaign_type: str = "") -> SeriesModel:
    df = (pd.DataFrame({"date": dates, "revenue": revenue, "spend": spend})
          .sort_values("date").reset_index(drop=True))
    n = len(df)

    if n == 0:
        return SeriesModel(key=key, level=level)

    t   = np.arange(n)
    rev = df["revenue"].values.astype(float)

    # Linear trend (OLS)
    if n >= 2 and np.std(t) > 0:
        slope, intercept = np.polyfit(t, rev, 1)
    else:
        slope, intercept = 0.0, float(rev[0]) if n else 0.0

    trend_fit = intercept + slope * t

    # Day-of-week seasonality multipliers
    df["dow"]       = pd.to_datetime(df["date"]).dt.dayofweek
    df["trend_fit"] = np.where(trend_fit > 1e-6, trend_fit, 1e-6)
    df["ratio"]     = df["revenue"] / df["trend_fit"]
    dow_raw         = df.groupby("dow")["ratio"].median().to_dict()
    seasonal_dow    = {int(d): float(dow_raw.get(d, 1.0)) for d in range(7)}
    mvals           = np.clip(list(seasonal_dow.values()), 0.3, 3.0)
    mean_mult       = mvals.mean() if mvals.mean() > 0 else 1.0
    seasonal_dow    = {k: float(v / mean_mult) for k, v in zip(seasonal_dow, mvals)}

    # Residual pool for bootstrap (capped at 180 most recent days)
    fitted    = np.array([trend_fit[i] * seasonal_dow.get(int(df["dow"].iloc[i]), 1.0)
                           for i in range(n)])
    fitted    = np.clip(fitted, 0, None)
    residuals = rev - fitted

    # Guard: std is NaN for n=1 → clamp to 0 (flat parametric range)
    std_val = float(np.std(rev, ddof=1)) if n > 1 else 0.0

    # Empirical ROAS distribution
    spend_arr = df["spend"].values.astype(float)
    mask      = spend_arr > 0
    if mask.sum() >= 3:
        roas_vals             = rev[mask] / spend_arr[mask]
        roas_p10, roas_p50, roas_p90 = np.percentile(roas_vals, [10, 50, 90])
    elif mask.sum() > 0:
        roas_p50              = float(np.median(rev[mask] / spend_arr[mask]))
        roas_p10, roas_p90    = roas_p50 * 0.7, roas_p50 * 1.3
    else:
        roas_p10 = roas_p50 = roas_p90 = 0.0

    return SeriesModel(
        key=key, level=level,
        daily_revenue_mean=float(rev.mean()),
        daily_revenue_std=std_val,
        daily_spend_mean=float(spend_arr.mean()),
        trend_slope=float(slope),
        seasonal_dow=seasonal_dow,
        roas_p10=float(roas_p10), roas_p50=float(roas_p50), roas_p90=float(roas_p90),
        residual_pool=residuals.tolist()[-180:],
        n_obs=n,
        last_date=str(df["date"].max()),
        campaign_type=campaign_type,
    )


def fit_all_series(channel_daily, campaign_type_daily, campaign_daily):
    channel_models = {}
    for ch, g in channel_daily.groupby("channel"):
        channel_models[ch] = _fit_series(g["date"], g["revenue"], g["spend"], ch, "channel")
    log.info(f"Fitted {len(channel_models)} channel-level models")

    campaign_type_models = {}
    for (ch, ct), g in campaign_type_daily.groupby(["channel", "campaign_type"]):
        key = f"{ch}|{ct}"
        campaign_type_models[key] = _fit_series(
            g["date"], g["revenue"], g["spend"], key, "campaign_type")
    log.info(f"Fitted {len(campaign_type_models)} campaign-type models")

    campaign_models = {}
    for cid, g in campaign_daily.groupby("campaign_id"):
        name    = g["campaign_name"].iloc[0]
        channel = g["channel"].iloc[0]
        ctype   = g["campaign_type"].iloc[0] if "campaign_type" in g.columns else ""
        key     = f"{channel}|{name}|{cid}"
        campaign_models[key] = _fit_series(
            g["date"], g["revenue"], g["spend"], key, "campaign",
            campaign_type=ctype)
    log.info(f"Fitted {len(campaign_models)} campaign-level models")

    return channel_models, campaign_type_models, campaign_models


# ── GBR cross-check ──────────────────────────────────────────────────────────────
def fit_gbr_model(feature_matrix: pd.DataFrame) -> tuple:
    target_col   = "revenue"
    drop_cols    = {"date", target_col}
    feature_cols = [c for c in feature_matrix.columns if c not in drop_cols]

    X = feature_matrix[feature_cols].values
    y = feature_matrix[target_col].values

    zero_var = [feature_cols[i] for i in range(X.shape[1]) if np.std(X[:, i]) == 0]
    if zero_var:
        log.warning(f"GBR: zero-variance features (ignored by tree model): {zero_var}")

    if len(X) < 20:
        log.warning("Not enough rows for robust GBR CV; fitting directly.")
        m = GradientBoostingRegressor(random_state=SEED, n_estimators=10, max_depth=2)
        m.fit(X, y)
        return m, feature_cols, 0.0

    tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(X) // 30)))
    maes = []
    for tr, te in tscv.split(X):
        m = GradientBoostingRegressor(
            random_state=SEED, n_estimators=200, max_depth=3,
            learning_rate=0.05, subsample=0.8)
        m.fit(X[tr], y[tr])
        maes.append(mean_absolute_error(y[te], m.predict(X[te])))

    mae = float(np.mean(maes)) if maes else 0.0
    log.info(f"GBR CV MAE (daily revenue): {mae:,.2f}")

    final = GradientBoostingRegressor(
        random_state=SEED, n_estimators=200, max_depth=3,
        learning_rate=0.05, subsample=0.8)
    final.fit(X, y)
    return final, feature_cols, mae


# ── Probabilistic forecasting ────────────────────────────────────────────────────
N_BOOTSTRAP = 2000


def forecast_series(model: SeriesModel, horizon_days: int) -> dict:
    """
    Produce probabilistic revenue and ROAS forecasts for one series.

    Returns a dict with exactly six keys:
        revenue_p10 / revenue_p50 / revenue_p90
        roas_p10    / roas_p50    / roas_p90
    """
    if model.n_obs == 0:
        return _empty_forecast()

    last_date    = pd.to_datetime(model.last_date)
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days)
    future_t     = np.arange(model.n_obs, model.n_obs + horizon_days)

    # Trend + seasonal point forecast per future day
    trend_vals    = model.daily_revenue_mean + model.trend_slope * (future_t - model.n_obs / 2)
    trend_vals    = np.clip(trend_vals, 0, None)
    seasonal_mult = np.array([model.seasonal_dow.get(int(d), 1.0)
                               for d in future_dates.dayofweek])
    point_daily   = np.clip(trend_vals * seasonal_mult, 0, None)
    point_total   = float(point_daily.sum())

    # Bootstrap uncertainty from historical residual pool.
    # RNG is created fresh each call (seeded from global SEED) so results are
    # identical whether this is the first or hundredth call in the process.
    if len(model.residual_pool) >= 5:
        rng          = np.random.default_rng(SEED)
        residual_arr = np.array(model.residual_pool)
        boot_totals  = np.empty(N_BOOTSTRAP)
        for i in range(N_BOOTSTRAP):
            sampled       = rng.choice(residual_arr, size=horizon_days, replace=True)
            boot_totals[i] = np.clip(point_daily + sampled, 0, None).sum()
        p10, p50, p90 = np.percentile(boot_totals, [10, 50, 90])
    else:
        # Parametric fallback (std=0 for single-row series → zero spread, safe)
        spread = model.daily_revenue_std * np.sqrt(horizon_days)
        p10    = max(0.0, point_total - 1.2816 * spread)
        p50    = point_total
        p90    = point_total + 1.2816 * spread

    # ROAS: use assumed spend = historical daily mean × horizon
    spend_total = max(model.daily_spend_mean * horizon_days, 0.0)
    if spend_total > 0:
        roas_p10 = p10 / spend_total
        roas_p50 = p50 / spend_total
        roas_p90 = p90 / spend_total
    else:
        roas_p10 = roas_p50 = roas_p90 = 0.0

    return {
        "revenue_p10": round(p10, 2),
        "revenue_p50": round(p50, 2),
        "revenue_p90": round(p90, 2),
        "roas_p10":    round(min(roas_p10, roas_p90), 3),
        "roas_p50":    round(roas_p50, 3),
        "roas_p90":    round(max(roas_p10, roas_p90), 3),
    }


def _empty_forecast() -> dict:
    return {
        "revenue_p10": 0.0, "revenue_p50": 0.0, "revenue_p90": 0.0,
        "roas_p10":    0.0, "roas_p50":    0.0, "roas_p90":    0.0,
    }


# ── Persistence ──────────────────────────────────────────────────────────────────
def save_bundle(bundle: ForecastBundle, path: str):
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"Saved model bundle → {path}")


def load_bundle(path: str) -> ForecastBundle:
    with open(path, "rb") as f:
        return pickle.load(f)
