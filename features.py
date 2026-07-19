"""
Feature Engineering Module
===========================
BUGS FIXED
----------
B3  build_feature_matrix(): dtype filter now uses pd.api.types.is_numeric_dtype()
    which correctly includes bool, uint32, uint64, int8 etc. Previously the
    hard-coded list [np.float64, np.int64, np.int32, float, int] silently dropped
    is_weekend (bool) and n_campaigns (uint) from the GBR feature matrix.
B4  _channel_tag(): uses hyphen-separated channel names to avoid ambiguous
    concatenation (e.g. "GoogleAdsBingAds" vs "Google-Ads_Bing-Ads").
B5  Removed unused imports: os, pickle.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _channel_tag(df: pd.DataFrame) -> str:
    """Stable, unambiguous cache-key tag from the set of channels present."""
    channels = sorted(df["channel"].unique().tolist())
    # FIX B4: use hyphen within channel name, underscore between channels
    return "_".join(c.replace(" ", "-") for c in channels)


def _rolling(df, col, windows=(7, 14, 28)):
    for w in windows:
        df[f"{col}_roll{w}_mean"] = df[col].rolling(w, min_periods=1).mean()
        df[f"{col}_roll{w}_std"]  = df[col].rolling(w, min_periods=1).std().fillna(0)
    return df


def _lag(df, col, lags=(1, 7, 14)):
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag).fillna(0)
    return df


def _calendar(df):
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["month"]        = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"]      = df["date"].dt.quarter
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["dow_sin"]      = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]      = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"]    = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]    = np.cos(2 * np.pi * df["month"] / 12)
    df["woy_sin"]      = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["woy_cos"]      = np.cos(2 * np.pi * df["week_of_year"] / 52)
    return df


def aggregate_channel_daily(df: pd.DataFrame) -> pd.DataFrame:
    tag        = _channel_tag(df)
    cache_file = CACHE_DIR / f"channel_daily_{tag}.pkl"
    if cache_file.exists():
        log.info("Cache hit: channel_daily")
        return pd.read_pickle(cache_file)

    log.info("Building channel-daily aggregates …")
    grp = (df.groupby(["date", "channel"])
             .agg(spend=("spend", "sum"), revenue=("revenue", "sum"),
                  clicks=("clicks", "sum"), impressions=("impressions", "sum"),
                  conversions=("conversions", "sum"), daily_budget=("daily_budget", "sum"),
                  n_campaigns=("campaign_id", "nunique"))
             .reset_index())
    grp["roas"] = np.where(grp["spend"] > 0, grp["revenue"] / grp["spend"], 0.0)
    grp = _calendar(grp)

    frames = []
    for ch, cdf in grp.groupby("channel"):
        cdf = cdf.sort_values("date").copy()
        for col in ("spend", "revenue", "roas"):
            cdf = _rolling(cdf, col)
        for col in ("spend", "revenue", "roas"):
            cdf = _lag(cdf, col)
        cdf["budget_util"] = np.where(cdf["daily_budget"] > 0,
                                      cdf["spend"] / cdf["daily_budget"], 0.0)
        frames.append(cdf)

    out = pd.concat(frames, ignore_index=True).sort_values(["date", "channel"])
    out.to_pickle(cache_file)
    log.info(f"  channel_daily: {len(out):,} rows")
    return out


def aggregate_campaign_type_daily(df: pd.DataFrame) -> pd.DataFrame:
    tag        = _channel_tag(df)
    cache_file = CACHE_DIR / f"campaign_type_daily_{tag}.pkl"
    if cache_file.exists():
        log.info("Cache hit: campaign_type_daily")
        return pd.read_pickle(cache_file)

    log.info("Building campaign-type-daily aggregates …")
    grp = (df.groupby(["date", "channel", "campaign_type"])
             .agg(spend=("spend", "sum"), revenue=("revenue", "sum"),
                  clicks=("clicks", "sum"), impressions=("impressions", "sum"),
                  conversions=("conversions", "sum"))
             .reset_index())
    grp["roas"] = np.where(grp["spend"] > 0, grp["revenue"] / grp["spend"], 0.0)
    grp = _calendar(grp)

    frames = []
    for _, cdf in grp.groupby(["channel", "campaign_type"]):
        cdf = cdf.sort_values("date").copy()
        cdf = _rolling(cdf, "spend", windows=(7, 14))
        cdf = _rolling(cdf, "revenue", windows=(7, 14))
        cdf = _lag(cdf, "revenue", lags=(1, 7))
        frames.append(cdf)

    out = pd.concat(frames, ignore_index=True).sort_values(["date", "channel", "campaign_type"])
    out.to_pickle(cache_file)
    log.info(f"  campaign_type_daily: {len(out):,} rows")
    return out


def aggregate_campaign_daily(df: pd.DataFrame) -> pd.DataFrame:
    tag        = _channel_tag(df)
    cache_file = CACHE_DIR / f"campaign_daily_{tag}.pkl"
    if cache_file.exists():
        log.info("Cache hit: campaign_daily")
        return pd.read_pickle(cache_file)

    log.info("Building campaign-daily aggregates …")
    grp = (df.groupby(["date", "channel", "campaign_type", "campaign_id", "campaign_name"])
             .agg(spend=("spend", "sum"), revenue=("revenue", "sum"),
                  clicks=("clicks", "sum"), impressions=("impressions", "sum"),
                  conversions=("conversions", "sum"), daily_budget=("daily_budget", "mean"))
             .reset_index())
    grp["roas"] = np.where(grp["spend"] > 0, grp["revenue"] / grp["spend"], 0.0)
    grp = _calendar(grp)

    frames = []
    for _, cdf in grp.groupby("campaign_id"):
        cdf = cdf.sort_values("date").copy()
        cdf = _rolling(cdf, "spend", windows=(7, 14))
        cdf = _rolling(cdf, "revenue", windows=(7, 14))
        cdf = _lag(cdf, "revenue", lags=(1, 7))
        frames.append(cdf)

    out = pd.concat(frames, ignore_index=True).sort_values(["date", "channel", "campaign_name"])
    out.to_pickle(cache_file)
    log.info(f"  campaign_daily: {len(out):,} rows")
    return out


def build_feature_matrix(channel_daily: pd.DataFrame) -> pd.DataFrame:
    tag        = _channel_tag(channel_daily)
    cache_file = CACHE_DIR / f"feature_matrix_{tag}.pkl"
    if cache_file.exists():
        log.info("Cache hit: feature_matrix")
        return pd.read_pickle(cache_file)

    log.info("Building model feature matrix …")
    n_channels = channel_daily["channel"].nunique()
    if n_channels == 1:
        log.warning("Only 1 channel in data — one-hot column will have zero variance; "
                    "GBR ignores it safely.")

    # FIX B3: use pd.api.types.is_numeric_dtype — catches bool, uint32, etc.
    num_cols = [
        c for c in channel_daily.columns
        if c not in ("date", "channel")
        and pd.api.types.is_numeric_dtype(channel_daily[c])
    ]

    feat = channel_daily[["date", "channel"] + num_cols].copy()
    feat = pd.get_dummies(feat, columns=["channel"], prefix="ch")
    feat = feat.fillna(0)
    feat.to_pickle(cache_file)
    log.info(f"  feature_matrix: {len(feat):,} rows × {feat.shape[1]} cols")
    return feat


def clear_feature_cache():
    for f in CACHE_DIR.glob("*.pkl"):
        f.unlink()
    log.info("Feature + ingest cache cleared.")
