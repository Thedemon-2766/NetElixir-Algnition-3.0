#!/usr/bin/env python3
"""
train.py
========
Ingests 1–3 channel CSVs from data_dir, builds features, fits forecasting
models, and saves model/model.pkl.

Notes
-----
- Accepts 1–3 channel CSVs; warns (does not fail) for any missing channel.
- load_all() takes a dict keyed by channel name → file path.
- training_summary.json records which channels were actually present.
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingest import load_all
from features import (aggregate_channel_daily, aggregate_campaign_type_daily,
                      aggregate_campaign_daily, build_feature_matrix)
from forecasting import (fit_all_series, fit_gbr_model, ForecastBundle, save_bundle)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── CSV discovery ────────────────────────────────────────────────────────────────
def find_csvs(data_dir: str) -> dict:
    """
    Scan data_dir for channel CSVs by filename heuristics.
    Returns a dict with a subset of keys {"google", "bing", "meta"}.
    Warns (does NOT raise) for any channel not found.
    Raises if the directory is empty of CSVs or doesn't exist.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir!r}")

    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir!r}")

    mapping = {}
    for f in files:
        lf   = f.lower()
        path = os.path.join(data_dir, f)
        if "google" in lf:
            mapping["google"] = path
        elif "bing" in lf or "ms_ads" in lf or "msads" in lf or "microsoft" in lf:
            mapping["bing"] = path
        elif "meta" in lf or "facebook" in lf or "fb_ads" in lf or "instagram" in lf:
            mapping["meta"] = path
        else:
            log.warning(f"Unrecognised CSV (skipped — name must contain 'google', "
                        f"'bing'/'microsoft', or 'meta'/'facebook'): {f}")

    if not mapping:
        raise FileNotFoundError(
            f"Found CSV(s) in {data_dir!r} but none matched a known channel "
            f"(google / bing / meta). Files: {files}"
        )

    for ch in ("google", "bing", "meta"):
        if ch not in mapping:
            log.warning(f"No {ch.title()} Ads CSV found — that channel will be excluded "
                        f"from training and forecasting.")

    log.info(f"Channels to train: {list(mapping.keys())}")
    return mapping


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    _here = os.path.dirname(os.path.abspath(__file__))   # always = src/
    _root = os.path.dirname(_here)                        # always = project root
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",  default=os.path.join(_root, "data"),
                        help="Folder containing channel CSVs")
    parser.add_argument("--model-dir", default=os.path.join(_root, "pickle"),
                        help="Folder to save model.pkl into")
    args = parser.parse_args()

    data_dir  = args.data_dir
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    log.info("=" * 70)
    log.info("STEP 1/4 — Data ingestion & validation")
    log.info("=" * 70)
    csvs = find_csvs(data_dir)
    df   = load_all(csvs)          # dict API — works with 1, 2, or 3 channels

    log.info("=" * 70)
    log.info("STEP 2/4 — Feature engineering")
    log.info("=" * 70)
    channel_daily      = aggregate_channel_daily(df)
    campaign_type_daily = aggregate_campaign_type_daily(df)
    campaign_daily     = aggregate_campaign_daily(df)
    feature_matrix     = build_feature_matrix(channel_daily)

    log.info("=" * 70)
    log.info("STEP 3/4 — Fitting statistical forecasting models")
    log.info("=" * 70)
    channel_models, campaign_type_models, campaign_models = fit_all_series(
        channel_daily, campaign_type_daily, campaign_daily
    )

    log.info("=" * 70)
    log.info("STEP 4/4 — Fitting gradient-boosted cross-check model")
    log.info("=" * 70)
    gbr_model, gbr_feature_cols, gbr_mae = fit_gbr_model(feature_matrix)

    # Global blended ROAS (across all available channels)
    blended = (df.groupby("date")
                 .agg(revenue=("revenue","sum"), spend=("spend","sum"))
                 .reset_index())
    blended = blended[blended["spend"] > 0]
    if len(blended) >= 3:
        roas_vals = blended["revenue"] / blended["spend"]
        g_p10, g_p50, g_p90 = np.percentile(roas_vals, [10, 50, 90])
    else:
        g_p10 = g_p50 = g_p90 = 0.0

    bundle = ForecastBundle(
        seed=SEED,
        trained_at=datetime.now(timezone.utc).isoformat(),
        channel_models=channel_models,
        campaign_type_models=campaign_type_models,
        campaign_models=campaign_models,
        gbr_model=gbr_model,
        gbr_feature_cols=gbr_feature_cols,
        gbr_mae=gbr_mae,
        global_roas_p10=float(g_p10),
        global_roas_p50=float(g_p50),
        global_roas_p90=float(g_p90),
        metadata={
            "n_rows_ingested":  int(len(df)),
            "n_channels":       int(df["channel"].nunique()),
            "channels_present": sorted(df["channel"].unique().tolist()),
            "n_campaign_types": int(df["campaign_type"].nunique()),
            "n_campaigns":      int(df["campaign_name"].nunique()),
            "date_min":         str(df["date"].min().date()),
            "date_max":         str(df["date"].max().date()),
            "data_files":       csvs,
        },
    )

    model_path = os.path.join(model_dir, "model.pkl")
    save_bundle(bundle, model_path)

    summary = {
        "trained_at":             bundle.trained_at,
        "seed":                   bundle.seed,
        "metadata":               bundle.metadata,
        "gbr_cv_mae":             bundle.gbr_mae,
        "global_roas_p10":        bundle.global_roas_p10,
        "global_roas_p50":        bundle.global_roas_p50,
        "global_roas_p90":        bundle.global_roas_p90,
        "n_channel_models":       len(channel_models),
        "n_campaign_type_models": len(campaign_type_models),
        "n_campaign_models":      len(campaign_models),
    }
    summary_path = os.path.join(model_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Training complete. Model → {model_path}")
    log.info(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
