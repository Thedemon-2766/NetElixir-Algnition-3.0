#!/usr/bin/env python3
"""
predict.py
==========
Loads the trained model bundle, generates probabilistic revenue & ROAS
forecasts, and writes predictions.csv in the same structural style as the
input channel CSVs.

Output columns (predictions.csv):
    campaign_id        — campaign identifier (or synthetic for aggregates)
    campaign_name      — campaign name (or "ALL_CAMPAIGNS" for roll-ups)
    channel            — ad platform (Google Ads / Bing Ads / Meta Ads / ALL)
    campaign_type      — campaign type (Search / Shopping / Social / etc.)
    forecast_period_days — planning window (30, 60, or 90)
    revenue_p10        — conservative revenue estimate (10th percentile)
    revenue_p50        — median revenue estimate (50th percentile)
    revenue_p90        — optimistic revenue estimate (90th percentile)
    roas_p10           — conservative ROAS estimate
    roas_p50           — median ROAS estimate
    roas_p90           — optimistic ROAS estimate

Usage (called by run.sh):
    python src/predict.py --pickle <MODEL_PATH> --data <DATA_DIR>
                          --output <OUTPUT_PATH> [--horizon 30|60|90]
"""

import argparse
import csv
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

from forecasting import load_bundle, forecast_series
from ai_interpret import (generate_series_summary, generate_portfolio_summary,
                          generate_assumptions_notes)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

VALID_HORIZONS = (30, 60, 90)

# Output column names — mirrors the field naming style of the input CSVs
OUTPUT_FIELDS = [
    "campaign_id",
    "campaign_name",
    "channel",
    "campaign_type",
    "forecast_period_days",
    "revenue_p10",
    "revenue_p50",
    "revenue_p90",
    "roas_p10",
    "roas_p50",
    "roas_p90",
]


def parse_args():
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",  default=os.path.join(_root, "pickle", "model.pkl"))
    p.add_argument("--data",    default=os.path.join(_root, "data"))
    p.add_argument("--output",  default=os.path.join(_root, "output", "predictions.csv"),
                   help="Full .csv path or a directory (predictions.csv appended)")
    p.add_argument("--horizon", type=int, default=30, choices=VALID_HORIZONS)
    return p.parse_args()


def _resolve_output_path(raw: str) -> str:
    """Accept full .csv path or a directory; always return a full .csv path."""
    return raw if raw.endswith(".csv") else os.path.join(raw, "predictions.csv")


def _make_row(campaign_id: str, campaign_name: str, channel: str,
              campaign_type: str, horizon: int, fc: dict) -> dict:
    """Build one output row from forecast dict — only the six forecast values."""
    return {
        "campaign_id":           campaign_id,
        "campaign_name":         campaign_name,
        "channel":               channel,
        "campaign_type":         campaign_type,
        "forecast_period_days":  horizon,
        "revenue_p10":           fc["revenue_p10"],
        "revenue_p50":           fc["revenue_p50"],
        "revenue_p90":           fc["revenue_p90"],
        "roas_p10":              fc["roas_p10"],
        "roas_p50":              fc["roas_p50"],
        "roas_p90":              fc["roas_p90"],
    }


def build_blended_forecast(channel_forecasts: dict) -> dict:
    """
    Sum channel P10/P50/P90 revenues; derive blended ROAS from total historical
    spend (sum of per-channel daily_spend_mean × horizon is not available here,
    so we use the ratio: blended_revenue_pX / sum(channel_revenue_pX) × sum(roas_pX)
    weighted by revenue share — conservative, avoids storing spend internally).

    Simpler approach used: blended ROAS P50 = median of channel ROAS P50 values
    weighted by their revenue P50 share. P10/P90 use the same weighting.
    """
    if not channel_forecasts:
        return {"revenue_p10": 0.0, "revenue_p50": 0.0, "revenue_p90": 0.0,
                "roas_p10":    0.0, "roas_p50":    0.0, "roas_p90":    0.0}

    rev_p10 = sum(f["revenue_p10"] for f in channel_forecasts.values())
    rev_p50 = sum(f["revenue_p50"] for f in channel_forecasts.values())
    rev_p90 = sum(f["revenue_p90"] for f in channel_forecasts.values())

    # Revenue-weighted average ROAS across channels
    def _weighted_roas(roas_key: str, rev_key: str) -> float:
        total_rev = sum(f[rev_key] for f in channel_forecasts.values())
        if total_rev <= 0:
            return 0.0
        return sum(f[roas_key] * f[rev_key] / total_rev
                   for f in channel_forecasts.values())

    return {
        "revenue_p10": round(rev_p10, 2),
        "revenue_p50": round(rev_p50, 2),
        "revenue_p90": round(rev_p90, 2),
        "roas_p10":    round(_weighted_roas("roas_p10", "revenue_p10"), 3),
        "roas_p50":    round(_weighted_roas("roas_p50", "revenue_p50"), 3),
        "roas_p90":    round(_weighted_roas("roas_p90", "revenue_p90"), 3),
    }


def main():
    args      = parse_args()
    pred_path = _resolve_output_path(args.output)

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(pred_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("STEP 1/4 — Loading trained model bundle")
    log.info("=" * 70)
    bundle            = load_bundle(args.pickle)
    channels_in_model = list(bundle.channel_models.keys())
    log.info(f"Channels in model : {channels_in_model}")
    log.info(f"Model trained at  : {bundle.trained_at}")

    horizon = args.horizon
    log.info("=" * 70)
    log.info(f"STEP 2/4 — Generating forecasts (horizon={horizon} days)")
    log.info("=" * 70)

    rows = []

    # ── Channel-level rows ────────────────────────────────────────────────────
    channel_forecasts = {}
    for ch_name, model in bundle.channel_models.items():
        fc = forecast_series(model, horizon)
        channel_forecasts[ch_name] = fc
        rows.append(_make_row(
            campaign_id="ALL_CAMPAIGNS",
            campaign_name="ALL_CAMPAIGNS",
            channel=ch_name,
            campaign_type="ALL_TYPES",
            horizon=horizon,
            fc=fc,
        ))
    log.info(f"  Channel-level    : {len(channel_forecasts)}")

    # ── Campaign-type-level rows ──────────────────────────────────────────────
    campaign_type_forecasts = {}
    for key, model in bundle.campaign_type_models.items():
        channel, ctype = key.split("|", 1)
        fc = forecast_series(model, horizon)
        campaign_type_forecasts[key] = fc
        rows.append(_make_row(
            campaign_id="ALL_CAMPAIGNS",
            campaign_name="ALL_CAMPAIGNS",
            channel=channel,
            campaign_type=ctype,
            horizon=horizon,
            fc=fc,
        ))
    log.info(f"  Campaign-type    : {len(campaign_type_forecasts)}")

    # ── Campaign-level rows ───────────────────────────────────────────────────
    campaign_forecasts = {}
    for key, model in bundle.campaign_models.items():
        parts   = key.split("|", 2)
        channel = parts[0]
        cname   = parts[1] if len(parts) > 1 else "Unknown"
        cid     = parts[2] if len(parts) > 2 else ""
        # campaign_type is now stored directly on the SeriesModel
        ctype   = getattr(model, "campaign_type", "")
        fc = forecast_series(model, horizon)
        campaign_forecasts[key] = fc
        rows.append(_make_row(
            campaign_id=cid,
            campaign_name=cname,
            channel=channel,
            campaign_type=ctype,
            horizon=horizon,
            fc=fc,
        ))
    log.info(f"  Campaign-level   : {len(campaign_forecasts)}")

    # ── Blended total row ─────────────────────────────────────────────────────
    blended = build_blended_forecast(channel_forecasts)
    rows.append(_make_row(
        campaign_id="ALL_CAMPAIGNS",
        campaign_name="ALL_CAMPAIGNS",
        channel="ALL",
        campaign_type="ALL_TYPES",
        horizon=horizon,
        fc=blended,
    ))

    # ── Write predictions.csv ─────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("STEP 3/4 — Writing predictions.csv (overwrite)")
    log.info("=" * 70)

    if os.path.exists(pred_path):
        os.remove(pred_path)
        log.info("  Existing predictions.csv removed.")

    with open(pred_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log.info(f"  Wrote {len(rows)} rows → {pred_path}")

    # ── AI-assisted summaries ─────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("STEP 4/4 — Generating AI-assisted causal summaries")
    log.info("=" * 70)

    portfolio_summary = generate_portfolio_summary(channel_forecasts, blended, horizon)

    channel_summaries = {
        ch: generate_series_summary(ch, "channel", model, channel_forecasts[ch], horizon)
        for ch, model in bundle.channel_models.items()
    }
    campaign_type_summaries = {
        key: generate_series_summary(key, "campaign_type", model,
                                     campaign_type_forecasts[key], horizon)
        for key, model in bundle.campaign_type_models.items()
    }
    top_campaigns = sorted(campaign_forecasts.items(),
                           key=lambda kv: kv[1]["revenue_p50"], reverse=True)[:10]
    campaign_summaries = {
        key: generate_series_summary(key, "campaign", bundle.campaign_models[key], fc, horizon)
        for key, fc in top_campaigns
    }

    # Sidecar outputs go into the same directory as predictions.csv
    report = {
        "generated_at":               datetime.now(timezone.utc).isoformat(),
        "model_trained_at":           bundle.trained_at,
        "channels_in_model":          channels_in_model,
        "seed":                       bundle.seed,
        "horizon_days":               horizon,
        "portfolio_summary":          portfolio_summary,
        "blended_forecast":           blended,
        "channel_forecasts":          channel_forecasts,
        "channel_summaries":          channel_summaries,
        "campaign_type_forecasts":    campaign_type_forecasts,
        "campaign_type_summaries":    campaign_type_summaries,
        "top_10_campaign_forecasts":  dict(top_campaigns),
        "top_10_campaign_summaries":  campaign_summaries,
        "assumptions_and_limitations": generate_assumptions_notes(
                                           horizon, len(channels_in_model)),
    }

    report_path = os.path.join(out_dir, "forecast_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"  Wrote → {report_path}")

    md_path = os.path.join(out_dir, "forecast_summary.md")
    with open(md_path, "w") as f:
        f.write(f"# Forecast Summary — {horizon}-Day Planning Window\n\n")
        f.write(f"_Generated: {report['generated_at']}_\n")
        f.write(f"_Channels: {', '.join(channels_in_model)}_\n\n")
        f.write("## Portfolio Overview\n\n" + portfolio_summary + "\n\n")
        f.write("## Blended Forecast\n\n")
        f.write(f"- Revenue P50: ${blended['revenue_p50']:,.0f} "
                f"(range ${blended['revenue_p10']:,.0f} – ${blended['revenue_p90']:,.0f})\n")
        f.write(f"- ROAS P50:    {blended['roas_p50']:.2f}x "
                f"(range {blended['roas_p10']:.2f}x – {blended['roas_p90']:.2f}x)\n\n")
        f.write("## Channel Detail\n\n")
        for ch_name, fc in channel_forecasts.items():
            f.write(f"### {ch_name}\n\n{channel_summaries[ch_name]}\n\n")
            f.write(f"- Revenue: ${fc['revenue_p10']:,.0f} – ${fc['revenue_p90']:,.0f} "
                    f"(P50 ${fc['revenue_p50']:,.0f})\n")
            f.write(f"- ROAS:    {fc['roas_p10']:.2f}x – {fc['roas_p90']:.2f}x "
                    f"(P50 {fc['roas_p50']:.2f}x)\n\n")
        f.write("## Assumptions & Limitations\n\n")
        for a in report["assumptions_and_limitations"]:
            f.write(f"- {a}\n")
    log.info(f"  Wrote → {md_path}")

    log.info("=" * 70)
    log.info("DONE.")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
