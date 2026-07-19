"""
AI-Assisted Interpretation Module
===================================
Fully offline deterministic narrative generator. No network calls, no API
keys, no external dependencies beyond the project's requirements.txt.

All functions receive both the SeriesModel (for historical stats) and the
forecast dict (for projected revenue_p10/p50/p90 and roas_p10/p50/p90).
The horizon_days planning window is passed explicitly since it is no longer
embedded in the forecast dict.
"""

import logging

log = logging.getLogger(__name__)


# ── Reasoning classifiers ────────────────────────────────────────────────────────
def _trend_label(model) -> str:
    """Classify the trend slope relative to the series mean."""
    if model.daily_revenue_mean <= 0:
        return "flat"
    relative_slope = model.trend_slope / max(model.daily_revenue_mean, 1e-6)
    if relative_slope > 0.01:
        return "rising"
    elif relative_slope < -0.01:
        return "declining"
    return "stable"


def _confidence_label(forecast: dict) -> str:
    """Classify forecast confidence from the P10-P90 spread relative to P50."""
    p50 = forecast["revenue_p50"]
    if p50 <= 0:
        return "low"
    spread_ratio = (forecast["revenue_p90"] - forecast["revenue_p10"]) / p50
    if spread_ratio < 0.3:
        return "high"
    elif spread_ratio < 0.7:
        return "moderate"
    return "low"


def _roas_trajectory(model, forecast: dict) -> str:
    """
    Compare projected ROAS P50 to historical ROAS P50 stored on the model.
    Uses model.roas_p50 (fitted from training data) vs forecast['roas_p50'].
    """
    hist = model.roas_p50
    fut  = forecast["roas_p50"]
    if hist <= 0:
        return "insufficient history to compare"
    delta_pct = (fut - hist) / hist * 100
    if delta_pct > 5:
        return f"improving (+{delta_pct:.1f}% vs. historical median)"
    elif delta_pct < -5:
        return f"softening ({delta_pct:.1f}% vs. historical median)"
    return "holding steady near historical median"


# ── Per-series causal summary ────────────────────────────────────────────────────
def generate_series_summary(key: str, level: str, model,
                             forecast: dict, horizon_days: int) -> str:
    """
    Produce a 3-sentence causal/business summary for one series.

    Parameters
    ----------
    key          : series identifier string
    level        : "channel" | "campaign_type" | "campaign"
    model        : SeriesModel (holds historical stats)
    forecast     : dict with revenue_p10/p50/p90 and roas_p10/p50/p90
    horizon_days : planning window (30, 60, or 90)
    """
    trend      = _trend_label(model)
    confidence = _confidence_label(forecast)
    roas_traj  = _roas_trajectory(model, forecast)

    label_map = {"channel": "Channel", "campaign_type": "Campaign type", "campaign": "Campaign"}
    subject   = label_map.get(level, "Series")

    parts = [
        f"{subject} '{key}' shows a {trend} revenue trend based on "
        f"{model.n_obs} days of history, with a projected "
        f"{horizon_days}-day revenue range of "
        f"${forecast['revenue_p10']:,.0f} (P10) to "
        f"${forecast['revenue_p90']:,.0f} (P90), "
        f"centered around ${forecast['revenue_p50']:,.0f} (P50).",

        f"Forecast confidence is {confidence} "
        f"(P10-P90 spread relative to median), "
        f"reflecting "
        f"{'low' if confidence == 'high' else 'meaningful' if confidence == 'moderate' else 'substantial'} "
        f"historical day-to-day volatility in this series.",

        f"Projected ROAS is {roas_traj}.",
    ]

    return " ".join(parts)


# ── Portfolio-level causal summary ───────────────────────────────────────────────
def generate_portfolio_summary(channel_forecasts: dict,
                                blended_forecast: dict,
                                horizon_days: int) -> str:
    """Generate a top-line blended narrative for all channels combined."""
    ranked    = sorted(channel_forecasts.items(),
                       key=lambda kv: kv[1]["revenue_p50"], reverse=True)
    total_p50 = sum(f["revenue_p50"] for _, f in channel_forecasts.items())

    channel_list = (", ".join(sorted(channel_forecasts.keys()))
                    if channel_forecasts else "available channels")

    lines = [
        f"Over the next {horizon_days} days, blended e-commerce revenue across "
        f"{channel_list} is projected at ${blended_forecast['revenue_p50']:,.0f} "
        f"(P50), with a plausible range of "
        f"${blended_forecast['revenue_p10']:,.0f} to "
        f"${blended_forecast['revenue_p90']:,.0f}.",

        f"Blended ROAS is projected at {blended_forecast['roas_p50']:.2f}x "
        f"(range {blended_forecast['roas_p10']:.2f}x – "
        f"{blended_forecast['roas_p90']:.2f}x).",
    ]

    if ranked:
        top_ch, top_fc = ranked[0]
        top_share = (top_fc["revenue_p50"] / total_p50 * 100) if total_p50 > 0 else 0
        lines.append(
            f"{top_ch} is the largest expected contributor at "
            f"~{top_share:.0f}% of blended revenue "
            f"(${top_fc['revenue_p50']:,.0f} P50)."
        )
        if len(ranked) > 1:
            sec_ch, sec_fc = ranked[1]
            sec_share = (sec_fc["revenue_p50"] / total_p50 * 100) if total_p50 > 0 else 0
            lines.append(
                f"{sec_ch} follows at ~{sec_share:.0f}% "
                f"(${sec_fc['revenue_p50']:,.0f} P50)."
            )

    return " ".join(lines)


# ── Assumptions / limitations notes ──────────────────────────────────────────────
def generate_assumptions_notes(horizon_days: int, n_channels: int) -> list:
    return [
        f"Forecast window is {horizon_days} aggregate days; figures are totals "
        f"across the whole window, not daily forecasts.",
        "Existing platform-reported attribution is treated as ground truth; "
        "no cross-channel attribution re-modeling is performed.",
        "Uncertainty ranges (P10/P50/P90) come from bootstrap resampling of "
        "historical residuals and reflect historical volatility only.",
        "This is not a Media Mix Model; channel/campaign forecasts are "
        "produced independently.",
        "No ROAS constraints, targets, or optimization recommendations are "
        "applied; outputs are descriptive forecasts only.",
    ]
