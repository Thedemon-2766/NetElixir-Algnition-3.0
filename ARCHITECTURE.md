# Architecture Overview

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.14 |
| Statistical forecasting | Custom trend + seasonality + bootstrap (NumPy / Pandas) |
| ML cross-check | scikit-learn `GradientBoostingRegressor` |
| AI interpretation | Rule-based NLG (fully offline, deterministic) |
| Entry point | Bash (`run.sh`) |
| Serialisation | Python `pickle` (protocol = `HIGHEST_PROTOCOL`) |
| Feature cache | Per-file MD5-keyed Pickle files in `src/cache/` |
| Frontend | None — CLI tool producing CSV + JSON + Markdown |

---

## Repository layout

```
run.sh                          Shell entry point (§3 contract: DATA_DIR MODEL_PATH OUTPUT_PATH)
requirements.txt                Pinned Python dependencies (numpy, pandas, scikit-learn)
pickle/
  model.pkl                     Committed ForecastBundle (GBR + per-series stats)
data/                           Replaced by grader at test time
output/                         Generated at run time
src/
  ingest.py                     Load → dtype-based column detection → validate → normalise → cache
  features.py                   Calendar + rolling + lag features; channel-tagged cache
  forecasting.py                _fit_series(), fit_gbr_model(), forecast_series(), bundle I/O
  train.py                      CLI: finds CSVs → ingests → features → fits → saves model.pkl
  predict.py                    CLI: loads model.pkl → forecasts → writes predictions.csv
  ai_interpret.py               Offline deterministic NLG for causal summaries
  cache/                        Intermediate data cache (auto-generated, not committed)
```

---

## Forecasting pipeline

```
DATA_DIR (CSVs)
      │
      ▼
  ingest.py
  ┌─────────────────────────────────┐
  │ 1. find_csvs() — filename match │
  │ 2. validate_schemas() — dtype   │
  │    detection pre-flight         │
  │ 3. load_csv() per channel:      │
  │    • _detect_columns() by dtype │
  │      + keyword + cardinality    │
  │    • micros detection (Google)  │
  │    • _clean_common()            │
  │    • _dedup(date+id+channel)    │
  │    • MD5 content cache          │
  └───────────────┬─────────────────┘
                  │ unified DataFrame
                  ▼
  features.py
  ┌─────────────────────────────────┐
  │ aggregate_channel_daily()       │
  │ aggregate_campaign_type_daily() │
  │ aggregate_campaign_daily()      │
  │ build_feature_matrix()          │
  │   • calendar (sin/cos encoding) │
  │   • 7/14/28-day rolling avg/std │
  │   • 1/7/14-day lags             │
  │   • budget utilisation ratio    │
  │   • one-hot encoded channel     │
  └───────────────┬─────────────────┘
                  ▼
  forecasting.py — fit phase
  ┌─────────────────────────────────┐
  │ _fit_series() per series:       │
  │   • OLS linear trend            │
  │   • DOW seasonality multipliers │
  │   • Residual pool (last 180d)   │
  │   • Empirical ROAS P10/P50/P90  │
  │   • campaign_type stored on     │
  │     SeriesModel                 │
  │ fit_gbr_model():                │
  │   • GradientBoostingRegressor   │
  │   • TimeSeriesSplit CV (5-fold) │
  └───────────────┬─────────────────┘
                  ▼
          pickle/model.pkl
          (ForecastBundle dataclass)
                  │
                  ▼
  forecasting.py — predict phase
  ┌─────────────────────────────────┐
  │ forecast_series() per series:   │
  │   • Project trend × seasonality │
  │   • 2,000 bootstrap resamples   │
  │     (fresh RNG per call)        │
  │   • P10 / P50 / P90 revenue     │
  │   • ROAS from historical spend  │
  │ build_blended_forecast()        │
  │   • Revenue-weighted ROAS blend │
  └───────────────┬─────────────────┘
                  │
                  ▼
  ai_interpret.py (fully offline)
  ┌─────────────────────────────────┐
  │ Classifiers (deterministic):    │
  │   • trend: rising/stable/decl.  │
  │   • confidence: P10-P90 spread  │
  │   • ROAS trajectory vs history  │
  │ Per-series + portfolio narrative│
  │ No network calls. No API keys.  │
  └───────────────┬─────────────────┘
                  │
                  ▼
  output/predictions.csv            ← 11 columns, scored by grader
  output/forecast_report.json       ← full structured report
  output/forecast_summary.md        ← human-readable narrative
```

---

## AI interpretation workflow

```
Statistical forecast numbers (revenue_p10/p50/p90, roas_p10/p50/p90)
  + SeriesModel (historical stats: trend_slope, roas_p50, n_obs)
         │
         ▼
ai_interpret.py — deterministic reasoning
  1. _trend_label()      → rising / stable / declining
  2. _confidence_label() → high / moderate / low  (P10-P90 spread / P50)
  3. _roas_trajectory()  → improving / softening / steady
  4. Compose 3-sentence narrative per series
  5. generate_portfolio_summary() → blended executive narrative
         │
         ▼
  forecast_report.json  (all summaries + all forecast numbers)
  forecast_summary.md   (human-readable planning document)
```

---

## Key design decisions

**Why retrain at test time?** The `SeriesModel` per channel/campaign contains trend slope, day-of-week multipliers, and a residual pool — all derived from the training data. The grader replaces `data/` with held-out test data, so retraining adapts these statistics to the new history. GBR hyperparameters (depth, learning rate, n_estimators) are fixed in code.

**Why not ARIMA/Prophet/deep learning?** Per-series histories can be short (weeks, not years). OLS-trend + DOW-seasonal + residual-bootstrap is stable on short series, has no extra dependencies, and is fully interpretable.

**Why offline-only AI?** The submission guide explicitly prohibits network calls at run time. All narrative generation is deterministic rule-based NLG — reproducible, free, zero latency.

**Why revenue-weighted ROAS blending?** A simple average ROAS across channels ignores that a $10M Google channel and a $50K Bing channel don't contribute equally. Revenue-weighted average gives the correct economic blended ROAS.
