# AIgnition 3.0 — Probabilistic Revenue & ROAS Forecasting

**NetElixir Hackathon 2026 — Submission**

A fully offline ML pipeline that ingests Google Ads, Bing (Microsoft) Ads, and Meta Ads campaign CSVs and produces probabilistic (P10/P50/P90) revenue and ROAS forecasts for 30/60/90-day planning windows at channel, campaign-type, and campaign granularity — with AI-assisted causal summaries.

---

## Python version

**Python 3.14** (falls back to `python3` if not found). Tested on Python 3.12+.

---

## How to run (one command)

```bash
./run.sh                                               # local defaults
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv   # explicit (grader form)
./run.sh <DATA_DIR> <MODEL_PATH> <OUTPUT_PATH>          # general form
```

The grader runs exactly: `./run.sh ./data ./pickle/model.pkl ./output/predictions.csv`

Output is written to `OUTPUT_PATH` (fresh on every run, never appended).

---

## Argument contract

| Position | Argument | Default |
|---|---|---|
| 1 | `DATA_DIR` — folder containing channel CSVs | `./data` |
| 2 | `MODEL_PATH` — path to pickled model | `./pickle/model.pkl` |
| 3 | `OUTPUT_PATH` — full path for predictions CSV | `./output/predictions.csv` |

Optional env-var override:
```bash
HORIZON=60 ./run.sh    # forecast window: 30 (default), 60, or 90 days
```

---

## Project structure

```
.
├── run.sh                  # Single entry point (required by submission guide)
├── requirements.txt        # Pinned dependencies
├── README.md               # This file
├── METHODOLOGY.md          # Full methodology / model / assumptions / AI strategy
├── ARCHITECTURE.md         # Stack overview and pipeline diagram
├── data/                   # Input CSVs (replaced by grader at test time)
│   ├── google_ads_campaign_stats.csv
│   ├── bing_campaign_stats.csv
│   └── meta_ads_campaign_stats.csv
├── pickle/                 # Trained model artifact (committed)
│   └── model.pkl           # ForecastBundle (GBR + per-series stats)
├── output/                 # Generated at run time (not committed)
│   ├── predictions.csv     # Primary output scored by grader
│   ├── forecast_report.json
│   └── forecast_summary.md
└── src/
    ├── ingest.py           # CSV loading, schema validation, normalisation
    ├── features.py         # Feature engineering (calendar, lag, rolling)
    ├── forecasting.py      # Statistical + GBR model fitting and forecasting
    ├── ai_interpret.py     # Offline narrative NLG + optional Anthropic API hook
    ├── train.py            # Trains model bundle → pickle/model.pkl
    ├── predict.py          # Loads model, generates predictions.csv
    └── cache/              # Intermediate data cache (auto-generated, not committed)
```

---

## What run.sh does

1. Creates `.venv/` and installs pinned dependencies from `requirements.txt` (one-time).
2. **Trains** the forecasting bundle on whatever CSVs are in `DATA_DIR` → saves to `MODEL_PATH`. This is required because our model (trend + day-of-week seasonality + residual-bootstrap uncertainty) is data-adaptive — it must be fitted to the held-out test data provided at run time.
3. **Predicts** probabilistic revenue and ROAS forecasts → writes `OUTPUT_PATH`.

No interactive prompts. No network calls at run time. Seeds fixed at 42.

---

## Input data requirements

CSVs in `data/` are auto-detected by filename heuristic (must contain `google`, `bing`/`microsoft`, or `meta`/`facebook`). Any combination of 1–3 channels is accepted. Schema is validated before training begins.

| Channel | Expected filename contains |
|---|---|
| Google Ads | `google` |
| Bing / MS Ads | `bing`, `microsoft`, or `ms_ads` |
| Meta Ads | `meta`, `facebook`, or `fb_ads` |

---

## Output format (`predictions.csv`)

| Column | Description |
|---|---|
| `level` | `channel`, `campaign_type`, `campaign`, or `blended_total` |
| `channel` | Ad platform name (`Google Ads`, `Bing Ads`, `Meta Ads`, or `ALL`) |
| `campaign_type` | e.g. `Search`, `Shopping`, `Social` (empty for channel/blended rows) |
| `campaign_name` | Individual campaign name (empty for non-campaign rows) |
| `horizon_days` | Planning window: 30, 60, or 90 |
| `revenue_p10/p50/p90` | Probabilistic revenue range (USD) |
| `spend_assumed` | Spend assumption used for the window (USD) |
| `roas_p10/p50/p90` | Projected ROAS range |
| `historical_roas_p10/p50/p90` | Historical ROAS reference percentiles |
| `n_historical_obs` | Number of historical days used to fit this series |

---

## Reproducibility

- Seeds: `SEED = 42` fixed globally (NumPy RNG + bootstrap RNG)
- All paths are relative or resolved to absolute from `$BASH_SOURCE[0]`
- Re-running with identical inputs produces byte-identical `predictions.csv`

See `METHODOLOGY.md` for full model documentation.
