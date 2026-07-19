# Forecasting Methodology & System Documentation

**Project:** Multi-Channel Ad Spend Revenue & ROAS Forecasting
**Scope:** Google Ads, Bing (Microsoft) Ads, Meta Ads — campaign-level ingestion through aggregate-period probabilistic forecasting

---

## 1. Problem Framing

This system forecasts **expected e-commerce revenue** (aggregated across Google Ads, Bing Ads, and Meta Ads) and **blended ROAS** over a future planning window — 30, 60, or 90 days — at three levels of granularity: channel, campaign type, and individual campaign.

It is explicitly **not**:
- A custom attribution engine. Each platform's own reported `revenue`/`conversion value` is treated as ground truth.
- A full Media Mix Model (MMM). It does not estimate cross-channel lift, halo effects, ad-stock decay, or saturation curves.
- A daily forecaster. Outputs are **aggregate totals/averages over the chosen window**, not day-by-day predictions.
- A single-number predictor. Every forecast is expressed as a **probabilistic range (P10 / P50 / P90)**.
- ROAS-constrained or optimization-driven. The system describes likely outcomes; it does not recommend budget reallocation against a ROAS target.

---

## 2. Data Preprocessing Logic

### 2.1 Schema normalization
Each platform export uses different column names and units (see `src/ingest.py`):

| Concept | Google Ads | Bing Ads | Meta Ads |
|---|---|---|---|
| Date | `segments_date` | `TimePeriod` | `date_start` |
| Spend | `metrics_cost_micros` ÷ 1,000,000 | `Spend` | `spend` |
| Revenue | `metrics_conversions_value` | `Revenue` | `conversion` (used as revenue proxy) |
| Campaign type | `campaign_advertising_channel_type` | `CampaignType` | not provided → hardcoded `"Social"` |
| Budget | `campaign_budget_amount` | `DailyBudget` | `daily_budget` |

All three are mapped into one unified schema: `date, campaign_id, campaign_name, campaign_type, channel, spend, revenue, clicks, impressions, conversions, daily_budget`, plus derived `roas`, `cpc`, `ctr`, `cvr`.

### 2.2 Cleaning rules
- Rows with unparseable dates are dropped.
- Negative values in spend/revenue/clicks/impressions/conversions/budget are clamped to zero (data entry artifacts).
- Campaign-type labels are normalized into a consistent taxonomy (`Search`, `Shopping`, `Display`, `Video`, `Performance Max`, `Social`, `Other`).
- Rows with revenue > 0 but spend = 0 are flagged as warnings (typically attribution lag near a reporting window edge) but retained, since dropping them would understate revenue.

### 2.3 Validation
`src/ingest.py::validate()` runs automatically on every load and reports: row/campaign/channel counts, date coverage, missing-budget campaigns, and the revenue-without-spend anomaly count. Errors (e.g. a missing channel file) halt the pipeline; warnings are logged but non-blocking.

### 2.4 Feature engineering
`src/features.py` builds three aggregation levels — channel-daily, campaign-type-daily, campaign-daily — each enriched with:
- Calendar features: day-of-week, month, week-of-year, quarter, weekend flag, and sine/cosine cyclical encodings (so the model treats "Sunday" and "Monday" as numerically adjacent, not 6 units apart).
- Rolling means/std (7/14/28-day windows) on spend, revenue, and ROAS.
- Lag features (1/7/14-day) on spend and revenue.
- Budget utilization ratio (spend ÷ daily_budget).

All intermediate aggregates are cached to `src/cache/` as pickled DataFrames, keyed by an MD5 hash of the source CSV bytes, so re-running the pipeline on unchanged inputs skips re-aggregation.

---

## 3. Model Selection

Two complementary components are used per series (channel / campaign-type / campaign), deliberately **not** a single complex model:

### 3.1 Statistical baseline (primary forecasting mechanism)
For every series, `src/forecasting.py::_fit_series()` fits:
- A **linear trend** (ordinary least squares) over the time index.
- **Day-of-week seasonal multipliers**, computed as the median ratio of actual-to-trend revenue per weekday, clipped to [0.3, 3.0] to avoid one noisy day distorting the whole multiplier, then re-normalized to mean 1.
- The **residual pool**: actual revenue minus the trend×seasonal fit, for every historical day (capped at the most recent 180 observations).
- An **empirical ROAS distribution** (P10/P50/P90 percentiles of daily ROAS, spend-days only).

This was chosen over ARIMA/Prophet/Holt-Winters library implementations because: (a) many series here are short (some campaigns have only weeks of history) where classical seasonal decomposition is unstable, (b) it has zero additional dependencies beyond NumPy/Pandas, keeping the offline footprint minimal, and (c) its parameters are fully transparent and auditable — important for a planning tool stakeholders need to trust.

### 3.2 Gradient-boosted cross-check (channel-level only)
`src/forecasting.py::fit_gbr_model()` trains a `sklearn.ensemble.GradientBoostingRegressor` on the engineered feature matrix (calendar + rolling + lag features, one-hot encoded channel) to predict daily revenue, validated with `TimeSeriesSplit` (not random K-fold, to avoid leaking future information into training folds). Its cross-validated MAE is reported in `model/training_summary.json` as a model-quality diagnostic. This model is **not** currently blended numerically into the per-series forecast output (to keep forecast provenance simple and explainable) — it serves as an independent sanity-check signal during training, and is persisted in the bundle for future extension.

### 3.3 Why not a heavier model
Deep learning (LSTM/Transformer) time-series models were deliberately avoided: the dataset (per-campaign daily series, often a few hundred points) is too small to train such models without severe overfitting, they add heavy dependencies inappropriate for a fully offline, pip-installable tool, and they sacrifice the interpretability that this trend+seasonal+residual-bootstrap approach provides.

---

## 4. Uncertainty Estimation

Point forecasts alone are not provided. For every series and horizon:

1. The point forecast is computed as `trend(t) × seasonal_multiplier(day_of_week)` for each future day in the window, summed to a window total.
2. **Residual bootstrap**: 2,000 bootstrap resamples are drawn (with replacement) from the historical residual pool, one residual per future day per resample; each resample's noised daily forecasts are summed to a window total, clamped at zero.
3. P10/P50/P90 are read off the empirical distribution of these 2,000 window totals.
4. If a series has fewer than 5 historical residuals (very new campaigns), a parametric fallback is used instead: `P50 ± 1.2816 × σ√horizon` (the 1.2816 z-score corresponds to the 10th/90th percentile of a normal distribution).

ROAS ranges are derived two ways and reconciled: (a) revenue-percentile ÷ assumed spend, and (b) the series' own historical ROAS percentile distribution — both are reported in `prediction.csv` (`roas_p10/p50/p90` vs `historical_roas_p10/p50/p90`) so a planner can see whether the forward-looking range is wider or narrower than history.

All randomness (NumPy global seed and a dedicated `np.random.default_rng(42)` for bootstrap) is fixed at `SEED = 42` throughout `train.py`, `features.py`, and `forecasting.py`, so re-running the pipeline on unchanged inputs reproduces byte-identical `prediction.csv` output.

---

## 5. Spend Assumption for ROAS

The system does not accept external future budget inputs. ROAS is derived internally from the series' own historical daily spend mean × the horizon window length. This keeps the pipeline fully offline and avoids implying causal spend→revenue relationships the model does not support. Revenue ranges are produced purely by the trend+seasonal+residual bootstrap model; ROAS is computed as revenue_pX ÷ assumed_spend.

---

## 6. AI Integration Strategy

The system must run **fully offline**, so the default "AI-assisted interpretation" layer (`src/ai_interpret.py`) is a **deterministic, template-driven narrative generator** that encodes the same reasoning steps an analyst (or an LLM) would apply:

- Trend classification (rising / stable / declining) from the fitted slope relative to series mean.
- Confidence classification (high / moderate / low) from the P10–P90 spread relative to P50.
- ROAS trajectory classification (improving / softening / steady) comparing forecast ROAS to historical ROAS.
- These are composed into 2–4 sentence per-series summaries and a portfolio-level executive narrative, all using only numbers already computed by the statistical model — no hallucination risk, fully reproducible, zero cost, zero network dependency.


---

## 7. Assumptions

- Each platform's reported revenue/conversion-value is accurate and is the correct unit of "expected e-commerce revenue" attributable to that platform (no de-duplication of cross-channel-influenced conversions is attempted).
- Daily granularity in the source CSVs is the finest available; intra-day patterns are out of scope.
- Future budget inputs represent planned, not guaranteed, spend.
- Historical day-of-week and trend patterns are assumed to persist into the forecast window, absent any external signal to the contrary (no holiday calendar, promotional calendar, or competitor data is ingested).
- Campaigns with very short history (a few weeks) receive wider, less reliable ranges; this is reflected in the "confidence" label in the AI summary, not suppressed.

## 8. Limitations

- No causal attribution or incrementality modeling — correlation between spend and revenue within a channel/campaign is not validated against a holdout/geo experiment.
- No explicit handling of structural breaks (e.g. a campaign pausing entirely, a platform policy change, a tracking outage) beyond what the residual bootstrap implicitly captures from past volatility.
- Campaign-type and campaign-level budget scaling under a supplied channel budget uses a simple proportional-historical-share heuristic, not a learned allocation model.
- The GBR cross-check model is trained and reported for diagnostic purposes but is not yet blended into the published forecast; this is a documented direction for future iteration, not a current limitation hidden from the user.
- Forecast quality depends directly on historical data volume and consistency per series; the system does not refuse to forecast low-data series, but flags them as lower-confidence via wider P10-P90 bands.

---

## 9. Reproducibility & Operations

- **Single entry point**: `./run.sh` creates a local virtual environment, installs pinned dependencies from `requirements.txt`, trains the model bundle, and generates forecasts — no manual steps, no notebooks.
- **Seeds**: `SEED = 42` fixed globally (NumPy global RNG and a dedicated bootstrap RNG).
- **Paths**: all paths used by the pipeline are relative to the script locations; no absolute paths are hardcoded.
- **Persistence**: the trained model is a single pickled `ForecastBundle` dataclass at `model/model.pkl`, containing all fitted `SeriesModel` objects (channel/campaign-type/campaign), the GBR model, and training metadata.
- **Output overwrite**: `predict.py` explicitly deletes any pre-existing `output/prediction.csv` before writing a fresh one (never appends).
- **Caching**: `src/cache/` stores intermediate ingestion and feature-engineering artifacts keyed by source-file content hash, so repeated runs against unchanged CSVs skip redundant computation.
