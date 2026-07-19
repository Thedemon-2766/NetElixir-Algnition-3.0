"""
Data Ingestion Module
=====================
Loads, normalizes, and validates campaign CSVs from any ad platform.

KEY DESIGN: column mapping is done by DTYPE + COLUMN NAME PATTERN analysis,
not by hardcoded per-channel schema dicts. This means the module works on
any CSV whose columns follow common naming conventions (e.g. 'spend', 'cost',
'revenue', 'clicks', 'impressions', 'date', 'campaign_id', etc.) regardless
of which platform exported it.

Detection strategy
------------------
For each CSV the detector inspects:
  • pandas dtype  (numeric vs string)
  • column name keywords  (e.g. 'spend', 'cost', 'revenue', 'click')
  • cardinality relative to row count (date col has many unique values;
    budget col has very few; campaign_type col has < 10 unique values)
  • parsability as %Y-%m-%d date

Special cases handled:
  • Google Ads stores spend as cost_micros (integer, needs ÷ 1,000,000) —
    detected when a numeric 'cost' column has values >> typical spend range.
  • When multiple candidates match a role (e.g. both 'conversions' and
    'conversions_value' match revenue), the one with higher cardinality /
    the one whose name contains 'value' is preferred.
  • If no campaign_type column is found, the type is inferred from filename
    keywords ('meta', 'facebook' → 'SOCIAL') or defaulted to 'Other'.
  • Duplicate files (same column fingerprint) are rejected before loading.
"""

import hashlib
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── Column-role detection ────────────────────────────────────────────────────────

def _str_cols(df: pd.DataFrame) -> pd.Index:
    """Return columns whose dtype is object or string (pandas-version safe)."""
    try:
        return df.select_dtypes(include=["object", "string"]).columns
    except TypeError:
        return df.select_dtypes(include=["object"]).columns


def _num_cols(df: pd.DataFrame) -> pd.Index:
    return df.select_dtypes(include=np.number).columns


def _is_date_col(series: pd.Series, n: int) -> bool:
    """
    True if the column looks like a daily date column.
    Criteria:
      - string/object dtype
      - parseable as %Y-%m-%d (checked on first 10 non-null values)
      - at least 5 unique values (rules out constant/flag columns)
    Note: cardinality relative to n is NOT used because in campaign-level data
    each date repeats once per campaign, so nuniq << n is expected and normal.
    """
    if not (pd.api.types.is_object_dtype(series) or
            pd.api.types.is_string_dtype(series)):
        return False
    sample = series.dropna().head(10)
    if len(sample) == 0:
        return False
    try:
        pd.to_datetime(sample, format="%Y-%m-%d")
        return series.nunique() >= 5   # absolute minimum — not relative to n
    except Exception:
        return False


def _detect_columns(df: pd.DataFrame, filename: str) -> dict:
    """
    Inspect df by dtype + keyword + cardinality and return a mapping:
        role → column_name  (or None if not found)

    Roles: date, campaign_id, campaign_name, campaign_type,
           spend, revenue, clicks, impressions, conversions, daily_budget
    """
    n       = max(len(df), 1)
    str_c   = list(_str_cols(df))
    num_c   = list(_num_cols(df))
    mapping = {}

    # ── date ────────────────────────────────────────────────────────────────
    mapping["date"] = next(
        (c for c in str_c if _is_date_col(df[c], n)), None
    )

    # ── campaign_id (numeric, low cardinality, name contains 'id') ──────────
    id_cands = [c for c in num_c
                if "id" in c.lower() and df[c].nunique() < 200]
    mapping["campaign_id"] = id_cands[0] if id_cands else None

    # ── campaign_name (string, name contains 'name') ────────────────────────
    name_cands = [c for c in str_c if "name" in c.lower()]
    mapping["campaign_name"] = name_cands[0] if name_cands else None

    # ── campaign_type (string, very low cardinality ≤ 15, name contains 'type') ─
    type_cands = [c for c in str_c
                  if "type" in c.lower() and df[c].nunique() <= 15]
    mapping["campaign_type"] = type_cands[0] if type_cands else None

    # ── spend (numeric, keyword: spend or cost) ─────────────────────────────
    # Prefer exact 'spend' first, then anything with 'spend', then 'cost'
    spend_cands = (
        [c for c in num_c if c.lower() == "spend"] or
        [c for c in num_c if "spend" in c.lower()] or
        [c for c in num_c if "cost" in c.lower()]
    )
    if spend_cands:
        # Pick the one with highest cardinality (most variation = actual spend)
        mapping["spend"] = max(spend_cands, key=lambda c: df[c].nunique())
    else:
        mapping["spend"] = None

    # ── revenue (numeric, keyword: revenue or conversions_value or conversion) ─
    # Priority: 'revenue' exact > contains 'revenue' > contains 'value' >
    #           contains 'conversion' (not 'conversions' alone — that's count)
    rev_cands = (
        [c for c in num_c if c.lower() == "revenue"] or
        [c for c in num_c if "revenue" in c.lower()] or
        [c for c in num_c if "value" in c.lower()] or
        [c for c in num_c if "conversion" in c.lower()]
    )
    if rev_cands:
        mapping["revenue"] = max(rev_cands, key=lambda c: df[c].nunique())
    else:
        mapping["revenue"] = None

    # ── clicks (numeric, keyword: click) ────────────────────────────────────
    click_cands = [c for c in num_c if "click" in c.lower()]
    mapping["clicks"] = click_cands[0] if click_cands else None

    # ── impressions (numeric, keyword: impression) ───────────────────────────
    imp_cands = [c for c in num_c if "impression" in c.lower()]
    mapping["impressions"] = imp_cands[0] if imp_cands else None

    # ── conversions (numeric, keyword: conversion — pick lower-cardinality one
    #    so we favour a count column over a value column) ────────────────────
    conv_cands = [c for c in num_c if "conversion" in c.lower()]
    if conv_cands:
        # Prefer column named exactly 'conversions' or containing 'conversions'
        exact = [c for c in conv_cands
                 if c.lower() in ("conversions", "conversion")
                 and "value" not in c.lower()]
        mapping["conversions"] = (exact[0] if exact
                                  else min(conv_cands,
                                           key=lambda c: df[c].nunique()))
    else:
        mapping["conversions"] = None

    # ── daily_budget (numeric, very low cardinality ≤ 10, keyword: budget) ──
    budget_cands = [c for c in num_c
                    if "budget" in c.lower() and df[c].nunique() <= 20]
    mapping["daily_budget"] = budget_cands[0] if budget_cands else None

    return mapping


def _is_micros(series: pd.Series) -> bool:
    """
    Heuristic: returns True when a numeric series looks like micros
    (i.e. actual spend in millionths — Google Ads cost_micros pattern).
    Triggered when the median non-zero value is > 50,000, which is implausible
    as a dollar/euro spend figure but makes sense as micro-currency units.
    """
    non_zero = series[series > 0]
    if len(non_zero) == 0:
        return False
    return float(non_zero.median()) > 50_000


# ── Pre-flight schema validation ─────────────────────────────────────────────────

def validate_schemas(paths: dict[str, str]) -> None:
    """
    Pre-flight check on all files before any data is loaded.
    Verifies:
      1. Each CSV has the minimum required detectable columns
         (date, campaign_id/name, spend, revenue, clicks, impressions).
      2. No two files have an identical column fingerprint (duplicate file guard).
    Raises ValueError with a consolidated report of ALL issues found.
    """
    REQUIRED_ROLES = ["date", "spend", "revenue", "clicks", "impressions"]
    errors       = []
    fingerprints = {}

    for ch_key, path in paths.items():
        try:
            # Read enough rows for reliable detection — 500 covers most edge cases
            # while staying fast even on large CSVs (no full file scan needed).
            header_df = pd.read_csv(path, nrows=500)
            header_df = _drop_index_col(header_df)
        except Exception as exc:
            errors.append(f"  [{ch_key}] Cannot read '{os.path.basename(path)}': {exc}")
            continue

        mapping = _detect_columns(header_df, os.path.basename(path))

        missing_roles = [r for r in REQUIRED_ROLES if mapping.get(r) is None]
        if missing_roles:
            errors.append(
                f"  [{ch_key}] '{os.path.basename(path)}': "
                f"could not detect columns for roles {missing_roles}.\n"
                f"    Detected mapping : {mapping}\n"
                f"    Available columns: {list(header_df.columns)}"
            )

        # Duplicate-file guard (same column set → same file supplied twice)
        actual_cols     = list(header_df.columns)
        col_fingerprint = hashlib.md5(
            ",".join(sorted(actual_cols)).encode()
        ).hexdigest()
        if col_fingerprint in fingerprints:
            errors.append(
                f"  [{ch_key}] '{os.path.basename(path)}' has the same column set as "
                f"[{fingerprints[col_fingerprint]}]. Each channel needs its own CSV."
            )
        else:
            fingerprints[col_fingerprint] = ch_key

    if errors:
        raise ValueError(
            "Schema validation failed — fix these issues before re-running:\n"
            + "\n".join(errors)
        )

    log.info(f"Schema pre-flight passed for {len(paths)} channel(s).")


# ── Cache helpers ─────────────────────────────────────────────────────────────────

def _cache_key(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_path(channel: str, key: str) -> Path:
    return CACHE_DIR / f"{channel}_{key}.pkl"


def _load_cached(channel: str, key: str) -> pd.DataFrame | None:
    p = _cache_path(channel, key)
    if p.exists():
        log.info(f"  Cache hit for {channel}")
        return pd.read_pickle(p)
    return None


def _save_cache(df: pd.DataFrame, channel: str, key: str):
    df.to_pickle(_cache_path(channel, key))
    log.info(f"  Cached {channel} → {_cache_path(channel, key).name}")


# ── Low-level helpers ─────────────────────────────────────────────────────────────

def _drop_index_col(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed:")])


def _sanitise_pipe(series: pd.Series) -> pd.Series:
    """Replace | (composite key separator) with a dash."""
    return series.astype(str).str.replace("|", "-", regex=False)


def _to_num(series: pd.Series, fill: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill)


def _infer_campaign_type_label(filename: str) -> str:
    """Fallback campaign type when no type column is found in the CSV."""
    fname = filename.lower()
    if any(kw in fname for kw in ("meta", "facebook", "instagram", "fb")):
        return "SOCIAL"
    if any(kw in fname for kw in ("bing", "microsoft", "msads")):
        return "SEARCH"
    if "google" in fname:
        return "SEARCH"
    return "UNKNOWN"


def _infer_channel_label(filename: str) -> str:
    """Derive a human-readable channel label from the filename."""
    fname = filename.lower()
    if "google" in fname:
        return "Google Ads"
    if any(kw in fname for kw in ("bing", "microsoft", "msads")):
        return "Bing Ads"
    if any(kw in fname for kw in ("meta", "facebook", "instagram", "fb")):
        return "Meta Ads"
    # Fallback: capitalise the stem
    stem = os.path.splitext(os.path.basename(filename))[0]
    return stem.replace("_", " ").title()


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate rows with identical (date, campaign_id, channel).
    channel is a groupby key so it must NOT appear in the agg dict.
    """
    num_cols = ["spend", "revenue", "clicks", "impressions", "conversions", "daily_budget"]
    cat_cols = ["campaign_name", "campaign_type"]
    agg = {c: "sum"   for c in num_cols if c in df.columns}
    agg.update({c: "first" for c in cat_cols if c in df.columns})
    before  = len(df)
    deduped = (df.groupby(["date", "campaign_id", "channel"], sort=False)
                 .agg(agg).reset_index())
    n_dropped = before - len(deduped)
    if n_dropped:
        log.warning(f"  Deduplicated {n_dropped} duplicate rows "
                    f"(same date + campaign_id + channel)")
    return deduped


# ── Generic CSV loader ────────────────────────────────────────────────────────────

def load_csv(path: str, channel_key: str) -> pd.DataFrame:
    """
    Load one ad-platform CSV, auto-detect its columns by dtype + keyword,
    and normalise into the unified schema.

    Parameters
    ----------
    path        : absolute or relative path to the CSV file
    channel_key : short key used for caching ('google', 'bing', 'meta', etc.)
    """
    key    = _cache_key(path)
    cached = _load_cached(channel_key, key)
    if cached is not None:
        return cached

    filename = os.path.basename(path)
    log.info(f"Loading {channel_key} from {path}")
    raw = _drop_index_col(pd.read_csv(path))

    m = _detect_columns(raw, filename)

    log.info(f"  Detected column mapping for '{filename}':")
    for role, col in m.items():
        log.info(f"    {role:15s} → {col}")

    # ── Build unified DataFrame ───────────────────────────────────────────────
    df = pd.DataFrame()

    # date
    df["date"] = pd.to_datetime(raw[m["date"]], errors="coerce") if m["date"] else pd.NaT

    # campaign_id (fallback: row index as string)
    df["campaign_id"] = (
        _sanitise_pipe(raw[m["campaign_id"]]) if m["campaign_id"]
        else raw.index.astype(str)
    )

    # campaign_name
    df["campaign_name"] = (
        _sanitise_pipe(raw[m["campaign_name"]].fillna("Unknown"))
        if m["campaign_name"] else "Unknown"
    )

    # campaign_type
    if m["campaign_type"]:
        df["campaign_type"] = raw[m["campaign_type"]].fillna("UNKNOWN").str.upper()
    else:
        inferred = _infer_campaign_type_label(filename)
        df["campaign_type"] = inferred
        log.info(f"  No campaign_type column found — defaulting to '{inferred}'")

    # spend — detect micros and convert if needed
    if m["spend"]:
        spend_raw = _to_num(raw[m["spend"]])
        if _is_micros(spend_raw):
            log.info(f"  '{m['spend']}' looks like micros → dividing by 1,000,000")
            df["spend"] = spend_raw / 1_000_000
        else:
            df["spend"] = spend_raw
    else:
        df["spend"] = 0.0

    # revenue
    df["revenue"]      = _to_num(raw[m["revenue"]])      if m["revenue"]      else 0.0

    # clicks
    df["clicks"]       = _to_num(raw[m["clicks"]])       if m["clicks"]       else 0.0

    # impressions
    df["impressions"]  = _to_num(raw[m["impressions"]])  if m["impressions"]  else 0.0

    # conversions (fallback to revenue as a proxy if no conversions column found)
    if m["conversions"]:
        df["conversions"] = _to_num(raw[m["conversions"]])
    else:
        df["conversions"] = df["revenue"].copy()
        log.info("  No conversions column found — using revenue as proxy")

    # daily_budget
    df["daily_budget"] = _to_num(raw[m["daily_budget"]]) if m["daily_budget"] else 0.0

    # channel label
    df["channel"] = _infer_channel_label(filename)

    df = _clean_common(df)
    df = _dedup(df)
    _save_cache(df, channel_key, key)
    return df


# ── Common cleaning ───────────────────────────────────────────────────────────────

def _clean_common(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    for col in ["spend", "revenue", "clicks", "impressions", "conversions", "daily_budget"]:
        if col in df.columns:
            df[col] = _to_num(df[col]).clip(lower=0)

    df["roas"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], 0.0)
    df["cpc"]  = np.where(df["clicks"] > 0, df["spend"] / df["clicks"], 0.0)
    df["ctr"]  = np.where(df["impressions"] > 0,
                          df["clicks"] / df["impressions"] * 100, 0.0)
    df["cvr"]  = np.where(df["clicks"] > 0,
                          df["conversions"] / df["clicks"] * 100, 0.0)

    type_map = {
        "SEARCH": "Search", "SHOPPING": "Shopping", "DISPLAY": "Display",
        "VIDEO": "Video", "PERFORMANCE_MAX": "Performance Max",
        "SOCIAL": "Social", "UNKNOWN": "Other",
    }
    df["campaign_type"] = df["campaign_type"].map(type_map).fillna("Other")
    return df.sort_values("date").reset_index(drop=True)


# ── Post-load validation ──────────────────────────────────────────────────────────

def validate(df: pd.DataFrame, expected_channels: list[str]) -> dict:
    report = {"warnings": [], "errors": []}

    zero_budget = df[df["daily_budget"] == 0]["campaign_name"].nunique()
    if zero_budget:
        report["warnings"].append(
            f"{zero_budget} campaigns have no budget info "
            f"(historical spend average will be used)")

    odd = df[(df["revenue"] > 0) & (df["spend"] == 0)]
    if len(odd):
        report["warnings"].append(
            f"{len(odd)} rows: revenue > 0 but spend = 0 "
            f"(possible attribution lag)")

    present = set(df["channel"].unique())
    for ch in expected_channels:
        if ch not in present:
            report["warnings"].append(
                f"Channel '{ch}' not in loaded data — forecasts will exclude it")

    report["date_range"] = {
        "min": str(df["date"].min().date()),
        "max": str(df["date"].max().date()),
        "days": (df["date"].max() - df["date"].min()).days + 1,
    }
    report["row_count"]      = len(df)
    report["campaign_count"] = df["campaign_name"].nunique()
    report["channel_count"]  = df["channel"].nunique()
    return report


# ── Public API ────────────────────────────────────────────────────────────────────

def load_all(paths: dict[str, str]) -> pd.DataFrame:
    """
    Pre-flight schema check, then load any subset of channel CSVs.

    Parameters
    ----------
    paths : dict  arbitrary string key → file path (1–N channels)
                  e.g. {"google": "data/google_ads.csv", "bing": "data/bing.csv"}

    Returns
    -------
    pd.DataFrame  Unified, channel-tagged dataset ready for feature engineering.
    """
    if not paths:
        raise ValueError("paths dict is empty — provide at least one channel CSV.")

    validate_schemas(paths)

    frames  = [load_csv(path, ch_key) for ch_key, path in paths.items()]
    combined = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)

    report = validate(combined, ["Google Ads", "Bing Ads", "Meta Ads"])
    log.info("=== Validation Report ===")
    log.info(f"  Rows       : {report['row_count']:,}")
    log.info(f"  Campaigns  : {report['campaign_count']}")
    log.info(f"  Channels   : {report['channel_count']} / {len(paths)} loaded")
    log.info(f"  Date range : {report['date_range']['min']} → "
             f"{report['date_range']['max']}")
    for w in report.get("warnings", []):
        log.warning(f"  ⚠  {w}")
    for e in report.get("errors", []):
        log.error(f"  ✖  {e}")

    return combined
