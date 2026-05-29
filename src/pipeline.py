"""Traffic demand prediction pipeline.

Split:
  train = day 48 (slots 0-95) + day 49 (slots 0-8)
  test  = day 49 (slots 9-55)
Strategy: gradient-boosted regression on engineered features. Lag features
from day-48 same-slot demand are expected to dominate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


# ---------- geohash decoding ----------
_GEO_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_GEO_DECODE = {c: i for i, c in enumerate(_GEO_BASE32)}


def decode_geohash(gh: str) -> tuple[float, float]:
    lat_lo, lat_hi = -90.0, 90.0
    lng_lo, lng_hi = -180.0, 180.0
    even = True
    for ch in gh:
        bits = _GEO_DECODE[ch]
        for mask in (16, 8, 4, 2, 1):
            if even:
                mid = (lng_lo + lng_hi) / 2
                if bits & mask:
                    lng_lo = mid
                else:
                    lng_hi = mid
            else:
                mid = (lat_lo + lat_hi) / 2
                if bits & mask:
                    lat_lo = mid
                else:
                    lat_hi = mid
            even = not even
    return (lat_lo + lat_hi) / 2, (lng_lo + lng_hi) / 2


def add_latlng(df: pd.DataFrame) -> pd.DataFrame:
    uniq = df["geohash"].unique()
    coords = {g: decode_geohash(g) for g in uniq}
    df["lat"] = df["geohash"].map(lambda g: coords[g][0])
    df["lng"] = df["geohash"].map(lambda g: coords[g][1])
    return df


# ---------- timestamp parsing ----------
def parse_time(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["timestamp"].str.split(":", expand=True)
    df["hour"] = parts[0].astype(int)
    df["minute"] = parts[1].astype(int)
    df["slot"] = df["hour"] * 4 + df["minute"] // 15
    df["slot_block"] = df["slot"] // 8  # 12 blocks of 2 hours
    # cyclical
    df["slot_sin"] = np.sin(2 * np.pi * df["slot"] / 96)
    df["slot_cos"] = np.cos(2 * np.pi * df["slot"] / 96)
    df["abs_time"] = df["day"] * 96 + df["slot"]  # monotonic time index
    return df


# ---------- categorical encoding ----------
_ROAD_MAP = {"Residential": 0, "Street": 1, "Highway": 2}
_WEATHER_MAP = {"Sunny": 0, "Foggy": 1, "Rainy": 2, "Snowy": 3}


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df["RoadType_enc"] = df["RoadType"].map(_ROAD_MAP)
    df["Weather_enc"] = df["Weather"].map(_WEATHER_MAP)
    df["LargeVehicles_enc"] = (df["LargeVehicles"] == "Allowed").astype(int)
    df["Landmarks_enc"] = (df["Landmarks"] == "Yes").astype(int)
    return df


# ---------- per-geohash imputation for "static" features ----------
def impute_static_by_geohash(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill missing RoadType/Weather/Temperature using the same geohash's other rows.

    RoadType is geohash-static in this dataset; Weather/Temp vary but per-geohash
    mode/mean is a strong baseline imputation.
    """
    full = pd.concat([train, test], ignore_index=True, sort=False)

    # mode per geohash for RoadType_enc; fall back to global mode
    road_mode = (
        full.dropna(subset=["RoadType_enc"]).groupby("geohash")["RoadType_enc"].agg(lambda s: s.mode().iloc[0])
    )
    global_road_mode = full["RoadType_enc"].mode().iloc[0]

    def fill_road(df):
        mask = df["RoadType_enc"].isna()
        if mask.any():
            df.loc[mask, "RoadType_enc"] = df.loc[mask, "geohash"].map(road_mode).fillna(global_road_mode)
        return df

    train = fill_road(train)
    test = fill_road(test)

    # Weather mode per geohash
    w_mode = full.dropna(subset=["Weather_enc"]).groupby("geohash")["Weather_enc"].agg(lambda s: s.mode().iloc[0])
    global_w_mode = full["Weather_enc"].mode().iloc[0]
    for df in (train, test):
        mask = df["Weather_enc"].isna()
        if mask.any():
            df.loc[mask, "Weather_enc"] = df.loc[mask, "geohash"].map(w_mode).fillna(global_w_mode)

    # Temperature mean per geohash
    t_mean = full.dropna(subset=["Temperature"]).groupby("geohash")["Temperature"].mean()
    global_t_mean = full["Temperature"].mean()
    for df in (train, test):
        mask = df["Temperature"].isna()
        if mask.any():
            df.loc[mask, "Temperature"] = df.loc[mask, "geohash"].map(t_mean).fillna(global_t_mean)
    return train, test


# ---------- target-encoded / lag features ----------
def add_geohash_aggregates_oof(train_src: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, n_folds: int = 5) -> None:
    """K-fold OOF target encoding for per-geohash aggregates on training rows.

    Training rows: for each fold, compute aggregates from the other K-1 folds
    of train_src and apply to the held-out fold of train_df.
    Test rows: use the full train_src to compute aggregates.

    This eliminates the row's own target from its own per-geohash mean.
    """
    rng = np.random.default_rng(0)
    n = len(train_src)
    fold_ids = rng.integers(0, n_folds, size=n)
    global_mean = train_src["demand"].mean()

    # Test gets full-train aggregates
    g_full = train_src.groupby("geohash")["demand"]
    test_df["geo_mean"] = test_df["geohash"].map(g_full.mean()).fillna(global_mean)
    test_df["geo_median"] = test_df["geohash"].map(g_full.median()).fillna(global_mean)
    test_df["geo_std"] = test_df["geohash"].map(g_full.std()).fillna(0.0)
    test_df["geo_max"] = test_df["geohash"].map(g_full.max()).fillna(global_mean)

    # For train_df rows, only OOF if train_df == train_src (same set). Otherwise
    # treat train_df as "another" frame and use full aggregates (no leakage anyway).
    same_rows = (len(train_df) == n)
    if same_rows:
        geo_mean = np.full(n, np.nan)
        geo_median = np.full(n, np.nan)
        geo_std = np.full(n, np.nan)
        geo_max = np.full(n, np.nan)
        for k in range(n_folds):
            holdout = fold_ids == k
            src = train_src.iloc[~holdout]
            g = src.groupby("geohash")["demand"]
            m = g.mean()
            med = g.median()
            std = g.std()
            mx = g.max()
            ghs = train_src.iloc[holdout]["geohash"]
            geo_mean[holdout] = ghs.map(m).fillna(global_mean).to_numpy()
            geo_median[holdout] = ghs.map(med).fillna(global_mean).to_numpy()
            geo_std[holdout] = ghs.map(std).fillna(0.0).to_numpy()
            geo_max[holdout] = ghs.map(mx).fillna(global_mean).to_numpy()
        train_df["geo_mean"] = geo_mean
        train_df["geo_median"] = geo_median
        train_df["geo_std"] = geo_std
        train_df["geo_max"] = geo_max
    else:
        # train_df is a separate frame (e.g., validation); use full aggregates
        train_df["geo_mean"] = train_df["geohash"].map(g_full.mean()).fillna(global_mean)
        train_df["geo_median"] = train_df["geohash"].map(g_full.median()).fillna(global_mean)
        train_df["geo_std"] = train_df["geohash"].map(g_full.std()).fillna(0.0)
        train_df["geo_max"] = train_df["geohash"].map(g_full.max()).fillna(global_mean)


def add_day48_stats_oof(train_src: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, n_folds: int = 5) -> None:
    """K-fold OOF for day-48 per-geohash aggregates on training rows."""
    src48 = train_src[train_src["day"] == 48]
    rng = np.random.default_rng(1)
    fold_ids = rng.integers(0, n_folds, size=len(src48))

    g_full = src48.groupby("geohash")["demand"]
    glb = src48["demand"].mean()

    # Test: full aggregates
    for col, agg in [("d48_mean", "mean"), ("d48_median", "median"), ("d48_max", "max"), ("d48_std", "std")]:
        vals = getattr(g_full, agg)()
        test_df[col] = test_df["geohash"].map(vals).fillna(glb if col != "d48_std" else 0.0)
    test_df["d48_p90"] = test_df["geohash"].map(g_full.quantile(0.9)).fillna(glb)

    same_rows = (len(train_df) == len(train_src))
    if same_rows:
        # For day-48 training rows: OOF.  For day-49 rows: full aggregates (no leak).
        is_d48 = train_src["day"].to_numpy() == 48
        d48_mean_arr = np.full(len(train_src), np.nan)
        d48_med_arr = np.full(len(train_src), np.nan)
        d48_max_arr = np.full(len(train_src), np.nan)
        d48_std_arr = np.full(len(train_src), np.nan)
        d48_p90_arr = np.full(len(train_src), np.nan)
        # Fill day-49 rows from full aggregate
        d49_mask = ~is_d48
        if d49_mask.any():
            ghs = train_src.loc[d49_mask, "geohash"]
            d48_mean_arr[d49_mask] = ghs.map(g_full.mean()).fillna(glb).to_numpy()
            d48_med_arr[d49_mask] = ghs.map(g_full.median()).fillna(glb).to_numpy()
            d48_max_arr[d49_mask] = ghs.map(g_full.max()).fillna(glb).to_numpy()
            d48_std_arr[d49_mask] = ghs.map(g_full.std()).fillna(0.0).to_numpy()
            d48_p90_arr[d49_mask] = ghs.map(g_full.quantile(0.9)).fillna(glb).to_numpy()
        # OOF for day-48 rows
        src48_pos = np.where(is_d48)[0]
        for k in range(n_folds):
            holdout_local = fold_ids == k
            src_fold = src48.iloc[~holdout_local]
            g = src_fold.groupby("geohash")["demand"]
            m = g.mean()
            med = g.median()
            mx = g.max()
            std = g.std()
            p90 = g.quantile(0.9)
            global_pos_in_train = src48_pos[holdout_local]
            ghs = train_src.iloc[global_pos_in_train]["geohash"]
            d48_mean_arr[global_pos_in_train] = ghs.map(m).fillna(glb).to_numpy()
            d48_med_arr[global_pos_in_train] = ghs.map(med).fillna(glb).to_numpy()
            d48_max_arr[global_pos_in_train] = ghs.map(mx).fillna(glb).to_numpy()
            d48_std_arr[global_pos_in_train] = ghs.map(std).fillna(0.0).to_numpy()
            d48_p90_arr[global_pos_in_train] = ghs.map(p90).fillna(glb).to_numpy()
        train_df["d48_mean"] = d48_mean_arr
        train_df["d48_median"] = d48_med_arr
        train_df["d48_max"] = d48_max_arr
        train_df["d48_std"] = d48_std_arr
        train_df["d48_p90"] = d48_p90_arr
    else:
        train_df["d48_mean"] = train_df["geohash"].map(g_full.mean()).fillna(glb)
        train_df["d48_median"] = train_df["geohash"].map(g_full.median()).fillna(glb)
        train_df["d48_max"] = train_df["geohash"].map(g_full.max()).fillna(glb)
        train_df["d48_std"] = train_df["geohash"].map(g_full.std()).fillna(0.0)
        train_df["d48_p90"] = train_df["geohash"].map(g_full.quantile(0.9)).fillna(glb)


def add_geohash_aggregates(train_src: pd.DataFrame, dfs: list[pd.DataFrame]) -> None:
    """Add per-geohash demand aggregates computed from train_src (must not leak).

    For test predictions we use ALL of train; for offline validation, train_src
    should exclude the held-out slots.
    """
    g = train_src.groupby("geohash")["demand"]
    geo_mean = g.mean()
    geo_med = g.median()
    geo_std = g.std()
    geo_max = g.max()
    global_mean = train_src["demand"].mean()
    for df in dfs:
        df["geo_mean"] = df["geohash"].map(geo_mean).fillna(global_mean)
        df["geo_median"] = df["geohash"].map(geo_med).fillna(global_mean)
        df["geo_std"] = df["geohash"].map(geo_std).fillna(0.0)
        df["geo_max"] = df["geohash"].map(geo_max).fillna(global_mean)


def add_geo_hour_smoothed(train_src: pd.DataFrame, dfs: list[pd.DataFrame], smoothing: float = 2.0) -> None:
    """Per-(geohash, hour) Bayesian-smoothed mean. Hour has 24 levels (4 slots
    each), so each cell has ~4 observations on day 48 — more stable than
    per-slot encoding with 1 obs per cell.
    """
    src48 = train_src[train_src["day"] == 48]
    prior = src48.groupby("geohash")["demand"].mean()
    glb = src48["demand"].mean()
    g = src48.groupby(["geohash", "hour"])["demand"].agg(["mean", "count"]).reset_index()
    g = g.rename(columns={"mean": "raw"})
    g["prior"] = g["geohash"].map(prior).fillna(glb)
    g["smoothed"] = (g["count"] * g["raw"] + smoothing * g["prior"]) / (g["count"] + smoothing)
    sm = g.set_index(["geohash", "hour"])["smoothed"]
    for df in dfs:
        key = pd.MultiIndex.from_arrays([df["geohash"], df["hour"]])
        vals = sm.reindex(key).values.astype(float)
        ghp = df["geohash"].map(prior).fillna(glb).to_numpy()
        nan_mask = np.isnan(vals)
        vals[nan_mask] = ghp[nan_mask]
        # mask day-48 rows
        if "day" in df.columns:
            vals = np.where(df["day"].to_numpy() == 48, np.nan, vals)
        df["geo_hour_smoothed"] = vals


def add_geo_bucket_smoothed(train_src: pd.DataFrame, dfs: list[pd.DataFrame], bucket_col: str, smoothing: float = 3.0) -> None:
    """Generic per-(geohash, <bucket>) Bayesian-smoothed mean. Masked NaN on
    day-48 rows (otherwise leakage: each cell has few obs).
    """
    src48 = train_src[train_src["day"] == 48]
    prior = src48.groupby("geohash")["demand"].mean()
    glb = src48["demand"].mean()
    g = src48.groupby(["geohash", bucket_col])["demand"].agg(["mean", "count"]).reset_index()
    g = g.rename(columns={"mean": "raw"})
    g["prior"] = g["geohash"].map(prior).fillna(glb)
    g["smoothed"] = (g["count"] * g["raw"] + smoothing * g["prior"]) / (g["count"] + smoothing)
    sm = g.set_index(["geohash", bucket_col])["smoothed"]
    col = f"geo_{bucket_col}_smoothed"
    for df in dfs:
        key = pd.MultiIndex.from_arrays([df["geohash"], df[bucket_col]])
        vals = sm.reindex(key).values.astype(float)
        ghp = df["geohash"].map(prior).fillna(glb).to_numpy()
        nan_mask = np.isnan(vals)
        vals[nan_mask] = ghp[nan_mask]
        if "day" in df.columns:
            vals = np.where(df["day"].to_numpy() == 48, np.nan, vals)
        df[col] = vals


def add_prefix_slot_smoothed(train_src: pd.DataFrame, dfs: list[pd.DataFrame], prefix_len: int = 5, smoothing: float = 5.0) -> None:
    """Per-(geohash-prefix, slot) Bayesian-smoothed mean. Prefix-5 = ~5km area,
    so each (prefix, slot) has many more observations than the full 6-char.
    """
    src48 = train_src[train_src["day"] == 48].copy()
    src48[f"gh{prefix_len}"] = src48["geohash"].str[:prefix_len]
    # Prior: per-prefix mean across all slots
    prior = src48.groupby(f"gh{prefix_len}")["demand"].mean()
    glb = src48["demand"].mean()
    g = src48.groupby([f"gh{prefix_len}", "slot"])["demand"].agg(["mean", "count"]).reset_index()
    g = g.rename(columns={"mean": "raw"})
    g["prior"] = g[f"gh{prefix_len}"].map(prior).fillna(glb)
    g["smoothed"] = (g["count"] * g["raw"] + smoothing * g["prior"]) / (g["count"] + smoothing)
    sm = g.set_index([f"gh{prefix_len}", "slot"])["smoothed"]
    col = f"gh{prefix_len}_slot_smoothed"
    for df in dfs:
        df[f"gh{prefix_len}"] = df["geohash"].str[:prefix_len]
        key = pd.MultiIndex.from_arrays([df[f"gh{prefix_len}"], df["slot"]])
        vals = sm.reindex(key).values.astype(float)
        gh_prior = df[f"gh{prefix_len}"].map(prior).fillna(glb).to_numpy()
        nan_mask = np.isnan(vals)
        vals[nan_mask] = gh_prior[nan_mask]
        # Don't mask for day-48 rows here: the prefix aggregates many rows so own-row leakage is tiny
        df[col] = vals


def add_bayes_smoothed_geo_slot(train_src: pd.DataFrame, dfs: list[pd.DataFrame], smoothing: float = 1.0) -> None:
    """Bayesian-smoothed per-(geohash, slot) target encoding using day-48 data.

    smoothed = (count * group_mean + smoothing * prior) / (count + smoothing)
    where prior = per-geohash day-48 mean. With smoothing=3 and count=1, the
    encoded value blends 25% exact lag + 75% geohash mean — regularizing the
    noisy per-cell signal toward a stable prior.
    """
    src48 = train_src[train_src["day"] == 48]
    # Per-geohash prior (the smoothing target)
    prior = src48.groupby("geohash")["demand"].mean()
    glb = src48["demand"].mean()

    g = src48.groupby(["geohash", "slot"])["demand"].agg(["mean", "count"]).reset_index()
    g = g.rename(columns={"mean": "raw"})
    # Map prior onto g
    g["prior"] = g["geohash"].map(prior).fillna(glb)
    g["smoothed"] = (g["count"] * g["raw"] + smoothing * g["prior"]) / (g["count"] + smoothing)

    smoothed_map = g.set_index(["geohash", "slot"])["smoothed"]
    for df in dfs:
        key = pd.MultiIndex.from_arrays([df["geohash"], df["slot"]])
        vals = smoothed_map.reindex(key).values.astype(float)
        # Fallback: per-geohash prior when (G, slot) missing
        gh_prior = df["geohash"].map(prior).fillna(glb).to_numpy()
        nan_mask = np.isnan(vals)
        vals[nan_mask] = gh_prior[nan_mask]
        # Mask out for day-48 training rows (they'd see their own row's contribution)
        if "day" in df.columns:
            vals = np.where(df["day"].to_numpy() == 48, np.nan, vals)
        df["geo_slot_smoothed"] = vals


def add_geohash_slot_aggregates(train_src: pd.DataFrame, dfs: list[pd.DataFrame]) -> None:
    """Day-48 demand per (geohash, slot). For day-48 rows this would leak their
    own target, so we mask it to NaN there; the model still gets clean signal
    from day-49 train rows and test rows.
    """
    src48 = train_src[train_src["day"] == 48]
    g = src48.groupby(["geohash", "slot"])["demand"].mean()
    g = g.rename("geo_slot_mean").reset_index()
    for df in dfs:
        merged = df.merge(g, on=["geohash", "slot"], how="left")
        vals = merged["geo_slot_mean"].values.astype(float)
        if "day" in df.columns:
            vals = np.where(df["day"].to_numpy() == 48, np.nan, vals)
        df["geo_slot_mean"] = vals


def add_lag_features(train_src: pd.DataFrame, dfs: list[pd.DataFrame]) -> None:
    """Day-48 same-slot demand (and neighbour-slot demand) per geohash."""
    src48 = train_src[train_src["day"] == 48]

    same_slot = src48.set_index(["geohash", "slot"])["demand"].rename("lag_same_slot")
    same_slot = same_slot[~same_slot.index.duplicated(keep="last")]
    for df in dfs:
        vals = same_slot.reindex(pd.MultiIndex.from_arrays([df["geohash"], df["slot"]])).values.astype(float)
        if "day" in df.columns:
            vals = np.where(df["day"].to_numpy() == 48, np.nan, vals)
        df["lag_same_slot"] = pd.Series(vals, index=df.index)

    # day-48 demand at slot±1 and slot±2 (vectorised via wide pivot)
    pivot48 = src48.pivot_table(index="geohash", columns="slot", values="demand", aggfunc="last")
    for df in dfs:
        day_arr = df["day"].to_numpy() if "day" in df.columns else None
        for off in (-2, -1, 1, 2):
            target_slot = df["slot"].to_numpy() + off
            vals = np.full(len(df), np.nan)
            valid = (target_slot >= 0) & (target_slot <= 95)
            if valid.any():
                idx = df["geohash"].to_numpy()
                sub_idx = np.where(valid)[0]
                for i in sub_idx:
                    gi = idx[i]
                    ts = target_slot[i]
                    if gi in pivot48.index and ts in pivot48.columns:
                        v = pivot48.at[gi, ts]
                        if pd.notna(v):
                            vals[i] = v
            if day_arr is not None:
                vals = np.where(day_arr == 48, np.nan, vals)
            df[f"lag_d48_off{off:+d}"] = vals
        # mean over the 5-slot window (already NaN-masked for day-48 rows)
        cols = [f"lag_d48_off{o:+d}" for o in (-2, -1, 1, 2)]
        df["lag_d48_neighbor_mean"] = df[cols + ["lag_same_slot"]].mean(axis=1)


def add_day48_stats(train_src: pd.DataFrame, dfs: list[pd.DataFrame]) -> None:
    """Per-geohash day-48 aggregate stats (independent of slot)."""
    src48 = train_src[train_src["day"] == 48]
    g = src48.groupby("geohash")["demand"]
    stats = pd.DataFrame({
        "d48_mean": g.mean(),
        "d48_median": g.median(),
        "d48_max": g.max(),
        "d48_p90": g.quantile(0.9),
        "d48_std": g.std(),
    })
    glb = src48["demand"].mean()
    for df in dfs:
        m = df["geohash"].map(stats["d48_mean"]).fillna(glb)
        df["d48_mean"] = m
        df["d48_median"] = df["geohash"].map(stats["d48_median"]).fillna(glb)
        df["d48_max"] = df["geohash"].map(stats["d48_max"]).fillna(glb)
        df["d48_p90"] = df["geohash"].map(stats["d48_p90"]).fillna(glb)
        df["d48_std"] = df["geohash"].map(stats["d48_std"]).fillna(0.0)


def add_slot_global(train_src: pd.DataFrame, dfs: list[pd.DataFrame]) -> None:
    """Cross-geohash mean demand per slot — captures the daily rush-hour curve.

    Also adds per-(RoadType, slot) mean: highway-vs-residential rush patterns.
    """
    slot_mean = train_src.groupby("slot")["demand"].mean()
    glb = train_src["demand"].mean()
    for df in dfs:
        df["slot_global_mean"] = df["slot"].map(slot_mean).fillna(glb)


def add_d49_d48_calibration(train_src: pd.DataFrame, dfs: list[pd.DataFrame]) -> None:
    """For each geohash, compute the day-49 vs day-48 demand ratio and delta
    at the overlapping slots 0-8. This captures 'how today differs from
    yesterday at the same geohash' — exactly the correction the model needs.
    """
    src49 = train_src[train_src["day"] == 49]
    src48 = train_src[train_src["day"] == 48]
    if len(src49) == 0:
        for df in dfs:
            df["d49_d48_ratio"] = 1.0
            df["d49_d48_delta"] = 0.0
        return

    # Per-geohash sum of demand at slots present on both days
    pair = src49.merge(
        src48[["geohash", "slot", "demand"]].rename(columns={"demand": "d48"}),
        on=["geohash", "slot"], how="inner",
    )
    pair = pair.rename(columns={"demand": "d49"})
    agg = pair.groupby("geohash").agg(
        d49_sum=("d49", "sum"),
        d48_sum=("d48", "sum"),
        n=("d49", "count"),
    )
    eps = 1e-3
    ratio = (agg["d49_sum"] + eps) / (agg["d48_sum"] + eps)
    delta = (agg["d49_sum"] - agg["d48_sum"]) / agg["n"].clip(lower=1)

    # Global fallback ratio
    global_ratio = (pair["d49"].sum() + eps) / (pair["d48"].sum() + eps)
    global_delta = (pair["d49"] - pair["d48"]).mean()
    for df in dfs:
        df["d49_d48_ratio"] = df["geohash"].map(ratio).fillna(global_ratio)
        df["d49_d48_delta"] = df["geohash"].map(delta).fillna(global_delta)


def add_geohash_prefix_stats(train_src: pd.DataFrame, dfs: list[pd.DataFrame]) -> None:
    """Aggregate demand by 5-char geohash prefix (broader spatial neighbourhood)
    and 4-char prefix (wider). Day-48 only to avoid leakage."""
    src48 = train_src[train_src["day"] == 48].copy()
    src48["gh5"] = src48["geohash"].str[:5]
    src48["gh4"] = src48["geohash"].str[:4]

    g5_mean = src48.groupby("gh5")["demand"].mean()
    g5_slot = src48.groupby(["gh5", "slot"])["demand"].mean()
    g4_mean = src48.groupby("gh4")["demand"].mean()
    global_mean = src48["demand"].mean()

    for df in dfs:
        df["gh5"] = df["geohash"].str[:5]
        df["gh4"] = df["geohash"].str[:4]
        df["nbr5_mean"] = df["gh5"].map(g5_mean).fillna(global_mean)
        df["nbr4_mean"] = df["gh4"].map(g4_mean).fillna(global_mean)
        # gh5 x slot mean: merge
        key = list(zip(df["gh5"], df["slot"]))
        df["nbr5_slot_mean"] = pd.Series(
            g5_slot.reindex(pd.MultiIndex.from_tuples(key)).values, index=df.index
        ).fillna(df["nbr5_mean"])


def add_day49_recent(train_src: pd.DataFrame, dfs: list[pd.DataFrame]) -> None:
    """Most recent day-49 demand for the same geohash, strictly before the
    current row's slot. For day-48 rows, also use latest day-49 (slot=any).

    Day-49 train rows: lookup must use slot < current_slot to avoid leakage.
    """
    src49 = train_src[train_src["day"] == 49]
    if len(src49) == 0:
        for df in dfs:
            df["d49_last_demand"] = np.nan
            df["d49_mean"] = np.nan
            df["d49_slot_gap"] = np.nan
        return

    # Per-geohash sorted demand history on day 49
    g_groups = {}
    for gh, sub in src49.sort_values("slot").groupby("geohash"):
        g_groups[gh] = (sub["slot"].to_numpy(), sub["demand"].to_numpy())

    for df in dfs:
        last_demand = np.full(len(df), np.nan)
        last_slot = np.full(len(df), np.nan)
        mean_before = np.full(len(df), np.nan)
        ghs = df["geohash"].to_numpy()
        slots = df["slot"].to_numpy()
        days = df["day"].to_numpy() if "day" in df.columns else np.full(len(df), 49)
        for i in range(len(df)):
            gh = ghs[i]
            if gh not in g_groups:
                continue
            slot_arr, demand_arr = g_groups[gh]
            if days[i] == 48:
                # use everything available
                if len(demand_arr) > 0:
                    last_demand[i] = demand_arr[-1]
                    last_slot[i] = slot_arr[-1]
                    mean_before[i] = demand_arr.mean()
            else:
                cur_slot = slots[i]
                mask = slot_arr < cur_slot
                if mask.any():
                    last_demand[i] = demand_arr[mask][-1]
                    last_slot[i] = slot_arr[mask][-1]
                    mean_before[i] = demand_arr[mask].mean()
        df["d49_last_demand"] = last_demand
        df["d49_mean"] = mean_before
        df["d49_slot_gap"] = slots - last_slot


def build_features(train_raw: pd.DataFrame, test_raw: pd.DataFrame, train_src_subset: pd.DataFrame | None = None, drop_features: list[str] | None = None, oof: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Apply all feature engineering steps. Returns (train_df, test_df, feature_cols).

    `train_src_subset` is the source used for target-encoding / lag features; if
    None, the full train is used (production case). For validation, pass only
    rows that pre-date the held-out window.
    """
    train = train_raw.copy()
    test = test_raw.copy()

    for df in (train, test):
        parse_time(df)
        add_latlng(df)
        encode_categoricals(df)

    train, test = impute_static_by_geohash(train, test)

    if train_src_subset is not None:
        src = train_src_subset.copy()
        parse_time(src)
        encode_categoricals(src)
    else:
        src = train
    if oof:
        add_geohash_aggregates_oof(src, train, test)
        add_day48_stats_oof(src, train, test)
    else:
        add_geohash_aggregates(src, [train, test])
        add_day48_stats(src, [train, test])
    add_geohash_slot_aggregates(src, [train, test])
    add_bayes_smoothed_geo_slot(src, [train, test])
    add_prefix_slot_smoothed(src, [train, test], prefix_len=5, smoothing=10.0)
    add_prefix_slot_smoothed(src, [train, test], prefix_len=4, smoothing=5.0)
    add_lag_features(src, [train, test])
    add_slot_global(src, [train, test])
    add_d49_d48_calibration(src, [train, test])
    add_day49_recent(src, [train, test])

    # Derived: calibrated lag = d48 lag scaled by per-geohash d49/d48 ratio
    for df in (train, test):
        lag = df["lag_same_slot"].fillna(df["d48_mean"])
        df["lag_calibrated"] = lag * df["d49_d48_ratio"]
        df["lag_calibrated_delta"] = lag + df["d49_d48_delta"]

    feature_cols = [
        "slot",
        "hour",
        "slot_sin",
        "slot_cos",
        "day",
        "lat",
        "lng",
        "NumberofLanes",
        "RoadType_enc",
        "Weather_enc",
        "LargeVehicles_enc",
        "Landmarks_enc",
        "Temperature",
        "geo_mean",
        "geo_median",
        "geo_std",
        "geo_max",
        "geo_slot_mean",
        "geo_slot_smoothed",
        "gh5_slot_smoothed",
        "gh4_slot_smoothed",
        "lag_same_slot",
        "lag_d48_off-2",
        "lag_d48_off-1",
        "lag_d48_off+1",
        "lag_d48_off+2",
        "lag_d48_neighbor_mean",
        "d48_mean",
        "d48_median",
        "d48_max",
        "d48_p90",
        "d48_std",
        "slot_global_mean",
        "d49_last_demand",
        "d49_mean",
        "d49_slot_gap",
        "d49_d48_ratio",
        "d49_d48_delta",
        "lag_calibrated",
        "lag_calibrated_delta",
    ]
    if drop_features:
        feature_cols = [f for f in feature_cols if f not in drop_features]
    return train, test, feature_cols


# ---------- model ----------
def train_catboost(X_tr, y_tr, X_val=None, y_val=None, params=None, sample_weight=None):
    from catboost import CatBoostRegressor

    default_params = dict(
        iterations=4000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=42,
        bootstrap_type="Bernoulli",
        subsample=0.85,
        rsm=0.85,
        loss_function="RMSE",
        eval_metric="RMSE",
        early_stopping_rounds=100,
        verbose=400,
        allow_writing_files=False,
    )
    if params:
        default_params.update(params)
    model = CatBoostRegressor(**default_params)
    eval_set = (X_val, y_val) if X_val is not None else None
    model.fit(X_tr, y_tr, eval_set=eval_set, sample_weight=sample_weight, use_best_model=eval_set is not None)
    return model


def train_xgb(X_tr, y_tr, X_val=None, y_val=None, params=None):
    import xgboost as xgb

    default_params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.0,
        "verbosity": 0,
        "seed": 42,
    }
    if params:
        default_params.update(params)

    # Use DataFrame directly so NaN handling is preserved
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    evals = [(dtr, "train")]
    if X_val is not None:
        dval = xgb.DMatrix(X_val, label=y_val)
        evals.append((dval, "val"))
    model = xgb.train(
        default_params,
        dtr,
        num_boost_round=4000,
        evals=evals,
        early_stopping_rounds=100 if X_val is not None else None,
        verbose_eval=400,
    )
    return model


def train_lgbm(X_tr, y_tr, X_val=None, y_val=None, params=None, sample_weight=None):
    import lightgbm as lgb

    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.02,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "lambda_l2": 2.0,
        "verbosity": -1,
        "seed": 42,
    }
    if params:
        default_params.update(params)

    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=sample_weight)
    valid_sets = [dtrain]
    valid_names = ["train"]
    if X_val is not None:
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        valid_sets.append(dval)
        valid_names.append("val")

    callbacks = [lgb.log_evaluation(period=200)]
    if X_val is not None:
        callbacks.append(lgb.early_stopping(stopping_rounds=100, verbose=False))

    model = lgb.train(
        default_params,
        dtrain,
        num_boost_round=4000,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return model


# ---------- main ----------
def main(args):
    print("Loading data...")
    tr_raw = pd.read_csv(DATA / "train.csv")
    te_raw = pd.read_csv(DATA / "test.csv")

    if args.validate:
        full = tr_raw.copy()
        parse = full["timestamp"].str.split(":", expand=True)
        full_slot = parse[0].astype(int) * 4 + parse[1].astype(int) // 15

        if args.slot_val:
            # Hold out 5% of day-48 slots 9-55 rows. Same slot distribution as
            # test, so this exposes slot-of-day prediction quality. Caveat:
            # val rows are day-48 (no d49 uplift), so this is harder than test.
            rng = np.random.default_rng(7)
            cand = full.index[(full["day"] == 48) & full_slot.between(9, 55)].to_numpy()
            cand = np.array(cand, copy=True)
            rng.shuffle(cand)
            keep = cand[: int(len(cand) * 0.05)]
            val_mask = pd.Series(False, index=full.index)
            val_mask.loc[keep] = True
        elif args.hard_val:
            # Hold out day-49 slots 5-8: forces val rows to be the "latest" d49
            # slots, so d49_last_demand has slot-gap 1-4 (closer to test).
            val_mask = (full["day"] == 49) & (full_slot >= 5) & (full_slot <= 8)
        else:
            rng = np.random.default_rng(42)
            d49_idx = np.array(full.index[full["day"] == 49].to_numpy(), copy=True)
            rng.shuffle(d49_idx)
            n_val = int(len(d49_idx) * 0.2)
            val_mask = pd.Series(False, index=full.index)
            val_mask.loc[d49_idx[:n_val]] = True

        tr_only = full.loc[~val_mask].reset_index(drop=True)
        val = full.loc[val_mask].reset_index(drop=True)
        print(f"Train rows: {len(tr_only)}  Validation rows: {len(val)}  (hard_val={args.hard_val})")

        drop = args.drop.split(",") if args.drop else None
        tr_df, val_df, feats = build_features(tr_only, val, train_src_subset=tr_only, drop_features=drop, oof=args.oof)
        if drop:
            print(f"Dropped features: {drop}")
            print(f"Remaining features: {len(feats)}")

        if args.mask_d49_val:
            # Simulate test-like staleness: pretend val rows had no recent d49.
            # Test rows have d49_last_demand from slot<=8 and current slot 9-55,
            # giving slot_gap up to 47. Force val rows into that regime.
            val_df["d49_last_demand"] = np.nan
            val_df["d49_slot_gap"] = 30.0  # representative test slot-gap
            print("Masked d49 recency features on val rows (simulating test staleness)")

        X_tr = tr_df[feats]
        y_tr = tr_df["demand"]
        X_val = val_df[feats]
        y_val = val_df["demand"]

        # Optional: restrict training to day-49 only to match test distribution
        if args.day49_only:
            mask49 = tr_df["day"] == 49
            X_tr = X_tr.loc[mask49]
            y_tr = y_tr.loc[mask49]
            print(f"Restricted to day-49 train rows only: {len(X_tr)}")

        if args.log_target:
            y_tr_used = np.log1p(y_tr)
        else:
            y_tr_used = y_tr

        if args.d49_weight != 1.0:
            day_fit = tr_df["day"].to_numpy() if not args.day49_only else np.full(len(X_tr), 49)
            weights = np.where(day_fit == 49, args.d49_weight, 1.0)
        else:
            weights = None

        model = train_lgbm(
            X_tr, y_tr_used, X_val, np.log1p(y_val) if args.log_target else y_val,
            sample_weight=weights,
        )
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        if args.log_target:
            preds = np.expm1(preds)
        preds = np.clip(preds, 0, 1)
        r2 = r2_score(y_val, preds)
        print(f"\n>>> Validation R^2 = {r2:.5f}   (score = {max(0, 100*r2):.3f})")

        # Sanity baselines
        lag_pred = val_df["lag_same_slot"].fillna(val_df["d48_mean"]).clip(0, 1)
        print(f"    Naive day-48 lag R^2 = {r2_score(y_val, lag_pred):.5f}")
        d48mean_pred = val_df["d48_mean"].clip(0, 1)
        print(f"    Naive geohash-mean R^2 = {r2_score(y_val, d48mean_pred):.5f}")

        imp = pd.DataFrame({"feature": feats, "gain": model.feature_importance("gain")}).sort_values("gain", ascending=False)
        print("\nTop features by gain:")
        print(imp.head(20).to_string(index=False))
        return

    # Full submission run: multi-seed LightGBM ensemble with day-49 hold-out
    # for per-seed early stopping. Predictions averaged on the original scale.
    tr_df, te_df, feats = build_features(tr_raw, te_raw, oof=args.oof)

    # Validation slice: 20% of day-49 train rows (same distribution as test)
    rng_split = np.random.default_rng(42)
    d49_pos = np.where(tr_df["day"].to_numpy() == 49)[0]
    rng_split.shuffle(d49_pos)
    n_val = int(len(d49_pos) * 0.2)
    val_pos = d49_pos[:n_val]
    fit_pos = np.setdiff1d(np.arange(len(tr_df)), val_pos)

    X = tr_df[feats]
    y = tr_df["demand"]
    X_te = te_df[feats]
    X_val = X.iloc[val_pos]
    y_val = y.iloc[val_pos]

    use_log = not args.no_log_target
    if use_log:
        print("Using log1p target transform")
        y_fit_full = np.log1p(y)
        y_val_used = np.log1p(y_val)
    else:
        y_fit_full = y
        y_val_used = y_val

    n_seeds = args.seeds
    test_preds = np.zeros(len(te_df))
    val_preds = np.zeros(len(val_pos))
    seed_r2s = []
    seed_imps = []

    n_total_members = n_seeds + (1 if args.xgb else 0) + args.cat_seeds
    weight_lgb = 1.0 / n_total_members

    # Weight day-49 rows higher so they steer the model toward test distribution
    day_arr = tr_df["day"].to_numpy()
    weights_full = np.where(day_arr == 49, args.d49_weight, 1.0)

    for s in range(n_seeds):
        seed = 42 + s * 7
        print(f"\n=== LGBM Seed {s+1}/{n_seeds} (seed={seed}) ===")
        params = {"seed": seed, "feature_fraction_seed": seed, "bagging_seed": seed}
        model = train_lgbm(
            X.iloc[fit_pos], y_fit_full.iloc[fit_pos], X_val, y_val_used,
            params=params, sample_weight=weights_full[fit_pos],
        )

        vp = model.predict(X_val, num_iteration=model.best_iteration)
        tp = model.predict(X_te, num_iteration=model.best_iteration)
        if use_log:
            vp = np.expm1(vp)
            tp = np.expm1(tp)
        vp = np.clip(vp, 0, 1)
        tp = np.clip(tp, 0, 1)
        val_preds += vp * weight_lgb
        test_preds += tp * weight_lgb

        r2 = r2_score(y_val, vp)
        seed_r2s.append(r2)
        print(f"   seed {seed} val R^2 = {r2:.5f}")
        seed_imps.append(model.feature_importance("gain"))

    if args.xgb:
        print(f"\n=== XGBoost member ===")
        xgb_model = train_xgb(X.iloc[fit_pos], y_fit_full.iloc[fit_pos], X_val, y_val_used)
        import xgboost as xgb
        dval = xgb.DMatrix(X_val)
        dte = xgb.DMatrix(X_te)
        vp = xgb_model.predict(dval, iteration_range=(0, xgb_model.best_iteration + 1))
        tp = xgb_model.predict(dte, iteration_range=(0, xgb_model.best_iteration + 1))
        if use_log:
            vp = np.expm1(vp)
            tp = np.expm1(tp)
        vp = np.clip(vp, 0, 1)
        tp = np.clip(tp, 0, 1)
        val_preds += vp * weight_lgb
        test_preds += tp * weight_lgb
        r2 = r2_score(y_val, vp)
        print(f"   xgb val R^2 = {r2:.5f}")
        seed_r2s.append(r2)

    for s in range(args.cat_seeds):
        seed = 100 + s * 11
        print(f"\n=== CatBoost seed {s+1}/{args.cat_seeds} (seed={seed}) ===")
        cat_model = train_catboost(
            X.iloc[fit_pos], y_fit_full.iloc[fit_pos], X_val, y_val_used,
            params={"random_seed": seed},
            sample_weight=weights_full[fit_pos],
        )
        vp = cat_model.predict(X_val)
        tp = cat_model.predict(X_te)
        if use_log:
            vp = np.expm1(vp)
            tp = np.expm1(tp)
        vp = np.clip(vp, 0, 1)
        tp = np.clip(tp, 0, 1)
        val_preds += vp * weight_lgb
        test_preds += tp * weight_lgb
        r2 = r2_score(y_val, vp)
        seed_r2s.append(r2)
        print(f"   cat seed {seed} val R^2 = {r2:.5f}")

    ens_r2 = r2_score(y_val, val_preds)
    print(f"\n>>> Ensemble val R^2 = {ens_r2:.5f}   (mean per-seed = {np.mean(seed_r2s):.5f})")

    sub = pd.DataFrame({"Index": te_raw["Index"], "demand": test_preds})
    out_path = OUT / args.out
    sub.to_csv(out_path, index=False)
    print(f"\nSaved submission: {out_path}  (rows={len(sub)})")
    print(sub.head())
    print("\nPrediction summary:")
    print(sub["demand"].describe())

    avg_imp = np.mean(seed_imps, axis=0)
    imp = pd.DataFrame({"feature": feats, "gain": avg_imp}).sort_values("gain", ascending=False)
    print("\nTop features by gain (avg over seeds):")
    print(imp.head(20).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="Run hold-out validation instead of full fit")
    ap.add_argument("--day49_only", action="store_true", help="Train on day-49 rows only (matches test distribution)")
    ap.add_argument("--hard_val", action="store_true", help="Use day-49 slots 5-8 as validation (closer to test slot-gap)")
    ap.add_argument("--log_target", action="store_true", help="Train on log1p(demand)")
    ap.add_argument("--no_log_target", action="store_true", help="Disable log target in submission mode")
    ap.add_argument("--seeds", type=int, default=5, help="Number of seeds for ensemble in submission mode")
    ap.add_argument("--xgb", action="store_true", help="Add an XGBoost member to the ensemble")
    ap.add_argument("--cat_seeds", type=int, default=0, help="Number of CatBoost members")
    ap.add_argument("--drop", default="", help="Comma-separated features to drop")
    ap.add_argument("--oof", action="store_true", help="Use K-fold OOF target encoding for geohash aggregates")
    ap.add_argument("--d49_weight", type=float, default=1.0, help="Sample weight multiplier for day-49 rows")
    ap.add_argument("--mask_d49_val", action="store_true", help="Diagnostic: mask d49 recency features on val rows to simulate test staleness")
    ap.add_argument("--slot_val", action="store_true", help="Val = random 5% of day-48 slots 9-55 (matches test slot distribution)")
    ap.add_argument("--out", default="submission_v2.csv")
    main(ap.parse_args())
