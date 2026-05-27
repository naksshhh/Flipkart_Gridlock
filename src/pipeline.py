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
    """Cross-geohash mean demand per slot — captures the daily rush-hour curve."""
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


def build_features(train_raw: pd.DataFrame, test_raw: pd.DataFrame, train_src_subset: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
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
    else:
        src = train
    add_geohash_aggregates(src, [train, test])
    add_geohash_slot_aggregates(src, [train, test])
    add_lag_features(src, [train, test])
    add_day48_stats(src, [train, test])
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
    return train, test, feature_cols


# ---------- model ----------
def train_lgbm(X_tr, y_tr, X_val=None, y_val=None, params=None):
    import lightgbm as lgb

    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 42,
    }
    if params:
        default_params.update(params)

    dtrain = lgb.Dataset(X_tr, label=y_tr)
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

        if args.hard_val:
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

        tr_df, val_df, feats = build_features(tr_only, val, train_src_subset=tr_only)
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

        model = train_lgbm(X_tr, y_tr_used, X_val, np.log1p(y_val) if args.log_target else y_val)
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

    # Full submission run with random 10% hold-out for early stopping
    tr_df, te_df, feats = build_features(tr_raw, te_raw)
    rng = np.random.default_rng(42)
    n = len(tr_df)
    perm = rng.permutation(n)
    cut = int(n * 0.1)
    val_idx = perm[:cut]
    fit_idx = perm[cut:]

    X = tr_df[feats]
    y = tr_df["demand"]
    X_te = te_df[feats]

    model = train_lgbm(X.iloc[fit_idx], y.iloc[fit_idx], X.iloc[val_idx], y.iloc[val_idx])
    preds_val = np.clip(model.predict(X.iloc[val_idx], num_iteration=model.best_iteration), 0, 1)
    print(f"\nRandom-hold-out R^2 = {r2_score(y.iloc[val_idx], preds_val):.5f}")

    preds = np.clip(model.predict(X_te, num_iteration=model.best_iteration), 0, 1)
    sub = pd.DataFrame({"Index": te_raw["Index"], "demand": preds})
    out_path = OUT / args.out
    sub.to_csv(out_path, index=False)
    print(f"\nSaved submission: {out_path}  (rows={len(sub)})")
    print(sub.head())
    print("\nPrediction summary:")
    print(sub["demand"].describe())

    imp = pd.DataFrame({"feature": feats, "gain": model.feature_importance("gain")}).sort_values("gain", ascending=False)
    print("\nTop features by gain:")
    print(imp.head(25).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="Run hold-out validation instead of full fit")
    ap.add_argument("--day49_only", action="store_true", help="Train on day-49 rows only (matches test distribution)")
    ap.add_argument("--hard_val", action="store_true", help="Use day-49 slots 5-8 as validation (closer to test slot-gap)")
    ap.add_argument("--log_target", action="store_true", help="Train on log1p(demand)")
    ap.add_argument("--out", default="submission_v2.csv")
    main(ap.parse_args())
