"""Lightweight sequence model for geohash demand curves.

This is intentionally separate from the v12 feature pipeline.  It treats each
geohash as one sequence and trains a small Transformer to fill masked future
slots from observed context.  At submission time the model sees:

- day-48 slots 0..95 as historical context
- day-49 slots 0..8 as observed adaptation context
- day-49 slots 9..55 masked for prediction

The output is a standalone submission plus optional blends with v12.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pipeline import DATA, OUT, encode_categoricals, impute_static_by_geohash, parse_time


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 152  # day48 0..95 + day49 0..55
PRED_START = 96 + 9
PRED_END = 96 + 55


class CurveDataset(Dataset):
    def __init__(self, curves: np.ndarray, statics: np.ndarray, repeats: int = 3, seed: int = 1):
        self.curves = curves.astype(np.float32)
        self.statics = statics.astype(np.float32)
        self.repeats = repeats
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.curves) * self.repeats

    def __getitem__(self, idx: int):
        i = idx % len(self.curves)
        y = self.curves[i].copy()

        # Self-supervised masks inside day 48.  The task mirrors submission:
        # enough early context is visible, then a contiguous future block is hidden.
        start = int(self.rng.integers(9, 42))
        width = int(self.rng.integers(12, 48))
        end = min(95, start + width)

        x = y.copy()
        obs = np.ones_like(x, dtype=np.float32)
        x[start : end + 1] = 0.0
        obs[start : end + 1] = 0.0

        target_mask = np.zeros_like(x, dtype=np.float32)
        target_mask[start : end + 1] = 1.0
        return x, obs, self.statics[i], y, target_mask


class SeqDemandModel(nn.Module):
    def __init__(self, static_dim: int, d_model: int = 48, nhead: int = 4, layers: int = 1):
        super().__init__()
        self.value_proj = nn.Linear(6, d_model)
        self.static_proj = nn.Linear(static_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.10,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1), nn.Sigmoid())

        pos = np.arange(SEQ_LEN)
        slot = pos % 96
        day49 = (pos >= 96).astype(np.float32)
        feats = np.stack(
            [
                np.sin(2 * np.pi * slot / 96),
                np.cos(2 * np.pi * slot / 96),
                day49,
                slot / 95.0,
            ],
            axis=1,
        ).astype(np.float32)
        self.register_buffer("time_feats", torch.tensor(feats))

    def forward(self, values: torch.Tensor, observed: torch.Tensor, statics: torch.Tensor) -> torch.Tensor:
        b = values.shape[0]
        time_feats = self.time_feats.unsqueeze(0).expand(b, -1, -1)
        token = torch.cat([values.unsqueeze(-1), observed.unsqueeze(-1), time_feats], dim=-1)
        h = self.value_proj(token) + self.static_proj(statics).unsqueeze(1)
        h = self.encoder(h)
        return self.head(h).squeeze(-1)


def prep(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    train = train_raw.copy()
    test = test_raw.copy()
    for df in (train, test):
        parse_time(df)
        encode_categoricals(df)
    train, test = impute_static_by_geohash(train, test)

    geos = pd.Index(pd.concat([train["geohash"], test["geohash"]]).unique()).sort_values()
    geo_to_i = {g: i for i, g in enumerate(geos)}
    curves = np.full((len(geos), SEQ_LEN), np.nan, dtype=np.float32)

    for _, row in train.iterrows():
        col = int(row["slot"]) if row["day"] == 48 else 96 + int(row["slot"])
        if col < SEQ_LEN:
            curves[geo_to_i[row["geohash"]], col] = row["demand"]

    d48_counts = np.isfinite(curves[:, :96]).sum(axis=1)
    d48_sums = np.nan_to_num(curves[:, :96], nan=0.0).sum(axis=1)
    global_mean = np.nanmean(curves[:, :96])
    d48_mean = np.divide(d48_sums, d48_counts, out=np.full(len(curves), global_mean), where=d48_counts > 0)
    for c in range(SEQ_LEN):
        fallback = d48_mean if c < 96 else curves[:, c - 96]
        fallback = np.where(np.isnan(fallback), d48_mean, fallback)
        curves[:, c] = np.where(np.isnan(curves[:, c]), fallback, curves[:, c])

    static_cols = [
        "RoadType_enc",
        "NumberofLanes",
        "LargeVehicles_enc",
        "Landmarks_enc",
        "Temperature",
        "Weather_enc",
    ]
    full = pd.concat([train, test], ignore_index=True)
    stat = full.groupby("geohash")[static_cols].mean().reindex(geos)
    stat = stat.fillna(stat.mean())
    stat = (stat - stat.mean()) / stat.std().replace(0, 1)
    return train, test, geos, geo_to_i, curves, stat.to_numpy(dtype=np.float32)


def train_model(curves: np.ndarray, statics: np.ndarray, epochs: int, batch_size: int, seed: int, device: str):
    torch.manual_seed(seed)
    ds = CurveDataset(curves[:, :SEQ_LEN], statics, repeats=3, seed=seed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    model = SeqDemandModel(static_dim=statics.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    model.train()
    for ep in range(1, epochs + 1):
        losses = []
        for x, obs, st, y, mask in dl:
            x = x.to(device)
            obs = obs.to(device)
            st = st.to(device)
            y = y.to(device)
            mask = mask.to(device)
            pred = model(x, obs, st)
            loss = (((pred - y) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"epoch {ep:02d} mse={np.mean(losses):.6f}")
    return model


@torch.no_grad()
def predict(model: nn.Module, curves: np.ndarray, statics: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    x = curves.copy()
    obs = np.ones_like(x, dtype=np.float32)
    x[:, PRED_START : PRED_END + 1] = 0.0
    obs[:, PRED_START : PRED_END + 1] = 0.0

    preds = []
    bs = 256
    for i in range(0, len(x), bs):
        p = model(
            torch.tensor(x[i : i + bs], device=device),
            torch.tensor(obs[i : i + bs], device=device),
            torch.tensor(statics[i : i + bs], device=device),
        )
        preds.append(p.detach().cpu().numpy())
    return np.vstack(preds)


def validate(args, train_raw: pd.DataFrame) -> None:
    full = train_raw.copy()
    parse_time(full)
    val_mask = (full["day"] == 49) & full["slot"].between(5, 8)
    tr_only = train_raw.loc[~val_mask].reset_index(drop=True)
    val = train_raw.loc[val_mask].drop(columns=["demand"]).reset_index(drop=True)
    y_val = train_raw.loc[val_mask, "demand"].reset_index(drop=True)
    _, val_df, _, geo_to_i, curves, statics = prep(tr_only, val)
    device = resolve_device(args.device)
    model = train_model(curves, statics, args.epochs, args.batch_size, args.seed, device)
    pred_mat = predict(model, curves, statics, device)
    rows = val_df["geohash"].map(geo_to_i).to_numpy()
    cols = 96 + val_df["slot"].to_numpy(dtype=int)
    preds = pred_mat[rows, cols].clip(0, 1)
    print(f"Seq2Seq hard-val R^2 = {r2_score(y_val, preds):.5f}")
    print(pd.Series(preds).describe())


def write_submission(test_raw: pd.DataFrame, preds: np.ndarray, out_name: str) -> Path:
    out_path = OUT / out_name
    sub = pd.DataFrame({"Index": test_raw["Index"], "demand": preds.clip(0, 1)})
    sub.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(sub["demand"].describe())
    return out_path


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch install cannot see CUDA. "
            "Install a CUDA-enabled torch wheel in .venv first."
        )
    return requested


def main(args):
    train_raw = pd.read_csv(DATA / "train.csv")
    test_raw = pd.read_csv(DATA / "test.csv")
    if args.validate:
        validate(args, train_raw)
        return

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    _, test, _, geo_to_i, curves, statics = prep(train_raw, test_raw)
    all_preds = np.zeros((len(curves), SEQ_LEN), dtype=np.float32)
    for s in range(args.seeds):
        seed = args.seed + s * 17
        print(f"\n=== Seq2Seq seed {s + 1}/{args.seeds} (seed={seed}) on {device} ===")
        model = train_model(curves, statics, args.epochs, args.batch_size, seed, device)
        all_preds += predict(model, curves, statics, device) / args.seeds

    rows = test["geohash"].map(geo_to_i).to_numpy()
    cols = 96 + test["slot"].to_numpy(dtype=int)
    preds = all_preds[rows, cols]
    write_submission(test_raw, preds, args.out)

    if args.blend_with:
        base_path = Path(args.blend_with)
        if not base_path.is_absolute():
            base_path = OUT / base_path
        base = pd.read_csv(base_path)
        for w in args.blend_weights:
            sub = base.copy()
            sub["demand"] = ((1.0 - w) * base["demand"] + w * preds).clip(0, 1)
            write_submission(test_raw.assign(Index=base["Index"]), sub["demand"].to_numpy(), f"{Path(args.out).stem}_blend_w{w:.2f}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--blend_with", default="submission_v12.csv")
    ap.add_argument("--blend_weights", type=float, nargs="*", default=[0.03, 0.05, 0.08, 0.10])
    ap.add_argument("--out", default="submission_seq2seq.csv")
    main(ap.parse_args())
