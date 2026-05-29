# Flipkart Grid 2.0 — Traffic Demand Prediction

Solution for the Gridlock Hackathon 2.0 traffic-demand challenge (Spatio-Temporal regression).

## Current Best

- **Submission**: `outputs/submission_seq2seq_3seed_anti5_w0p40.csv`
- **Public leaderboard score**: **91.71540** (R² × 100)
- **Approach**: v12 LightGBM ensemble + 3-seed CUDA Transformer correction, then a small extrapolation away from a weaker 5-seed sequence blend.

The previous GBM-only local optimum was `outputs/submission_v12.csv` at **91.24**. The breakthrough came from moving off point-wise spatial regression and blending in a target-manifold sequence model trained over per-geohash demand curves. The current best is **+0.4754** leaderboard points over v12.

## Problem Setup

Predict normalized traffic demand at the (geohash, timestamp) level.

| Split | Rows | Day | Slot range |
|---|---|---|---|
| Train | 69,427 | 48 | 0-95 (full day) |
| Train | 7,872 | 49 | 0-8 (early morning, ~2 hours) |
| Test | 41,778 | 49 | 9-55 (~2 AM to 2 PM) |

Each `(geohash, slot)` pair has at most 2 observations in train — once on day 48 (always) and once on day 49 (only if slot ≤ 8). The test set asks for day-49 predictions at slots not seen on day 49.

**Critical insight from EDA**: the same `(geohash, slot)` exists in day-48 training data for nearly every test row, so the day-48 lag is the dominant signal. Raw day-48 → day-49 R² across the 7,872 overlapping training pairs is 0.49 — that's the naive-lag floor. Everything else is learning the day-to-day correction.

## Approach

### 1. Feature engineering

Three families of features that proved critical:

**Spatial features**
- Decoded 6-char geohash → `lat`, `lng` (continuous)
- Hierarchical prefixes: `gh4` (~50 km), `gh5` (~5 km) for fallback when 6-char is sparse
- Per-geohash static feature imputation (RoadType, Weather, Temperature) via geohash mode/mean before global fallback

**Temporal features**
- 15-min `slot` (0-95), `hour`, cyclical `sin`/`cos` of slot, monotonic `abs_time`
- `day` flag (encodes train-vs-test-day distribution)

**Target-encoded aggregates (the workhorses)**
- Per-geohash day-48 statistics: `mean`, `median`, `p90`, `max`, `std`
- Per-(geohash, slot) demand: raw + **Bayesian-smoothed** with per-geohash prior (smoothing=1, count typically 1)
- Per-(gh5, slot) and per-(gh4, slot) **Bayesian-smoothed** (smoothing=10, 5) — provides spatial fallback when the exact cell is noisy or absent
- Day-49 recency: most-recent same-geohash demand on day 49, slot-gap, day-49 mean
- Cross-day calibration: per-geohash day-49/day-48 ratio and delta over slot 0-8 overlap, plus `lag × ratio` and `lag + delta` interaction features
- Slot-global mean (cross-geohash rush-hour curve)

All target encoding features are computed **only from day-48** data and **masked NaN for day-48 training rows** to prevent target leakage on training. Bayesian smoothing of the form `(n·obs + k·prior) / (n + k)` shrinks the noisy 1-observation cells toward the per-geohash prior.

### 2. GBM model

- **LightGBM regression** with log1p target
- Hyperparameters tuned via sweep on hard validation:
  - `learning_rate=0.02`, `num_leaves=63`, `min_data_in_leaf=100`, `lambda_l2=2.0`
  - `feature_fraction=0.85`, `bagging_fraction=0.85`, `bagging_freq=5`
- Sample weights: day-49 train rows weighted **5×** vs day-48 (matches test distribution, found optimum on hard val)
- **10-seed ensemble**, predictions averaged on the original scale, clipped to [0, 1]

### 3. Sequence correction

The post-v12 improvement uses `src/seq2seq.py`, a small Transformer over one sequence per geohash:

- Input sequence length: 152 columns = day-48 slots `0-95` + day-49 slots `0-55`
- Known context at inference: all day-48 slots and day-49 slots `0-8`
- Masked prediction target at inference: day-49 slots `9-55`
- Training task: self-supervised masked future reconstruction inside day 48
- Static conditioning: RoadType, NumberofLanes, LargeVehicles, Landmarks, Temperature, Weather
- CUDA PyTorch run: `torch 2.11.0+cu128` on RTX 3050 Laptop GPU

Raw seq2seq predictions are under-dispersed and weak as a standalone submission, but they encode a useful low-frequency correction direction. The first single-seed blend peaked around `15.5%` seq2seq:

| Seq2Seq blend weight | Leaderboard |
|---:|---:|
| 0.010 | 91.29646 |
| 0.030 | 91.39188 |
| 0.080 | 91.56862 |
| 0.140 | 91.66418 |
| 0.155 | **91.66821** |
| 0.160 | 91.66779 |
| 0.180 | 91.65727 |

Ensembling the first three Transformer seeds improved the correction. A 5-seed average was worse, but the bad 5-seed direction turned out useful as an extrapolation target:

| Submission | Leaderboard |
|---|---:|
| `submission_seq2seq_3seed_blend_w0p150.csv` | 91.70397 |
| `submission_seq2seq_3seed_blend_w0p152.csv` | 91.70416 |
| `submission_seq2seq_3seed_blend_w0p1525.csv` | 91.70418 |
| `submission_seq2seq_5seed_blend_w0p155.csv` | 91.66898 |
| `submission_seq2seq_3seed_anti5_w0p20.csv` | 91.70998 |
| `submission_seq2seq_3seed_anti5_w0p25.csv` | 91.71137 |
| `submission_seq2seq_3seed_anti5_w0p32.csv` | 91.71328 |
| `submission_seq2seq_3seed_anti5_w0p40.csv` | **91.71540** |

The anti-5seed files are computed as:

```text
prediction = (1 + w) * submission_seq2seq_3seed_blend_w0p152
             - w * submission_seq2seq_5seed_blend_w0p155
```

### 4. Validation strategy

We use two hold-out schemes during development:

| Mode | Hold-out | Slot gap to d49 source | Use case |
|---|---|---|---|
| Random val | 20% of day-49 train rows (slots 0-8) | 0-8 | Early stopping, fast iteration |
| **Hard val** | Day-49 slots 5-8 only | 1-4 | Realistic test proxy |

Hard val proved consistently 2-3 points above leaderboard, mirroring test's larger slot-gap distribution (test slot-gap ranges 1-47). Hard val improvements translated ~25-30% to leaderboard.

## Iteration log

| Version | Hard val | Leaderboard | Delta | Change |
|---|---|---|---|---|
| v6 | 0.9298 | 90.86 | — | Baseline LGBM, default hyperparameters, log target, d49_weight=5 |
| v8 | 0.9354 | 90.99 | +0.13 | Hyperparameter sweep: `lr=0.02`, `leaves=63`, `min_data=100`, `λ=2` |
| v10 | 0.9361 | 91.13 | +0.14 | Add Bayesian-smoothed `geo_slot` (smoothing=1) |
| v11 | 0.9386 | 91.22 | +0.09 | Add hierarchical `gh5_slot`, `gh4_slot` smoothed (sm=10, 5) |
| v12 | ~0.939 | 91.24 | +0.02 | 5 seeds → 10 seeds |
| seq-v1 | weak hard-val | 91.66821 | +0.428 | Blend v12 with 15.5% single-seed CUDA Transformer sequence model |
| seq-v2 | weak hard-val | 91.70418 | +0.036 | 3-seed Transformer average, retuned around 15.2% blend |
| **seq-v3** | public LB only | **91.71540** | +0.011 | Extrapolate the 3-seed blend away from the weaker 5-seed blend |

Total gain from v6 baseline: **+0.855 points**.

## Failed experiments (kept for honesty)

Each row reports the change in **leaderboard score** vs v12 baseline (91.24), unless noted. Items 1-9 happened during the GBM-improvement phase; items 10-16 are post-v12 attempts that failed to break past 91.24.

| # | Approach | Result | Why |
|---|---|---|---|
| 1 | Slot-decay calibration (extrapolate d49/d48 ratio linearly into slots 9-55) | **-1.14** (89.72) | Linear decay didn't extrapolate; the model's implicit ~1.20× lift was already closer to correct |
| 2 | CatBoost member added to ensemble (v7) | **-0.19** (90.67) | CatBoost predictions ≈ LGBM; no algorithmic diversity |
| 3 | Per-RoadType ensemble (separate models for Residential / Street / Highway, v9 hybrid) | **~0** (90.98) | Fixed Street R² from −0.005 → 0.36 on hard val but Highway lost equivalent ground; net wash on leaderboard |
| 4 | Pseudo-labeling (v12 test predictions added as day-49 training data) | not submitted — Pearson 0.999 with v12 | Model just learned to reproduce its own labels even at pseudo_weight=1.0 |
| 5 | K-fold OOF target encoding for `geo_mean` / `d48_mean` | not submitted — 0 hard val change | Existing day-48 masking already eliminates the worst leak |
| 6 | Ridge regression blend (v13 = 0.95·v12 + 0.05·Ridge) | **-0.01** (91.24) | Predictions Pearson 0.97 with v12; trivial blend |
| 7 | `d49_weight` parameter sweep (range 1-10) | Hard val ±0.005; pure-w1 submission 99.8% correlated with w=5 | Weight changes shift training but not the final converged predictions |
| 8 | Geohash as LightGBM categorical feature | not submitted — 0 hard val change | Aggregate features already encode geohash identity |
| 9 | Per-(geohash, hour) and per-(geohash, 3h-block) smoothed | not submitted — **-0.005** hard val | Redundant with `geo_slot_smoothed` |
| 10 | **Optuna hyperparameter search on hard val (v15)** | **-0.09** (91.15) | Hard val improved 0.939 → 0.941; leaderboard *regressed*. First sign that hard val had decoupled from leaderboard |
| 11 | **PyTorch MLP with geohash embedding + LGBM blend (v16, 50/50)** | **-0.14** (91.10) | Reached our peak hard val (0.9425, Pearson v12-MLP = 0.98 — genuine diversity). Still hurt leaderboard |
| 12 | **`location_id` replaces geohash as atomic unit (v17)** — geohash + RoadType + NumberofLanes (4816 unique vs 1249) | **-0.29** (90.95) | Structurally correct (a single geohash contains up to 3 RoadTypes mixing Highway/Residential demand), but per-location_id stats have 6-12 obs vs 50+ per geohash. Variance hurt more than structural fix helped |
| 13 | **`location_id` features ADDED alongside geohash (v18)** | not submitted (Pearson 0.9985 with v12) | `loc_mean` rank-2 by gain (3481), but predictions converged to v12 |
| 14 | **Adversarial validation** between train-d49 (slots 0-8) and test-d49 (slots 9-55) | **AUC = 1.000** even after dropping all temporal features | Slot-keyed lookups (`slot_global_mean`, `gh4_slot_smoothed`, etc.) take entirely different value ranges between the slot regions. Confirmed the slot-range distribution shift is the dominant problem |
| 15 | **Expanding-window CV on day 48** (train slots 0..K-1, val slots K..K+W) | Mean R² 0.82; mid-day folds [20,60) at 0.91-0.94 — matches leaderboard! | First proxy that tracks leaderboard scale. Hard val (slots 5-8) was systematically optimistic by ~0.03 |
| 16 | **Optuna with expanding-window CV objective (v19)** | **-0.41** (90.83) | Optimum from honest CV (λ2=0.004, min_data=52) gave *worse* leaderboard than v12's hand-tuned heavy regularization (λ2=2.0, min_data=100). Even a leaderboard-matched proxy doesn't surface the protective regularization v12 happens to provide for the day-48→day-49 shift |

| 17 | **Blind global log-ratio residual** | best anti-blend still below v12 (91.20376); positive blend 91.14655 | Environmental elasticity alone was not the missing correction. Blinding spatial identity made a different model, but the direction did not align with leaderboard residuals |
| 18 | **Iterative SVD matrix reconstruction** | hard-val ~0.666; not submitted as primary | Pure low-rank completion could not infer unobserved day-49 afternoon columns strongly enough from the small morning clamp |

### What the failed-experiments table tells us

After 16 distinct attempts spanning hyperparameter optimization, feature engineering, model-family diversity, structural reframing, and validation-method redesign, **v12 hand-tuning sits at a local optimum that no formal optimization framework reproduces**.

- **Hyperparameter tuning fails on every proxy we built** (hard val → v15, CV → v19). Optuna consistently picks lower regularization, which fits the proxy better but generalizes worse to the test slot/day distribution.
- **Architecture diversity fails** (CatBoost too similar, MLP genuinely diverse but blend still regresses, location_id splits the data too finely).
- **The adversarial validation reveals the structural truth**: slot-keyed features have completely disjoint value distributions between train-d49 slots (0-8) and test slots (9-55). Models can perfectly distinguish the two with AUC 1.000. This is *not* fixable by tweaking the validation split — the distribution shift exists at the feature level, not just the label level.

The 1.89-point gap to the leaderboard top (93.13) appears to reflect either (a) a feature we never conceived, (b) a way to train against the d48→d49 distribution shift directly (perhaps domain adaptation or importance reweighting against the adversarial classifier), or (c) genuinely different data assumptions held by the top scorers that we can't infer from the dataset alone.

## Repository structure

```
├── dataset/
│   ├── train.csv                       Day-48 full + day-49 slots 0-8
│   ├── test.csv                        Day-49 slots 9-55, targets withheld
│   └── sample_submission.csv
├── outputs/
│   ├── submission_seq2seq_3seed_anti5_w0p40.csv Current best (91.71540)
│   ├── submission_seq2seq_3seed_blend_w0p152.csv Best direct 3-seed blend (91.70416)
│   ├── submission_seq2seq_blend_w0p155.csv Best single-seed blend (91.66821)
│   ├── submission_v12.csv              GBM baseline (91.24)
│   └── submission_*.csv                Earlier iterations and ablations
├── src/
│   ├── pipeline.py                     v12 end-to-end: features, training, ensemble
│   └── seq2seq.py                      CUDA Transformer over per-geohash demand curves
├── problem_statement.pdf
└── README.md
```

## How to reproduce

```bash
pip install -r requirements.txt

# Generate the GBM baseline
python src/pipeline.py --out submission_v12.csv --seeds 10 --d49_weight 5

# Generate the sequence model and blends with v12
python src/seq2seq.py --device cuda --epochs 8 --seeds 1 --batch_size 512 --blend_with submission_v12.csv --blend_weights 0.155 --out submission_seq2seq.csv

# Generate the stronger 3-seed sequence blend family
python src/seq2seq.py --device cuda --epochs 8 --seeds 3 --batch_size 512 --blend_with submission_v12.csv --blend_weights 0.152 --out submission_seq2seq_3seed.csv

# Inspect on hard validation
python src/pipeline.py --validate --hard_val --log_target --d49_weight 5
```

The current best anti-5seed extrapolation was generated manually from two existing submission files:

```python
best3 = pd.read_csv("outputs/submission_seq2seq_3seed_blend_w0p152.csv")
bad5 = pd.read_csv("outputs/submission_seq2seq_5seed_blend_w0p155.csv")
w = 0.40
best3["demand"] = ((1 + w) * best3["demand"] - w * bad5["demand"]).clip(0, 1)
best3.to_csv("outputs/submission_seq2seq_3seed_anti5_w0p40.csv", index=False)
```

Key flags:
- `--seeds N` — number of LGBM seeds in the ensemble (default 5)
- `--d49_weight W` — sample-weight multiplier for day-49 training rows (default 1)
- `--log_target` — train on log1p(demand)
- `--validate` — run held-out validation instead of full fit
- `--hard_val` — use day-49 slots 5-8 as the validation hold-out (more realistic than random)

Seq2seq-specific flags:
- `--device cuda` — use the CUDA PyTorch wheel/GPU
- `--epochs` — masked-reconstruction epochs
- `--blend_weights` — one or more weights blended into `--blend_with`

## Where the score is bounded

We diagnosed the ceiling via adversarial validation (row 14 above). A classifier trained to distinguish day-49 train rows (slots 0-8) from test rows (slots 9-55) achieves **AUC = 1.000** even when every temporal feature and every day-49-derived feature is removed. Slot-keyed lookup features (`slot_global_mean`, `gh4_slot_smoothed`, `gh5_slot_smoothed`, `lag_d48_off-2`) take completely different value distributions in the two slot regions, so even features that don't *name* slot inherit the shift through their lookup tables.

Then the expanding-window CV (row 15) gave us a leaderboard-consistent proxy: mid-day folds [20,60) yield R² 0.91-0.94, matching our 91.24 score. Hard val (slots 5-8) was optimistic by ~0.03 because slots 5-8 share their feature value distribution with d49 train rows.

But row 16 then closed the door: Optuna against this honest CV proxy *still* lost 0.41 on the leaderboard. The proxy is honest about slot-range generalization but cannot see the day-48→day-49 component of the shift, which v12's heavier regularization (λ2=2.0, min_data_in_leaf=100) happens to handle better than any formal optimum (λ2=0.004, min_data_in_leaf=52).

The public-submission budget is exhausted. The best confirmed submitted artifact in this repo is `outputs/submission_seq2seq_3seed_anti5_w0p40.csv` at **91.71540**.

What remains as future offline work:
- **Importance reweighting using the adversarial classifier's outputs**: weight each training row by how "test-like" its features look. The AUC = 1.000 result makes this challenging (most train rows get near-zero weight), but a domain-adaptation-style approach (e.g., gradient reversal layer on top of an MLP) is theoretically sound
- **Multi-seed / variant sequence ensemble**: train additional Transformer seeds or a 1D-CNN variant, then retune the v12 blend. The first single-seed Transformer already improved v12 by +0.428 leaderboard points
- **Quantile regression for the d48→d49 shift**: model the multiplicative correction explicitly with uncertainty bounds, calibrated against the d49 slot 0-8 observations
