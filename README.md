# Flipkart GRaD (Grid) - Traffic Demand Prediction

This repository contains our solution and submissions for the Flipkart Grid hackathon, focusing on Spatio-Temporal Traffic Demand Prediction.

##  Problem Statement
The challenge is to forecast traffic demand (e.g., number of orders, vehicle density) at a granular spatio-temporal level. We are provided with varying temporal intervals and spatial identifiers (Geohashes) spanning across two main days (Day 48 and Day 49). 

The goal is to accurately predict the traffic demand for upcoming time slots (Day 49, slots 9-55) using historical data (Day 48, slots 0-95 and Day 49, slots 0-8).

##  Approach & Methodology

Our approach frames this as a **Time-Series Regression** problem on structured tabular data, heavily utilizing historical lags and target encoding to capture temporal seasonality and spatial recurring behaviors. 

### 1. Data Processing & Feature Engineering
We perform robust feature engineering focused on extracting maximum signal from spatial and chronological relationships without introducing data leakage. 

- **Geohash Decoding**: Converted spatial Geohash strings into high-precision continuous `Latitude` and `Longitude` numerical coordinates to allow continuous distance evaluations.
- **Cyclical Time Encoding**: Extracted `hour` and `minute` and assigned consecutive `slots` (15-min bins). Added cyclical sine and cosine transformations (`slot_sin`, `slot_cos`) for models to map the daily cyclical nature of traffic.
- **Aggregated Imputation**: Instead of global mean imputation, we fill missing static features (`RoadType`, `Weather`, `Temperature`) using the **mode or mean for that specific geohash**.
- **Categorical Mapping**: Ordinal encoding of categorical properties like `RoadType` (Residential/Street/Highway), `Weather`, and binary variables (`LargeVehicles`, `Landmarks`).

### 2. Spatio-Temporal Target Features (Lags & Summaries)
To capture spatial momentum, we generate aggressive Lag attributes:
- **Same-Slot Lags**: Demand at the exact same geographic location & time slot from the previous 24 hours (Day 48).
- **Neighbor-Slot Smoothing**: Gathered demand values at `slot ± 1` and `slot ± 2` on Day 48 to compute a 5-slot moving window mean (helps smooth out isolated demand spikes).
- **Geohash Rolling Summary Stats**: Extracted historical statistics for each geohash—`mean`, `median`, `max`, `std`, and `90th-percentile` to evaluate base congestion scale per area.
- **Day-49 Calibration Adjustments**: Extracted ratio and delta between overlapping Day 49 (morning) and Day 48 slots. We multiply this `d49/d48 ratio` calibration with Day 48 lags to offset recent macro spikes on the target day.
- **Recent Momentum Tracks**: Added the exact preceding day-49 demand immediately strictly before the target slot prediction to maintain sequential momentum.

### 3. Model Architecture
- **Algorithm**: **LightGBM (Gradient Boosted Decision Trees)**
- **Why LightGBM?** 
  - Iterative tree models are exceptional at mapping continuous variables (Lat/Lng) against disjointed categorical locations.
  - Effectively manages sparse arrays / NaNs untouched by imputation.
  - Highly robust towards complex interactions between temporal (slots) and spatial (Lags over Geohash) features.

### 4. Validation Strategy
Created specialized cross-validation splits over `Day 49`. A random sampling method performs inadequately for timeseries, so we implemented a **"Hard Validation Mask"**, selectively holding out grouped continuous subsequent slots (e.g., day-49 slots 5-8) to mirror testing distribution (day-49 slots 9-55).

##  Repository Structure

```text
├── dataset/
│   ├── train.csv                 # Historical training data (Day 48 full, Day 49 early)
│   ├── test.csv                  # Target predictions (Day 49 mid)
│   └── sample_submission.csv
├── outputs/
│   ├── submission_v1.csv         # Initial baseline features predictions
│   ├── submission_v2.csv         # Interpolated lags & missing imputation added
│   └── submission_v3.csv         # Calibration scales & Localized geohash stats tuned 
├── src/
│   └── pipeline.py               # Main modular ML pipeline & LightGBM logic
├── README.md                     # Approach details and instructions
└── .gitignore            
```
