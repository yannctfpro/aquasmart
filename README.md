# AquaSmart 🌱💧

**AI-powered decision support tool for precision irrigation on small and medium French farms.**

AquaSmart replaces fixed irrigation schedules with data-driven recommendations based on weather, soil moisture, crop growth stage, and recent irrigation history. It targets 15 major French crops grouped into 4 agronomic clusters, delivering substantial improvements over the professional FAO-56 water-balance baseline on the most challenging cluster.

![Python](https://img.shields.io/badge/python-3.11-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Status](https://img.shields.io/badge/status-capstone_final-orange)

---

## Problem

Traditional irrigation relies on habits or fixed calendars. With climate change and rising water costs, this approach increasingly leads to:

- **Over-irrigation** — wasted water, higher pumping costs, nutrient leaching
- **Under-irrigation** — crop stress, reduced yields
- **Complexity barrier** — existing precision farming tools are either too expensive or too complex for smaller farms

AquaSmart bridges that gap with a free, open-source, lightweight recommendation engine.

---

## Key Results

The ML pipeline is evaluated against the **FAO-56 water-balance baseline with soil reserve** (the professional reference used by modern precision agriculture tools), using a **geographic split** (train on 8 French cities, test on Montpellier + Rennes — two climatically distinct regions never seen during training).

| Cluster | Crops | Baseline F1 | ML F1 | MAE gain vs baseline |
|---------|-------|:-----------:|:-----:|:--------------------:|
| 1 — Winter cereals        | Winter wheat, durum wheat, winter barley, oats, triticale | 0.723 | 0.549 | **−39.7%** |
| 2 — Summer deep-rooted    | Corn, sunflower, sorghum, soybean                         | 0.939 | **0.949** | **+20.0%** |
| 3 — Winter oilseeds       | Rapeseed, winter pea, faba bean                           | 0.710 | 0.680 | **+18.9%** |
| 4 — Shallow-rooted crops  | Potato, sugar beet, field vegetables                      | 0.337 | **0.945** | **+91.9%** |

**Headline**: +91.9% MAE improvement on shallow-rooted crops (cluster 4), where dynamic soil management matters most. For winter cereals (cluster 1), the rule-based baseline already captures most of the signal — a honest finding demonstrating where ML adds value and where it does not.

Full methodology, error analysis, and discussion are in the [Technical Report](#technical-report).

---

## Architecture

```
aquasmart/
├── config/
│   └── config.yaml              # Project configuration
├── data/
│   ├── raw/                     # Weather cache + v4 crop-specific datasets
│   └── processed/               # Preprocessed data per cluster (gitignored, heavy)
│       ├── cluster_1/           # data.npz, meta_test.csv, scaler.pkl
│       ├── cluster_2/
│       ├── cluster_3/
│       └── cluster_4/
├── models/                      # Trained model artefacts (gitignored)
│   ├── cluster_1/               # classifier.pkl, regressor.pkl
│   ├── cluster_2/
│   ├── cluster_3/
│   ├── cluster_4/
│   └── results_v4.csv           # Consolidated evaluation metrics
├── notebooks/
│   └── 01_eda.ipynb             # Exploratory Data Analysis
├── reports/
│   └── figures/                 # Plots used in the technical report
├── src/
│   ├── api/                     # FastAPI backend (optional)
│   ├── baselines/
│   │   └── baseline_fao56.py    # FAO-56 simple + with soil reserve
│   ├── core/
│   │   └── recommendation_engine.py  # Shared inference engine (Streamlit + FastAPI)
│   ├── data/
│   │   ├── collect_data_v2.py   # Open-Meteo weather collection (10 cities, 5 years)
│   │   ├── generate_target_v4.py # Per-crop water-balance simulation (variable doses)
│   │   └── preprocess_v4.py     # Temporal features + geographic split
│   └── models/
│       └── train_v4.py          # Two-stage training per cluster
├── Dockerfile                   # Containerization
├── docker-compose.yml
├── requirements.txt
├── streamlit_app.py             # Farmer-facing dashboard
├── .gitignore
└── README.md
```

**Design principles**:
- **One model per agronomic cluster**, not per crop — avoids fragmenting training data across 15 individual crops while retaining crop specificity via features (Kc coefficient, soil water capacity, growth stage).
- **Two-stage ML pipeline**: classification ("irrigate today?") → regression ("how many mm?").
- **Clean separation**: `src/data/` (data ops), `src/models/` (training), `src/core/` (inference), `src/api/` (HTTP layer), `streamlit_app.py` (UI). None of these can reach into another's internals.

---

## Installation

### Option 1 — Local virtual environment (recommended for development)

Requires **Python 3.11** installed on your system.

```bash
git clone https://github.com/<username>/aquasmart.git
cd aquasmart

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate           # Linux/macOS
# or:  .\venv\Scripts\Activate.ps1  (Windows PowerShell)

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2 — Docker (recommended for reproducible deployment)

Requires **Docker** and **Docker Compose** installed.

```bash
git clone https://github.com/<username>/aquasmart.git
cd aquasmart

# Build and start the Streamlit app (and the FastAPI backend if wanted)
docker compose up --build

# The app will be available at http://localhost:8501
```

To stop: `docker compose down`.

---

## Running AquaSmart

### Quick-start: Streamlit dashboard (local inference)

```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501). Select a city, a crop, optionally provide recent irrigation history, and get an irrigation recommendation based on today's weather from Open-Meteo.

### Model artefacts

Trained model bundles live in `models/cluster_<N>/` and are **not checked into the repository** (too heavy). Either:
- Download the pre-trained models archive from the project releases page, unzip into `models/`
- Or rebuild them from scratch — see the full pipeline below

---

## Full pipeline reproduction (from scratch)

Run these commands in order from the project root. Total time: ~15 minutes on a laptop, mostly dominated by the weather API calls during step 1.

```bash
# 1) Collect weather data for 10 French cities over 5 years (2020-2024)
#    → writes data/raw/weather_cache.csv
python src/data/collect_data_v2.py

# 2) Generate per-crop datasets with variable irrigation doses
#    → writes data/raw/aquasmart_v4_<crop>.csv for the 15 crops
python src/data/generate_target_v4.py

# 3) Preprocess and split by cluster (geographic split: Montpellier + Rennes held out)
#    → writes data/processed/cluster_<N>/data.npz, scaler.pkl, meta_test.csv
python src/data/preprocess_v4.py

# 4) Train two-stage ML pipeline per cluster, compare to FAO-56 baselines
#    → writes models/cluster_<N>/classifier.pkl, regressor.pkl
#    → writes models/results_v4.csv with metrics summary
python src/models/train_v4.py
```

Once `models/cluster_*/` is populated, launch the Streamlit app and everything works.

---

## Data Sources

All data sources are free and do not require an API key.

| Source | Content | Resolution | Used for |
|--------|---------|------------|----------|
| **Open-Meteo Historical API** | Temperature, humidity, precipitation, ET₀ (FAO Penman-Monteith), wind, soil moisture | 9-25 km, daily | Training + inference |
| **FAO-56 Kc tables** | Crop coefficient per growth stage (15 crops) | Per crop | Target engineering |
| **FAO-56 water balance** | Professional irrigation baseline with soil reserve | Simulation | Comparison reference |

Weather variables used as features: `temperature_2m_mean`, `relative_humidity_2m_mean`, `precipitation_sum`, `et0_fao_evapotranspiration`, `wind_speed_10m_max`, `soil_moisture_0_to_7cm_mean`.

---

## Methodology at a glance

**Target engineering** — A daily water-balance simulation (the "checkbook method" used by agronomists) produces irrigation targets:
- Maintain a soil water stock bounded by `RU_max` (soil water holding capacity, crop-dependent)
- Daily update: `stock += 0.8 × rainfall − ETc`, where `ETc = ET₀ × Kc`
- Irrigate when stock drops below 50% of `RU_max`, refilling to 70-90% (dose scaled by recent evapotranspiration pressure)
- Dose is **variable** between 15 and 40 mm — this is what the regressor learns to predict

**Clustering** — 15 crops mapped to 4 clusters based on rooting depth, growth cycle, and seasonality:
1. Winter cereals (medium roots, winter cycle)
2. Summer deep-rooted crops (long cycle, high demand)
3. Winter oilseeds / legumes (medium-deep, 2-season cycle)
4. Shallow-rooted row crops (low water capacity, sensitive to stress)

**Features (16 total)** — 9 static (weather + crop Kc + `ru_max`) + 7 temporal (7/14-day rolling sums of rainfall, ETc, irrigation; days since last irrigation; previous-day soil stock). Temporal features enable the model to track the farm's actual hydric state rather than treating each day in isolation.

**Models** — Two-stage pipeline per cluster, with automatic selection between Logistic Regression, Random Forest, and Gradient Boosting:
- Stage 1 (classification): best F1 on imbalanced target (~2-3% positive class)
- Stage 2 (regression): best RMSE on irrigation-day subset

**Baselines** — Two FAO-56 references, not the naive "daily average" strawman:
- `fao56_simple`: daily formula `max(0, ETc − 0.8 × rainfall)` with no soil memory
- `fao56_with_ru`: full water-balance simulation matching the target generator's logic (this is the tough baseline ML must beat)

**Split** — Geographic: train on 8 cities (Chartres, Amiens, Reims, Melun, Strasbourg, Dijon, Toulouse, Bordeaux), test on **Montpellier (Mediterranean dry) and Rennes (Atlantic humid)** — the two most climatically distinct cities in the collection. Both cities are held out across all 5 years (2020-2024), ensuring the model is evaluated on unseen geographic conditions.

---

## Limitations & Ethics

**Limitations**
- No real soil moisture sensors — we rely on Open-Meteo's modeled soil moisture proxy
- Growth stage derived from a simplified sowing-month calendar, not real phenology observations
- Training targets come from a water-balance simulation, not from real farmer irrigation logs — the model learns to replicate a well-calibrated agronomic rule with additional signals, not to replicate individual farmer preferences
- Cluster 1 and 3 show the ML is not strictly better than a well-tuned FAO-56 baseline in scenarios where the rule-based approach already captures most of the signal — ML's marginal value depends on the cluster

**Ethics**
- Free, open-source — no paywall prevents smaller farms from accessing the tool
- Transparent: all code, baselines, and model artefacts are inspectable
- No user data is stored by the app; weather queries are proxied to Open-Meteo on-demand
- Carbon footprint: training on a laptop CPU in under 2 minutes per cluster; no GPU required

---

## Tech Stack

- **Python 3.11** — core language
- **scikit-learn** — ML models (Logistic Regression, Random Forest, Gradient Boosting)
- **pandas / NumPy** — data manipulation
- **Streamlit** — farmer-facing dashboard
- **FastAPI + Uvicorn** — optional HTTP backend
- **Plotly** — interactive charts
- **Open-Meteo API** — live weather for inference
- **Docker / Docker Compose** — reproducible deployment

---

## Technical Report

The full academic technical report (problem framing, data exploration, modeling experimentation, error analysis, interpretability, ethics) is available under `reports/` along with all generated figures in `reports/figures/`.

---

## License

MIT License — university capstone project. Free to use, study, and adapt.

## Authors

Capstone project — SKEMA AI Capstone, 2026.