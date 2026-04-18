# AquaSmart 🌱💧

**An AI-powered decision support tool to optimize water consumption through intelligent irrigation management.**

AquaSmart replaces fixed irrigation schedules with data-driven recommendations based on weather, soil conditions, and crop growth stages — helping farmers irrigate at the right time with the right amount of water.

## Project Structure

```
aquasmart/
├── config/                  # Configuration files
│   └── config.yaml
├── data/
│   ├── raw/                 # Original datasets (never modified)
│   ├── processed/           # Cleaned and feature-engineered data
│   └── external/            # Data from APIs (AgERA5, SoilGrids, etc.)
├── models/                  # Trained model artifacts (.pkl, .joblib)
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── reports/
│   └── figures/             # Generated plots and visualizations
├── src/
│   ├── data/                # Data loading and cleaning
│   ├── features/            # Feature engineering
│   ├── models/              # Training and prediction
│   ├── visualization/       # Plotting utilities
│   └── api/                 # FastAPI endpoints
├── tests/                   # Unit tests
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

```bash
git clone https://github.com/<your-username>/aquasmart.git
cd aquasmart
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_eda.ipynb
```

## Run AquaSmart

Local Streamlit mode uses the shared inference engine directly:

```bash
streamlit run streamlit_app.py
```

FastAPI mode is also available if you want the UI to call the backend over HTTP:

```bash
uvicorn src.api.main:app --reload --port 8000
streamlit run streamlit_app.py
```

If model artifacts are missing, rebuild them with:

```bash
python src/data/collect_data_v2.py
python src/data/preprocess_v2.py
python src/models/train_v2.py
```

## Data Sources

| Category | Source | Resolution | Access |
|----------|--------|------------|--------|
| Weather + ET₀ | Copernicus AgERA5 | ~10 km daily | Free (CDS API) |
| Weather | Open-Meteo | 9-25 km | Free, no key |
| Soil properties | ISRIC SoilGrids 2.0 | 250 m | Free (REST API) |
| Soil moisture | NASA SMAP | ~9 km | Free (Earthdata) |
| MVP Baseline | Kaggle | N/A | Free |

## Tech Stack

- **Python** — Core language
- **Scikit-learn** — ML models
- **Pandas / NumPy** — Data manipulation
- **Matplotlib / Seaborn / Plotly** — Visualization
- **FastAPI** — Model serving API
- **Streamlit** — Dashboard prototype

## License

University capstone project. All rights reserved.
