# wc26_prediction_model

A Monte Carlo tournament simulator for the 2026 FIFA World Cup using a Dixon-Coles Poisson model fit on historical international results.

## Structure
- `data/` — raw results CSV (gitignored)
- `model/data_prep.py` — data cleaning, filtering, and match weighting
- `explore_data.ipynb` — initial data exploration
- `model/poisson_model.py` — poisson model for predicting goals scored

## Data
International football results from 1872–present via [Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017).
