"""
This module is responsible for preparing the data for the model.
It includes functions for loading, cleaning, and
transforming the data as needed for the modeling process
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# tournaments weighted by importance, WC most important, then qualifiers, then friendlies

TOURNAMENT_WEIGHTS = {
    'FIFA World Cup': 3.0,
    'UEFA European Championship': 2.5,
    'Copa América': 2.5,
    'Africa Cup of Nations': 2.0,
    'AFC Asian Cup': 2.0,
    'CONCACAF Gold Cup': 2.0,
    'FIFA World Cup qualification': 1.5,
    'UEFA European Championship qualification': 1.5,
    'Copa América qualification': 1.5,
    'Friendly': 0.5
}

DEFAULT_TOURNAMENT_WEIGHT = 1.0

def load_and_prepare(file_path: str, min_year: int = 1993, time_decay_years: float = 3.0,
                     reference_date: datetime | None = None) -> pd.DataFrame:
    """
    Loads results.csv and returns clean, weighted DataFrame.

    Parameters:
    - file_path: Path to the raw results CSV
    - min_year: drop matches before this year, reasonable because older matches
        less relevant and it covers rule changes and modern era and soccer and politics
    - time_decay_years: controls how fast older mathces lose weight
    - reference_date: if provided, used to calculate time decay based on match date

    Returns:
    pd.DataFrame with columns:
    date, home_team, away_team, home_score, away_score,
            tournament, neutral,
            tournament_weight,   ← tier weight for the competition
            time_weight,         ← exponential decay from reference_date
            match_weight         ← tournament_weight × time_weight (use this)
    """
    if reference_date is None:
        reference_date = datetime.today()

    df = pd.read_csv(file_path, parse_dates=['date'])

    df = df.dropna(subset=["home_score", "away_score"])

    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)


    df = df[df['date'].dt.year >= min_year].copy()

    #drop unnecessary columns
    df = df.drop(columns=["city", "country"], errors="ignore")

    # Add tournament weight and time weight
    df['tournament_weight'] = (df['tournament'].map(TOURNAMENT_WEIGHTS).fillna(DEFAULT_TOURNAMENT_WEIGHT))
    #time decay weight as exponential decay, so
    # weight = 0.5^(days_old / half_life_days)
    days_old = (reference_date - df["date"]).dt.days.clip(lower=0)
    half_life_days = time_decay_years * 365.25
    df["time_weight"] = np.power(0.5, days_old / half_life_days)

    df['match_weight'] = df['tournament_weight'] * df['time_weight']

    df = df.sort_values("date").reset_index(drop=True)

    return df

if __name__ == "__main__":
    import sys

    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/results.csv"
    df = load_and_prepare(filepath)

    print(f"Shape:        {df.shape}")
    print(f"Date range:   {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Null scores:  {df[['home_score','away_score']].isnull().sum().sum()}")
    print(f"\nWeight range: {df['match_weight'].min():.4f} → {df['match_weight'].max():.2f}")
    print(f"\nTop tournaments by avg weight:\n"
          f"{df.groupby('tournament')['tournament_weight'].first().sort_values(ascending=False).head(10)}")
    print(f"\nSample rows:\n{df.tail(5).to_string()}")
