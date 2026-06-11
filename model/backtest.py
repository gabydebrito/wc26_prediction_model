import pandas as pd
import numpy as np
from data_prep import load_and_prepare
from poisson_model import fit, predict_outcome_probs

def brier_score(p_home, p_draw, p_away, actual_home, actual_away):
    if actual_home > actual_away:
        r_home, r_draw, r_away = 1, 0, 0
    elif actual_home < actual_away:
        r_home, r_draw, r_away = 0, 0, 1
    else:
        r_home, r_draw, r_away = 0, 1, 0
    return (p_home - r_home)**2 + (p_draw - r_draw)**2 + (p_away - r_away)**2

def backtest_wc(results_path, fifa_csv_path, wc_year=2022, reg_strength=50.0):
    # Load all data
    df_all = load_and_prepare(results_path, min_year=1993)

    # Cutoff: only train on matches strictly before the WC
    cutoff = pd.Timestamp(f'{wc_year}-11-01')  # WC 2022 started Nov 20
    df_train = df_all[df_all['date'] < cutoff].copy()

    # Test set: 2022 WC group stage matches only (scores are known)
    df_raw = pd.read_csv(results_path, parse_dates=['date'])
    df_test = df_raw[
        (df_raw['tournament'] == 'FIFA World Cup') &
        (df_raw['date'].dt.year == wc_year) &
        (df_raw['home_score'].notna())
    ].copy()

    print(f"Training on {len(df_train)} matches before {cutoff.date()}")
    print(f"Testing on {len(df_test)} WC {wc_year} matches")

    # Fit model on training data only
    params = fit(df_train, fifa_csv_path=fifa_csv_path, reg_strength=reg_strength)
    print(f"Model fit: success={params['success']}, log-likelihood={params['log_likelihood']:.2f}")

    # Score each test match
    scores = []
    missing = []

    for _, row in df_test.iterrows():
        home, away = row['home_team'], row['away_team']

        # Skip if either team wasn't in training data
        if home not in params['attack'] or away not in params['attack']:
            missing.append(f"{home} vs {away}")
            continue

        probs = predict_outcome_probs(home, away, params, max_goals=10)
        bs = brier_score(probs['home_win'], probs['draw'], probs['away_win'],
                         row['home_score'], row['away_score'])
        scores.append({
            'match': f"{home} vs {away}",
            'p_home': probs['home_win'],
            'p_draw': probs['draw'],
            'p_away': probs['away_win'],
            'result': ('H' if row['home_score'] > row['away_score']
                       else 'D' if row['home_score'] == row['away_score'] else 'A'),
            'brier': bs
        })

    if missing:
        print(f"\nSkipped {len(missing)} matches (teams not in training data):")
        for m in missing: print(f"  {m}")

    results_df = pd.DataFrame(scores)
    mean_bs = results_df['brier'].mean()

    print(f"\nMean Brier score: {mean_bs:.4f}")
    print(f"(baseline random: 0.6667, naive 'home wins every game': ~0.56)")
    print(f"\nWorst predicted matches:")
    print(results_df.nlargest(5, 'brier')[['match', 'p_home', 'p_draw', 'p_away', 'result', 'brier']].to_string(index=False))

    return results_df, mean_bs

if __name__ == '__main__':
    # results_df, mean_bs = backtest_wc(
    #     results_path='data/results.csv',
    #     fifa_csv_path='data/elo_ratings_wc2026.csv',
    #     wc_year=2022,
    #     reg_strength=50.0
    # )

    for rs in [10, 50, 100, 200, 500]:
        _, bs = backtest_wc('data/results.csv', 'data/elo_ratings_wc2026.csv', reg_strength=rs)
        print(f"reg_strength={rs}: Brier={bs:.4f}")
