import numpy as np
import pandas as pd
from collections import defaultdict
from data_prep import load_and_prepare
from simulator import simulate_tournament, GROUPS
from poisson_model import fit

N_SIMS = 10000
ROUND_ORDER = ['Group Stage', 'Round of 32', 'Round of 16',
               'Quarter-Final', 'Semi-Final', 'Final', 'Winner']

def run_sim(n: int, params:dict) -> pd.DataFrame:
    counts = defaultdict(lambda: defaultdict(int))

    for i in range(n):
        if (i+1) % 1000 == 0:
            print(f"Simulation {i+1}/{n}")
        results = simulate_tournament(GROUPS, params)
        for team, round_reached in results.items():
            counts[team][round_reached] += 1

    rows = []
    for team in sorted(counts.keys()):
        row = {"team": team}
        for round_name in ROUND_ORDER:
            row[round_name] = counts[team][round_name] / n * 100
        rows.append(row)

    df = pd.DataFrame(rows).set_index("team")
    return df

if __name__ == '__main__':
    print("Loading data and fitting model...")
    wc_teams = {team for teams in GROUPS.values() for team in teams}
    df = load_and_prepare('data/results.csv')
    # df = df[df['home_team'].isin(wc_teams) & df['away_team'].isin(wc_teams)]
    params1 = fit(df, fifa_csv_path='data/elo_ratings_wc2026.csv', reg_strength=50.0)
    params2 = fit(df, fifa_csv_path='data/elo_ratings_wc2026.csv', reg_strength=200.0)
    print(f"Model fit: success={params1['success']}, log-likelihood={params1['log_likelihood']:.2f}\n")

    # print("\nAttack strengths:")
    # for t, v in sorted(params['attack'].items(), key=lambda x: -x[1]):
    #     print(f"{t:<25} {v:.4f}")

    # print("\nDefense strengths (lower = harder to score against):")
    # for t, v in sorted(params['defense'].items(), key=lambda x: x[1]):
    #     print(f"{t:<25} {v:.4f}")
    # print(f"Running {N_SIMS} simulations...")

    results = run_sim(N_SIMS, params1)

    pd.set_option('display.float_format', '{:.1f}'.format)
    pd.set_option('display.max_rows', 60)
    pd.set_option('display.width', 120)
    print("\nTournament probabilities (%):")
    print(results.sort_values('Winner', ascending=False).to_string())

    # for param in [params1, params2]:
    #     for team in ["Argentina","Brazil","Colombia","Ecuador",
    #          "Spain","France","England","Portugal"]:
    #         fifa_prior = param['fifa_prior']
    #         print(f"\n{team:<25} {fifa_prior[team]:.4f} {param['attack'][team]:.4f} {param['defense'][team]:.4f}")
