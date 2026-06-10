import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from typing import Optional

FIFA_NAME_MAP = {
    "USA":                        "United States",
    "IR Iran":                    "Iran",
    "Korea Republic":             "South Korea",
    "Korea DPR":                  "North Korea",
    "Côte d'Ivoire":              "Ivory Coast",
    "Türkiye":                    "Turkey",
    "China PR":                   "China",
    "Cabo Verde":                 "Cape Verde",
    "São Tomé e Príncipe":        "Sao Tome and Principe",
    "North Macedonia":            "Macedonia",
    "Trinidad and Tobago":        "Trinidad & Tobago",
    "St. Kitts and Nevis":        "Saint Kitts and Nevis",
    "St. Lucia":                  "Saint Lucia",
    "St. Vincent / Grenadines":   "Saint Vincent and the Grenadines",
    "Antigua and Barbuda":        "Antigua and Barbuda",
    "Dominican Rep.":             "Dominican Republic",
    "Guinea-Bissau":              "Guinea-Bissau",
    "Eswatini":                   "Swaziland",
    "Congo DR":                   "DR Congo",
    "Kyrgyz Republic":            "Kyrgyzstan",
    "Brunei Darussalam":          "Brunei",
    "Czech Republic":             "Czechia"
}

def load_fifa_points(fifa_csv_path: str) -> dict:
    """
    Load the most recent FIFA ranking points for each team.

    Returns dict mapping team name to total points.
    Team names translated using FIFA_NAME_MAP to match those in results.csv.
    """
    df = pd.read_csv(fifa_csv_path)
    df.columns = [c.strip() for c in df.columns]

    df = df.sort_values("snapshot_date", ascending=False)
    latest = df.groupby("country", as_index=False).first()

    pts = {}
    for _, row in latest.iterrows():
        name = FIFA_NAME_MAP.get(row['country'], row['country'])
        pts[name] = float(row['rating'])

    return pts

def _build_fifa_prior(teams: np.ndarray, fifa_points: dict) -> np.ndarray:
    """
    Returns normalized FIFA points vector for given teams
    If a team is missing from fifa_points, it gets the average points of all teams.
    Vector normalized so mean is 1, to be used as a prior for attack strength in the model,
    which matches the constraint.
    """

    raw = np.array([fifa_points.get(team, np.nan) for team in teams])
    avg_pts = np.nanmedian(raw)

    raw = np.where(np.isnan(raw), avg_pts, raw)

    return raw / np.mean(raw)

def build_params(teams: np.ndarray, fifa_prior: Optional[np.ndarray]) -> np.ndarray:
    """
    Builds initial flat parameter for scipy.optiimize.minimize
    All attack/defense strengths start at 1, and home advantage starts at 1.1
    """
    n = len(teams)
    attack = fifa_prior if fifa_prior is not None else np.ones(n)
    defense = np.ones(n)
    home_adv = np.array([1.1])
    rho = np.array([-0.05])
    return np.concatenate([attack, defense, home_adv, rho])

def unpack_params(params: np.ndarray, teams: np.ndarray) -> tuple[dict, dict, float, float]:
    """
    Unpack flat parameters into attack/defense strength dictionaries
    and home advantage multiplier
    """
    n = len(teams)
    attack = dict(zip(teams, params[:n]))
    defense = dict(zip(teams, params[n:2*n]))
    home_adv = float(params[2*n])
    rho = float(params[2*n + 1])
    return attack, defense, home_adv, rho

def pack_params(attack: dict, defense: dict, home_adv: float, rho: float, teams: np.ndarray) -> np.ndarray:
    """
    Pack attack/defense strength dictionaries and home advantage multiplier
    into a flat parameter array for optimization
    """
    attack_values = np.array([attack[t] for t in teams])
    defense_values = np.array([defense[t] for t in teams])
    return np.concatenate([attack_values, defense_values, [home_adv], [rho]])

def tau_factor(x, y, _lambda, mu, rho):
    """
    Returns tau factor for a single match based on scoreline
    """
    if x == 0 and y == 0:
        return 1 - _lambda * mu * rho
    elif x == 0 and y == 1:
        return 1 + _lambda * rho
    elif x == 1 and y == 0:
        return 1 + mu * rho
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0

def log_likelihood(params: np.ndarray, df: pd.DataFrame, teams: np.ndarray,
                   fifa_prior: Optional[np.ndarray], reg_strength: float = 0.0) -> float:
    """
    Returns total log likelihood of the observed match results under independent
    Poisson model with given parameters

    Optional FIFA prior regularization:
        penalty = -reg_strength * sum((attack_i - fifa_prior_i)^2)
    This pulls attack strengths towards the FIFA points prior, which can help with convergence and
    """
    # Extract attack/defense strengths from params
    attack_strength, defense_strength, home_adv, rho = unpack_params(params, teams)

    # Calculate expected goals for each match
    home_attack = df['home_team'].map(attack_strength).to_numpy()
    away_attack = df['away_team'].map(attack_strength).to_numpy()
    home_defense = df['home_team'].map(defense_strength).to_numpy()
    away_defense = df['away_team'].map(defense_strength).to_numpy()

    lambda_home = home_attack * away_defense * home_adv
    lambda_away = away_attack * home_defense

    home_scores = df['home_score'].to_numpy()
    away_scores = df['away_score'].to_numpy()

    base_log_like = (poisson.logpmf(home_scores, lambda_home) +
     poisson.logpmf(away_scores, lambda_away))

    if "match_weight" in df.columns:
        weights = df["match_weight"].to_numpy()
    else:
        weights = np.ones(len(df))

    tau = np.array([tau_factor(x, y, lh, la, rho) for x, y, lh, la
                    in zip(home_scores, away_scores, lambda_home, lambda_away)])

    tau = np.clip(tau, 1e-10, None)

    log_like = base_log_like + np.log(tau)
    log_like *= weights

    if fifa_prior is not None and reg_strength > 0:
        n = len(teams)
        attack_vec = params[:n]
        prior_penalty = reg_strength *np.sum((attack_vec - fifa_prior) ** 2)

        return float(log_like.sum()) - prior_penalty
    return float(log_like.sum())

def _build_bounds(n_teams: int):
    """
    Keep attack/defense strengths positive; home advantage > 0.
    scipy L-BFGS-B uses (lower, upper) tuples; None means unbounded.
    """
    eps = 1e-6
    attack_bounds  = [(0.1, 5.0)] * n_teams
    defense_bounds = [(0.1, 5.0)] * n_teams
    ha_bounds      = [(1.0, 2.0)]
    rho_bounds     = [(-0.2, 0.2)]  # reasonable range for Dixon-Coles rho
    return attack_bounds + defense_bounds + ha_bounds + rho_bounds

def _avg_attack_constraint(params: np.ndarray, n_teams: int):
    """
    Identification constraint: mean(attack) == 1.
    Without this the attack and defense scales are not separately identified.
    """
    return params[:n_teams].mean() - 1.0

def fit(df: pd.DataFrame, verbose:bool = False,
        fifa_csv_path: Optional[str] = None,
        reg_strength: float = 200.0) -> dict:
    """
    Fits the Poisson model to the data and returns dictionary with parameters
    """
    teams = np.sort(pd.unique(df[['home_team', 'away_team']].values.ravel()))
    n = len(teams)
    bounds = _build_bounds(n)

    fifa_prior = None
    if fifa_csv_path is not None:
        fifa_points = load_fifa_points(fifa_csv_path)
        fifa_prior = _build_fifa_prior(teams, fifa_points)

        if verbose:
            covered = sum(1 for team in teams if team in fifa_points)
            print(f"FIFA prior: {covered}/{n} teams covered, reg_strength={reg_strength}")

    params = build_params(teams, fifa_prior)

    constraints = {
        'type': 'eq',
        'fun': lambda p: _avg_attack_constraint(p, n)
    }

    result = minimize(fun = lambda p: -log_likelihood(p, df, teams, fifa_prior, reg_strength),
                      x0 = params,
                      method = 'SLSQP',
                      bounds = bounds,
                      constraints = constraints,
                      options = {'maxiter': 2000, 'ftol': 1e-10}
    )
    if verbose:
        print(result)

    if not result.success:
        import warnings
        warnings.warn(f"Optimisation did not converge: {result.message}")

    attack, defense, home_adv, rho = unpack_params(result.x, teams)

    return {
        'attack':         attack,
        'defense':        defense,
        'home_advantage': home_adv,
        'teams':          teams,
        'success':        result.success,
        'log_likelihood': -result.fun,
        'rho':           rho,
        'fifa_prior':     dict(zip(teams, fifa_prior)) if fifa_prior is not None else None,
    }

def predict(home: str, away: str, params: dict, neutral: bool=False) -> tuple[float, float]:
    """
    Predicts the expected goals for a single match between home and away teams
    Parameters:
    - home: name of the home team
    - away: name of the away team
    - params: dict returned by fit
    Returns:
    Tuple of (expected_home_goals, expected_away_goals)
    """
    attack = params['attack']
    defense = params['defense']
    home_adv = 1.0 if neutral else params['home_advantage']

    return (attack[home] * defense[away] * home_adv,
            attack[away] * defense[home])

def predict_scoreline_probs(home: str, away: str, params: dict,
                            max_goals: int = 10, neutral: bool = False) -> pd.DataFrame:
    """
    Return a (max_goals+1) x (max_goals+1) DataFrame where cell [i, j] is
    P(home scores i, away scores j).
    """
    lh, la = predict(home, away, params, neutral=neutral)
    rho = params['rho']

    home_probs = poisson.pmf(np.arange(max_goals + 1), lh)
    away_probs = poisson.pmf(np.arange(max_goals + 1), la)

    matrix = np.outer(home_probs, away_probs)

    #apply Dixon-Coles tau adjustment for low-scoring outcomes
    for i, j in [(0,0), (0,1), (1,0), (1,1)]:
        matrix[i, j] *= tau_factor(i, j, lh, la, rho)

    matrix /= matrix.sum()  # normalize to ensure probabilities sum to 1

    return pd.DataFrame(matrix,
                        index=pd.RangeIndex(max_goals + 1, name='home_goals'),
                        columns=pd.RangeIndex(max_goals + 1, name='away_goals'))


def predict_outcome_probs(home: str, away: str,
                          params: dict, max_goals: int = 10) -> dict:
    """
    Return win/draw/loss probabilities for the home team.
    """
    matrix = predict_scoreline_probs(home, away, params, max_goals)
    home_win = float(np.tril(matrix.values, k=-1).sum())
    draw     = float(np.trace(matrix.values))
    away_win = float(np.triu(matrix.values, k=1).sum())
    return {'home_win': home_win, 'draw': draw, 'away_win': away_win}



if __name__ == '__main__':
    # Minimal synthetic dataset so the script runs without external data
    rng = np.random.default_rng(42)
    teams_demo = ['Arsenal', 'Chelsea', 'Liverpool', 'Man City',
                  'Man Utd',  'Spurs',   'Newcastle', 'Everton']

    rows = []
    for _ in range(200):
        h, a = rng.choice(teams_demo, size=2, replace=False)
        rows.append({
            'home_team':  h,
            'away_team':  a,
            'home_score': int(rng.poisson(1.5)),
            'away_score': int(rng.poisson(1.1)),
        })
    df_demo = pd.DataFrame(rows)

    print("Fitting model …")
    fitted = fit(df_demo, verbose=False)

    print(f"\nConverged : {fitted['success']}")
    print(f"Log-lik   : {fitted['log_likelihood']:.2f}")
    print(f"Home adv  : {fitted['home_advantage']:.4f}")

    print("\nAttack strengths:")
    for t, v in sorted(fitted['attack'].items(), key=lambda x: -x[1]):
        print(f"  {t:<12} {v:.4f}")

    print("\nDefence strengths (lower = harder to score against):")
    for t, v in sorted(fitted['defense'].items(), key=lambda x: x[1]):
        print(f"  {t:<12} {v:.4f}")

    home_team, away_team = 'Arsenal', 'Chelsea'
    lh, la = predict(home_team, away_team, fitted)
    print(f"\n{home_team} vs {away_team}")
    print(f"  λ_home = {lh:.3f}   λ_away = {la:.3f}")

    probs = predict_outcome_probs(home_team, away_team, fitted)
    print(f"  Home win : {probs['home_win']:.1%}")
    print(f"  Draw     : {probs['draw']:.1%}")
    print(f"  Away win : {probs['away_win']:.1%}")
