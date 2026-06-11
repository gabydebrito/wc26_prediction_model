import random
import numpy as np
import matplotlib.pyplot as plt
from poisson_model import predict_scoreline_probs

HOSTS = {'United States', 'Canada', 'Mexico'}
GROUPS = {'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
          'B': ['Canada', 'Switzerland', 'Qatar', 'Bosnia and Herzegovina'],
          'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
          'D': ['United States', 'Turkey', 'Australia', 'Paraguay'],
          'E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'],
          'F': ['Netherlands', 'Sweden', 'Japan', 'Tunisia'],
          'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
          'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
          'I': ['France', 'Senegal', 'Iraq', 'Norway'],
          'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
          'K': ['Portugal', 'Uzbekistan', 'DR Congo', 'Colombia'],
          'L': ['England', 'Croatia', 'Ghana', 'Panama']}

BRACKET_R32 = [
    ('W_E',  '3rd_ABCDF'),
    ('W_I',  '3rd_CDFGH'),
    ('R_A',  'R_B'),
    ('W_F',  'R_C'),
    ('R_K',  'R_L'),
    ('W_H',  'R_J'),
    ('W_D',  '3rd_BEFIJ'),
    ('W_G',  '3rd_AEHIJ'),
    ('W_C',  'R_F'),
    ('R_E',  'R_I'),
    ('W_A',  '3rd_CEFHI'),
    ('W_L',  '3rd_EHIJK'),
    ('W_J',  'R_H'),
    ('R_D',  'R_G'),
    ('W_B',  '3rd_EFGIJ'),
    ('W_K',  '3rd_DEIJL'),
]

def is_neutral(home: str, away: str) -> bool:
    return home not in HOSTS and away not in HOSTS

def simulate_match(home:str, away:str, params:dict, neutral: bool=True) -> tuple[int, int]:
    """
    Simulates match between home and away teams using Poission model parameters
    Returns 'home', 'draw', or 'away' depending on the outcome of the match
    If neutral is True, then the home advantage is removed from the model

    """
    score_mtx = predict_scoreline_probs(home, away, params, neutral=neutral)
    probs = score_mtx.values.ravel()
    probs = probs / probs.sum()  # normalize to guard against floating point drift

    idx = np.random.choice(len(probs), p=probs)
    home_goals = idx // score_mtx.shape[1]
    away_goals = idx % score_mtx.shape[1]

    return home_goals, away_goals

def simulate_knockout_match(home:str, away:str, params:dict) -> str:
    """
    Simulates a knockout match between home and away teams using Poission model parameters
    Returns 'home' or 'away' depending on the outcome of the match
    If the match is a draw, rerolls until there is a winner

    """
    while True:
        home_goals, away_goals = simulate_match(home, away, params, neutral=True)
        if home_goals > away_goals:
            return 'home', home_goals, away_goals
        elif away_goals > home_goals:
            return 'away', home_goals, away_goals

def simulate_group_stage(teams: list[str], params: dict) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """
    Simulates a group stage where each team plays each other once
    Returns a dictionary mapping each team to their points in the group stage
    """
    points = {team: 0 for team in teams}
    goals_for = {team:0 for team in teams}
    goals_against = {team:0 for team in teams}
    match_log = []

    for i in range(len(teams)):
        for j in range(i+1, len(teams)):
            home = teams[i]
            away = teams[j]
            outcome = simulate_match(home, away, params, neutral=is_neutral(home, away))
            home_score, away_score = outcome
            match_log.append((home, away, home_score, away_score))

            goals_for[home] += home_score
            goals_against[home] += away_score
            goals_for[away] += away_score
            goals_against[away] += home_score


            if home_score > away_score:
                points[home] += 3
            elif away_score > home_score:
                points[away] += 3
            else:
                points[home] += 1
                points[away] += 1

    return points, goals_for, goals_against, match_log

def get_qualifiers(group_results: dict) -> list[str]:
    """
    Given results of group stage, returns the teams that qualify for the knockout stage.
    This includes top two from each group, as well as eight best third place teams
    across the groups.
    if there are ties in points, then the tiebreakers are applied in the following order:
        1. Goal difference
        2. Goals scored
        3. Random draw (coin flip)
    """
    group_winners = {}
    group_runners = {}
    third_place_teams = []

    points_lookup = {}
    goals_for_lookup = {}
    goals_against_lookup = {}

    for group, (points, goals_for, goals_against, _) in group_results.items():
        points_lookup.update(points)
        goals_for_lookup.update(goals_for)
        goals_against_lookup.update(goals_against)

    for group, (points, goals_for, goals_against, _) in group_results.items():
        teams = sorted(points.keys(), key=lambda t: (
            points[t],
            goals_for[t] - goals_against[t],
            goals_for[t]),
            reverse=True)
        group_winners[group] = teams[0]
        group_runners[group] = teams[1]
        third_place_teams.append((teams[2], group))

    # rank third place teams and take best 8
    third_place_teams.sort(key=lambda x: (
        points_lookup[x[0]],
        goals_for_lookup[x[0]] - goals_against_lookup[x[0]],
        goals_for_lookup[x[0]]
    ), reverse=True)

    return group_winners, group_runners, third_place_teams[:8]

def resolve_slot(slot: str, group_winners: dict,
                 group_runners: dict, third_place_ranked: list) -> str:
    if slot.startswith('W_'):
        return group_winners[slot[2:]]
    elif slot.startswith('R_'):
        return group_runners[slot[2:]]
    elif slot.startswith('3rd_'):
        team, group = third_place_ranked.pop(0)
        return team

def simulate_knockout_stage(group_results: dict, params: dict) -> dict:
    """
    Simulates the knockout stage of the tournament given the qualifiers from the group stage
    Returns the winner of the tournament
    """
    group_winners, group_runners, third_place_ranked = get_qualifiers(group_results)
    third_place_copy = list(third_place_ranked)
    random.shuffle(third_place_copy)  # shuffle to ensure random tiebreaker for third place teams

    results = {}
    knockout_log = {}

    #Marks all group stage exits
    all_teams = set(team for group, (points, _, _, _) in group_results.items() for team in points)
    qualifiers = set(group_winners.values()) | set(group_runners.values()) | set(t for t, _ in third_place_ranked)
    for team in all_teams - qualifiers:
        results[team] = 'Group Stage'

    current_round = []
    for home_slot, away_slot in BRACKET_R32:
        current_round.append(resolve_slot(home_slot, group_winners, group_runners, third_place_copy))
        current_round.append(resolve_slot(away_slot, group_winners, group_runners, third_place_copy))

    round_names = ['Round of 32', 'Round of 16', 'Quarter-Final', 'Semi-Final', 'Final']

    for round_name in round_names:
        next_round = []
        round_matches = []
        for i in range(0, len(current_round), 2):
            home = current_round[i]
            away = current_round[i+1]
            result, home_goals, away_goals = simulate_knockout_match(home, away, params)
            winner = home if result == 'home' else away
            loser = away if result == 'home' else home

            round_matches.append((home, away, home_goals, away_goals, winner))
            results[loser] = round_name
            next_round.append(winner)

        knockout_log[round_name] = round_matches
        current_round = next_round

    results[current_round[0]] = 'Winner'
    return results, knockout_log


def simulate_tournament(groups: dict[str, list[str]], params: dict) -> str:
    """
    Simulates an entire tournament given the group stage teams and model parameters
    Returns the winner of the tournament
    """
    group_results = {group: simulate_group_stage(teams, params) for group, teams in groups.items()}
    results, knockout_log = simulate_knockout_stage(group_results, params)
    return results, group_results, knockout_log
