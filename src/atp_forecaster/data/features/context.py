import pandas as pd
import logging

from collections import defaultdict
from collections import deque

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_data():
    df1 = pd.read_parquet("./data/cleaned/atp_matches_cleaned.parquet")
    df2 = pd.read_parquet("./data/features/base/glicko2_ratings.parquet")

    # removes overlapping columns
    overlap_cols = df1.columns.intersection(df2.columns)
    if len(overlap_cols) > 0:
        df2 = df2.drop(columns=list(overlap_cols))

    df = df1.join(df2, how="left")
    return df

def head_to_head(df, k=10):
    hth_df = df[['tourney_date', 'id_a', 'name_a', 'id_b', 'name_b']].copy()

    hth_df['hth_win_p_a'] = 0.0
    hth_df['hth_matches'] = 0
    
    # stores 1 if player in the the first key wins
    matchups = defaultdict(list)

    def add_matchups(id_a, id_b, result):
        # ensure id_a is always the smaller id
        ids = sorted((id_a, id_b))

        # always store the result from POV of first entry in the key
        if ids[0] == id_b:
            value = 1 - result
        else:
            value = result

        matchups[tuple(ids)].append(value)

    '''returns win percentage of id_a, and number of matches processed up to k games back'''
    def get_matchup_stats(id_a, id_b, k=10):
        ids = sorted((id_a, id_b))
        results = matchups[tuple(ids)]

        if len(results) == 0:
            return 0, 0
        
        recent_results = results[-k:] 
        matches = len(recent_results)
        win_percentage = 0
        for _, r in enumerate(recent_results):
            win_percentage += r

        win_percentage /= matches

        # we want to return the win percentage of id_a
        if ids[0] == id_b:
            win_percentage = 1 - win_percentage

        return win_percentage, matches
    
    for index, row in df.iterrows():
        # update df
        win_percentage, matches = get_matchup_stats(row['id_a'], row['id_b'], k)

        hth_df.loc[index, 'hth_win_p_a'] = win_percentage
        hth_df.loc[index, 'hth_matches'] = matches

        # update matchups
        add_matchups(row['id_a'], row['id_b'], row['result'])

    return hth_df

def experience(df):
    exp_df = df[['tourney_date', 'id_a', 'name_a', 'id_b', 'name_b']].copy()

    exp_df['total_matches_a'] = 0
    exp_df['total_matches_b'] = 0
    exp_df['total_surface_matches_a'] = 0
    exp_df['total_surface_matches_b'] = 0
    
    # stores 1 if player in the the first key wins
    matchups = defaultdict(int)

    def add_match(id_, surface):
        matchups[(id_, surface)] += 1

    def get_matches(id_, surface):
        return matchups[(id_, surface)]
    
    for index, row in df.iterrows():
        # update df
        matches_a = get_matches(row['id_a'], 'All')
        matches_b = get_matches(row['id_b'], 'All')

        matches_a_surf = get_matches(row['id_a'], row['surface'])
        matches_b_surf = get_matches(row['id_b'], row['surface'])

        exp_df.loc[index, 'total_matches_a'] = matches_a
        exp_df.loc[index, 'total_matches_b'] = matches_b
        exp_df.loc[index, 'total_surface_matches_a'] = matches_a_surf
        exp_df.loc[index, 'total_surface_matches_b'] = matches_b_surf

        # update matchups
        add_match(row['id_a'], 'All')
        add_match(row['id_b'], 'All')
        add_match(row['id_a'], row['surface'])
        add_match(row['id_b'], row['surface'])

    return exp_df

def momentum_features(df, fast_span=5, slow_span=20):
    """
    Compute per-match momentum features for each player:
      - form_delta: actual win rate - Elo-expected win rate over last 5 matches
      - elo_momentum: fast EWMA(elo) - slow EWMA(elo)

    Assumes df has columns:
      - tourney_date
      - id_a, name_a, id_b, name_b
      - elo_a, elo_b          (pre-match Elo ratings)
      - result                (1 if player A wins, 0 if player B wins)
    """
    mom_df = df[['tourney_date', 'id_a', 'name_a', 'id_b', 'name_b']].copy()

    mom_df['form_delta_a'] = 0.0
    mom_df['form_delta_b'] = 0.0
    mom_df['elo_momentum_a'] = 0.0
    mom_df['elo_momentum_b'] = 0.0

    # per-player: last 5 actual results and expected win probs
    recent_results = defaultdict(lambda: deque(maxlen=5))
    recent_expected = defaultdict(lambda: deque(maxlen=5))

    def get_form_delta(player_id):
        res_hist = recent_results[player_id]
        exp_hist = recent_expected[player_id]
        if not res_hist:
            return 0.0
        actual = sum(res_hist) / len(res_hist)
        expected = sum(exp_hist) / len(exp_hist)
        return actual - expected

    def update_form(player_id, actual, expected):
        recent_results[player_id].append(actual)
        recent_expected[player_id].append(expected)

    alpha_fast = 2 / (fast_span + 1)
    alpha_slow = 2 / (slow_span + 1)

    fast_elo = {}  # player_id -> fast EWMA
    slow_elo = {}  # player_id -> slow EWMA

    def get_elo_momentum(player_id):
        if player_id not in fast_elo:
            return 0.0
        return fast_elo[player_id] - slow_elo[player_id]

    def update_elo(player_id, current_elo):
        if player_id not in fast_elo:
            fast_elo[player_id] = current_elo
            slow_elo[player_id] = current_elo
        else:
            fast_elo[player_id] = (
                alpha_fast * current_elo + (1 - alpha_fast) * fast_elo[player_id]
            )
            slow_elo[player_id] = (
                alpha_slow * current_elo + (1 - alpha_slow) * slow_elo[player_id]
            )

    for index, row in df.iterrows():
        id_a = row['id_a']
        id_b = row['id_b']

        Ea = row['elo_a']
        Eb = row['elo_b']

        # elo based win probabilities for this match (before result)
        elo_prob_a = 1.0 / (1.0 + 10.0 ** ((Eb - Ea) / 400.0))
        elo_prob_b = 1.0 - elo_prob_a

        mom_df.loc[index, 'form_delta_a'] = get_form_delta(id_a)
        mom_df.loc[index, 'form_delta_b'] = get_form_delta(id_b)

        mom_df.loc[index, 'elo_momentum_a'] = get_elo_momentum(id_a)
        mom_df.loc[index, 'elo_momentum_b'] = get_elo_momentum(id_b)

        actual_a = row['result']        
        actual_b = 1 - actual_a

        # update form state
        update_form(id_a, actual_a, elo_prob_a)
        update_form(id_b, actual_b, elo_prob_b)

        # update momentum state (using pre-match Elo)
        update_elo(id_a, Ea)
        update_elo(id_b, Eb)

    return mom_df

def fatigue_features(df, window_days=14):
    """
    Compute per-match fatigue features:
      - recent_matches: matches played in the last `window_days` days
      - recent_minutes: minutes played in the last `window_days` days
    """

    df_iter = df.copy()
    df_iter['tourney_date'] = pd.to_datetime(df_iter['tourney_date'].astype(str), format='%Y%m%d')

    fat_df = df_iter[['tourney_date', 'id_a', 'name_a', 'id_b', 'name_b']].copy()

    fat_df['recent_matches_a'] = 0
    fat_df['recent_matches_b'] = 0
    fat_df['recent_minutes_a'] = 0.0
    fat_df['recent_minutes_b'] = 0.0

    history = defaultdict(lambda: deque())

    def get_fatigue(player_id, current_date):
        dq = history[player_id]
        if not dq:
            return 0, 0.0

        cutoff = current_date - pd.Timedelta(days=window_days)

        # Remove entries older than the cutoff date
        while dq and dq[0][0] < cutoff:
            dq.popleft()

        matches = len(dq)
        minutes = sum(m for _, m in dq)
        return matches, minutes

    def update_history(player_id, date, minutes):
        history[player_id].append((date, minutes))

    for index, row in df_iter.iterrows():
        date = row['tourney_date']
        mins = row['minutes']

        id_a = row['id_a']
        id_b = row['id_b']

        matches_a, minutes_a = get_fatigue(id_a, date)
        matches_b, minutes_b = get_fatigue(id_b, date)

        fat_df.loc[index, 'recent_matches_a'] = matches_a
        fat_df.loc[index, 'recent_matches_b'] = matches_b
        fat_df.loc[index, 'recent_minutes_a'] = minutes_a
        fat_df.loc[index, 'recent_minutes_b'] = minutes_b

        update_history(id_a, date, mins)
        update_history(id_b, date, mins)

    return fat_df

def main():
    logger.info("Starting context feature calculation...")
    df = get_data()

    hth_df = head_to_head(df)
    hth_df.to_parquet("./data/features/base/head_to_head.parquet", index=True)
    logger.info(f"Head to head feature calculation completed! Saved {len(hth_df)} rows to ./data/features/base/head_to_head.parquet")

    exp_df = experience(df)
    exp_df.to_parquet("./data/features/base/experience.parquet", index=True)
    logger.info(f"Experience feature calculation completed! Saved {len(exp_df)} rows to ./data/features/base/experience.parquet")


    mom_df = momentum_features(df)
    mom_df.to_parquet("./data/features/base/momentum.parquet", index=True)
    logger.info(f"Momentum feature calculation completed! Saved {len(mom_df)} rows to ./data/features/base/momentum.parquet")

    fat_df = fatigue_features(df)
    fat_df.to_parquet("./data/features/base/fatigue.parquet", index=True)
    logger.info(f"Fatigue feature calculation completed! Saved {len(fat_df)} rows to ./data/features/base/fatigue.parquet")

    logger.info("Context feature calculation completed!")

if __name__ == "__main__":
    main()