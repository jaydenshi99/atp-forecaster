import logging
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_data(start_year=1980, end_year=2024):
    """ read raw data from csv files """
    logger.info(f"Loading ATP match data from {start_year} to {end_year}")
    # List of CSV file paths (example: if files are in the 'data' folder)
    files = []

    for year in range(start_year, end_year + 1):
        files.append('./data/raw/tennis_atp-master/atp_matches_' + str(year) + '.csv')

    # Initialize an empty list to store the DataFrames
    dfs = []

    # Loop through the files, read each one, and append to the list
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)

    # Concatenate all DataFrames in the list vertically (stacking rows)
    df = pd.concat(dfs, axis=0, ignore_index=True)
    logger.info(f"Loaded {len(df)} total rows from {len(files)} files")

    return df

def remove_null_values(df):
    """ remove null values from the dataframe """
    df_res = df.copy()

    # Remove null values
    null_counts = df_res.isnull().sum()
    logger.info(f"Null value counts:\n{null_counts}")
    logger.info(f"Total Rows: {len(df_res)}")

    # Not enough data
    columns_to_drop = ['winner_seed', 'winner_entry', 'loser_seed', 'loser_entry']

    # Randomised features with almost no predictive power
    columns_to_drop += [
        'tourney_id', 'tourney_name', 'match_num', 'winner_ioc', 'loser_ioc'
    ]

    for col in columns_to_drop:
        if col in df_res.columns:
                df_res.drop(col, axis=1, inplace=True)
    
    return df_res

def filter_atp_matches(df):
    """ filter out non-ATP matches """
    df_atp = df.copy()
    
    # Keep ATP matches only
    ATP_LEVELS = {"G", "M", "A", "F"}
    df_atp = df_atp[df_atp["tourney_level"].isin(ATP_LEVELS)].copy()

    df_atp.dropna(axis=0, inplace=True)
    
    return df_atp

def rename_columns(df):
    """ rename columns to make result anonymous """
    df_res = df.copy()
    
    # Rename to make result anonymous
    df_res = df_res.rename(columns={
        # === Player A (Winner) ===
        'winner_id'          : 'id_a',
        'winner_name'        : 'name_a',
        'winner_hand'        : 'hand_a',
        'winner_ht'          : 'ht_a',
        'winner_age'         : 'age_a',
        'winner_rank'        : 'rank_a',
        'winner_rank_points' : 'rank_points_a',
        
        # Winner Match Stats → Player A
        'w_ace'      : 'ace_a',
        'w_df'       : 'df_a',
        'w_svpt'     : 'svpt_a',
        'w_1stIn'    : '1stIn_a',
        'w_1stWon'   : '1stWon_a',
        'w_2ndWon'   : '2ndWon_a',
        'w_SvGms'    : 'SvGms_a',
        'w_bpSaved'  : 'bpSaved_a',
        'w_bpFaced'  : 'bpFaced_a',

        # === Player B (Loser) ===
        'loser_id'          : 'id_b',
        'loser_name'        : 'name_b',
        'loser_hand'        : 'hand_b',
        'loser_ht'          : 'ht_b',
        'loser_age'         : 'age_b',
        'loser_rank'        : 'rank_b',
        'loser_rank_points' : 'rank_points_b',
        
        # Loser Match Stats → Player B
        'l_ace'      : 'ace_b',
        'l_df'       : 'df_b',
        'l_svpt'     : 'svpt_b',
        'l_1stIn'    : '1stIn_b',
        'l_1stWon'   : '1stWon_b',
        'l_2ndWon'   : '2ndWon_b',
        'l_SvGms'    : 'SvGms_b',
        'l_bpSaved'  : 'bpSaved_b',
        'l_bpFaced'  : 'bpFaced_b',
    })

    rng = np.random.default_rng(seed=123)

    # p(swap) = 0.5
    swap_mask = rng.random(len(df_res)) < 0.5 

    a_cols = ['id_a', 'name_a', 'hand_a', 'ht_a', 'age_a', 'rank_a', 'rank_points_a']
    b_cols = ['id_b', 'name_b', 'hand_b', 'ht_b', 'age_b', 'rank_b', 'rank_points_b']

    # extract both sides
    a_values = df_res.loc[swap_mask, a_cols].values  # original A
    b_values = df_res.loc[swap_mask, b_cols].values  # original B

    # swap
    df_res.loc[swap_mask, a_cols] = b_values
    df_res.loc[swap_mask, b_cols] = a_values

    # no swap means player A is winner
    df_res['result'] = 0
    df_res.loc[~swap_mask, 'result'] = 1

    return df_res

def filter_corrupted_matches(df):
    """ filter out corrupted matches """

    # 1. Basic svpt sanity: require at least 10 serve points for BOTH players
    mask_svpt = (df["svpt_a"] >= 10) & (df["svpt_b"] >= 10)

    # 2. Corruption checks for side a
    valid_a = (
        (df["svpt_a"] > 0) &
        (df["1stIn_a"] <= df["svpt_a"]) &
        (df["1stWon_a"] <= df["1stIn_a"]) &
        (df["2ndWon_a"] <= (df["svpt_a"] - df["1stIn_a"]))
    )

    # 3. Corruption checks for side b
    valid_b = (
        (df["svpt_b"] > 0) &
        (df["1stIn_b"] <= df["svpt_b"]) &
        (df["1stWon_b"] <= df["1stIn_b"]) &
        (df["2ndWon_b"] <= (df["svpt_b"] - df["1stIn_b"]))
    )

    # 4. Combine all conditions
    clean_mask = mask_svpt & valid_a & valid_b

    dropped = len(df) - clean_mask.sum()
    logger.info(f"Dropping {dropped} corrupted/short matches")

    return df[clean_mask].reset_index(drop=True)

def main():
    logger.info("Starting data cleaning pipeline...")
    df = get_data()
    df = remove_null_values(df)
    df = filter_atp_matches(df)
    df = rename_columns(df)
    df = filter_corrupted_matches(df)
    df.to_csv('./data/cleaned/atp_matches_cleaned.csv', index=False)
    logger.info(f"Data cleaning completed! Saved {len(df)} rows to ./data/cleaned/atp_matches_cleaned.csv")

if __name__ == "__main__":
    main()