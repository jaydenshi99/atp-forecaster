import numpy as np
import pandas as pd
import os
from pathlib import Path

from atp_forecaster.data.full.build_dataset_v1 import one_hot_encode
from atp_forecaster.data.full.build_dataset_v1 import build_matchup_features


def find_project_root() -> Path:
    """Find repo root by walking up until data/raw/tennis-data exists."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data" / "raw" / "tennis-data").exists():
            return parent
    # Fallback to previous assumption (5 levels up) if not found
    return Path(__file__).resolve().parents[5]

def get_dataframes():
    start_year = 2001
    end_year = 2024
    project_root = find_project_root()
    odds_dir = project_root / "data" / "raw" / "tennis-data"
    files = []

    for year in range(start_year, end_year + 1):
        xls_path = odds_dir / f"{year}.xls"
        xlsx_path = odds_dir / f"{year}.xlsx"

        if xls_path.exists():
            files.append(xls_path)
        elif xlsx_path.exists():
            files.append(xlsx_path)
        else:
            print(f"Warning: no file found for year {year} ({xls_path} or {xlsx_path})")

    # Initialize an empty list to store the DataFrames
    dfs = []

    # Loop through the files, read each one, and append to the list
    for file in files:
        df = pd.read_excel(file)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No odds files found in {odds_dir}. Expected files like 2001.xls/.xlsx"
        )

    # Concatenate all DataFrames in the list vertically (stacking rows)
    odds_df = pd.concat(dfs, axis=0, ignore_index=True)
    sackman_path = project_root / "data" / "features" / "feature_sets" / "dataset_v1_combined.parquet"
    sackman_df = pd.read_parquet(sackman_path)

    # fix error in EXW
    odds_df["EXW"] = (
        odds_df["EXW"]
        .replace({",": ""}, regex=True)
        .replace("", np.nan)
        .astype(float)
    )

    return odds_df, sackman_df

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def old_to_new_name(name):
    """
    Convert 'Firstname Lastname' -> 'Lastname F.'
    """
    if pd.isna(name):
        return np.nan
    parts = name.split()
    if len(parts) < 2:
        return name
    first, last = parts[0], " ".join(parts[1:])
    initial = first[0].upper()
    return f"{last} {initial}."

def new_to_old_name(name, old_name_lookup):
    """
    Convert 'Lastname F.' -> approximate 'Firstname Lastname' via lookup table.
    If not found, returns original.
    """
    return old_name_lookup.get(name, name)

def monday_round(date_obj):
    """
    Round a date down to the most recent Monday.
    """
    if isinstance(date_obj, str):
        date_obj = pd.to_datetime(date_obj)
    return date_obj - timedelta(days=date_obj.weekday())

def yyyymmdd_to_monday(yyyymmdd):
    """
    Convert YYYYMMDD integer into datetime, then round to Monday.
    """
    if pd.isna(yyyymmdd):
        return np.nan
    s = str(int(yyyymmdd))
    dt = datetime.strptime(s, "%Y%m%d")
    return monday_round(dt)

def merge_old_new(old_df, new_df):
    df_old = old_df.copy()
    df_new = new_df.copy()

    df_old["name_a_new"] = df_old["name_a"].apply(old_to_new_name)
    df_old["name_b_new"] = df_old["name_b"].apply(old_to_new_name)

    # find winner and loser names
    df_old["old_winner"] = np.where(df_old["result"] == 1,
                                df_old["name_a_new"],
                                df_old["name_b_new"])

    df_old["old_loser"] = np.where(df_old["result"] == 1,
                                df_old["name_b_new"],
                                df_old["name_a_new"])

    # For new_df, Winner/Loser are already in Lastname F. format → keep them
    df_new.rename(columns={"Winner": "winner_new",
                           "Loser": "loser_new"}, inplace=True)

    # old_df tourney_date → Monday
    df_old["merge_date"] = df_old["tourney_date"].apply(yyyymmdd_to_monday)

    # new_df Date (YYYY-MM-DD) → round to Monday
    df_new["merge_date"] = df_new["Date"].apply(lambda d: monday_round(pd.to_datetime(d)))

    df_old["surface_norm"] = df_old["surface"].str.lower().str.strip()
    df_old["best_of_norm"] = df_old["best_of"].astype("Int64")

    df_new["surface_norm"] = df_new["Surface"].str.lower().str.strip()
    df_new["best_of_norm"] = df_new["Best of"].astype("Int64")

    merge_cols_old = ["old_winner", "old_loser", "merge_date",
                      "surface_norm", "best_of_norm"]

    merge_cols_new = ["winner_new", "loser_new", "merge_date",
                      "surface_norm", "best_of_norm"]

    merged = df_old.merge(
        df_new,
        left_on=merge_cols_old,
        right_on=merge_cols_new,
        how="left",
        suffixes=("_old", "_new")
    )

    key_cols = ["id_a", "id_b", "merge_date"]
    dup_mask = merged.duplicated(subset=key_cols, keep=False)

    # keep non duplicated rows
    merged = merged[~dup_mask].copy()

    # remove rows with no odds
    odds_cols = ['B365W', 'B365L', "PSW", "PSL"] # primary odds columns
    merged = merged.dropna(subset=odds_cols, how="all")

    return merged

def drop_columns(df):
    identifier_cols = ["name_a","name_b","id_a","id_b","score","minutes"]
    identifier_df = df[identifier_cols]

    columns_to_drop = [
        # merge helper
        "name_a_new","name_b_new","old_winner","old_loser",
        "merge_date","surface_norm","best_of_norm",

        # odds metadata
        "ATP","Location","Tournament","Date","Series","Court","Surface",
        "Round","Best of","winner_new","loser_new","WRank","LRank",
        "W1","L1","W2","L2","W3","L3","W4","L4","W5","L5","Wsets","Lsets","Comment",

        # sackmann identifiers
        "name_a","name_b","id_a","id_b","score","minutes",

        # raw stats A
        "ace_a","df_a","svpt_a","1stIn_a","1stWon_a","2ndWon_a","SvGms_a","bpSaved_a","bpFaced_a",

        # raw stats B
        "ace_b","df_b","svpt_b","1stIn_b","1stWon_b","2ndWon_b","SvGms_b","bpSaved_b","bpFaced_b",
    ]

    df = df.drop(columns=columns_to_drop, errors='ignore')

    return df, identifier_df
    
def process_columns(df):
    df = one_hot_encode(df)
    df = build_matchup_features(df)

    df = df[[col for col in df.columns if col != 'result'] + ['result']]

    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df

def main():
    odds_df, sackman_df = get_dataframes()
    df = merge_old_new(sackman_df, odds_df)

    og_rows = odds_df.shape[0]
    merged_rows = df.shape[0]

    print("rows in sackman_df", og_rows)
    print("rows in merged_df", merged_rows)
    print("percent lost", (og_rows - merged_rows) / og_rows)

    df, identifier_df = drop_columns(df)
    df = process_columns(df)

    # create missing rows from one hot encoding
    df['hand_a_A'] = 0
    df['hand_b_A'] = 0
    df['round_BR'] = 0
    df['round_ER'] = 0

    # add back in identifiers
    df = pd.concat([identifier_df, df], axis=1)

    print(df.isnull().sum().to_string())

    project_root = find_project_root()
    backtest_dir = project_root / "data" / "backtest"
    df.to_parquet(backtest_dir / "backtest_v1.parquet")

if __name__ == "__main__":
    main()