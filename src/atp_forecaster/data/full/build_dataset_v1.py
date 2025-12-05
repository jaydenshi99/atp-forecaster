import numpy as np
import pandas as pd


def load_base_data() -> pd.DataFrame:
    """Load and join all base feature tables into a single dataframe."""
    df_original = pd.read_parquet("./data/cleaned/atp_matches_cleaned.parquet")
    df_skills = pd.read_parquet("./data/features/base/player_performance.parquet")
    df_elo = pd.read_parquet("./data/features/base/glicko2_ratings.parquet")
    df_exp = pd.read_parquet("./data/features/base/experience.parquet")
    df_fatigue = pd.read_parquet("./data/features/base/fatigue.parquet")
    df_hth = pd.read_parquet("./data/features/base/head_to_head.parquet")
    df_mom = pd.read_parquet("./data/features/base/momentum.parquet")

    dfs = [df_original, df_skills, df_elo, df_exp, df_fatigue, df_hth, df_mom]
    df_full = pd.concat(dfs, axis=1)
    df_full = df_full.loc[:, ~df_full.columns.duplicated()]

    # Save the raw joined feature set
    df_full.to_parquet(
        "./data/features/feature_sets/dataset_v1_combined.parquet", index=True
    )
    return df_full


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are not needed for modelling."""
    cols_to_drop = [
        "name_a",
        "name_b",
        "id_a",
        "id_b",
        "score",
        "minutes",
        "ace_a",
        "df_a",
        "svpt_a",
        "1stIn_a",
        "1stWon_a",
        "2ndWon_a",
        "SvGms_a",
        "bpSaved_a",
        "bpFaced_a",
        "ace_b",
        "df_b",
        "svpt_b",
        "1stIn_b",
        "1stWon_b",
        "2ndWon_b",
        "SvGms_b",
        "bpSaved_b",
        "bpFaced_b",
    ]
    return df.drop(columns=cols_to_drop, errors="ignore")


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    df = df.copy()
    df = pd.get_dummies(df, drop_first=False)
    return df


def build_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create matchup-level features (A vs B differences, log-differences, etc.)."""
    df = df.copy()

    cols_to_drop: list[str] = []

    if {"ht_a", "ht_b"}.issubset(df.columns):
        df["height_diff"] = df["ht_a"] - df["ht_b"]
        cols_to_drop += ["ht_a", "ht_b"]

    if {"age_a", "age_b"}.issubset(df.columns):
        df["age_diff"] = df["age_a"] - df["age_b"]
        cols_to_drop += ["age_a", "age_b"]

    difference_prefixes = [
        "elo",
        "elo_surface",
        "p_ace",
        "p_df",
        "p_1stIn",
        "p_1stWon",
        "p_2ndWon",
        "p_2ndWon_inPlay",
        "p_bpSaved",
        "p_rpw",
        "p_retAceAgainst",
        "p_ret1stWon",
        "p_ret2ndWon",
        "p_ret2ndWon_inPlay",
        "p_bpConv",
        "p_totalPtsWon",
        "dominance_ratio",
        "age",
        "ht",
        "form_delta",
        "elo_momentum",
        "recent_minutes",
    ]

    log_difference_prefixes = [
        "rank_points",
        "total_matches",
        "total_surface_matches",
        "recent_matches",
    ]

    for prefix in difference_prefixes:
        col_a = f"{prefix}_a"
        col_b = f"{prefix}_b"
        if col_a in df.columns and col_b in df.columns:
            df[f"{prefix}_diff"] = df[col_a] - df[col_b]
            cols_to_drop += [col_a, col_b]

    for prefix in log_difference_prefixes:
        col_a = f"{prefix}_a"
        col_b = f"{prefix}_b"
        if col_a in df.columns and col_b in df.columns:
            df[f"log_{prefix}_diff"] = np.log1p(df[col_a]) - np.log1p(df[col_b])
            cols_to_drop += [col_a, col_b]

    # rank: use inverse rank difference; handle zeros as NaN
    if {"rank_a", "rank_b"}.issubset(df.columns):
        rank_a = df["rank_a"].replace(0, np.nan)
        rank_b = df["rank_b"].replace(0, np.nan)
        df["inv_rank_diff"] = (1.0 / rank_a) - (1.0 / rank_b)
        cols_to_drop += ["rank_a", "rank_b"]

    # hth_matches: use log_hth_matches
    if {"hth_matches"}.issubset(df.columns):
        hth_matches = df["hth_matches"].replace(0, np.nan)
        df["log_hth_matches"] = np.log1p(hth_matches)
        cols_to_drop += ["hth_matches"]

    # drop originals
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df


def build_dataset() -> pd.DataFrame:
    """Full pipeline to build the training dataset and save it to disk."""
    df_full = load_base_data()
    df_full = drop_unused_columns(df_full)
    df_full = one_hot_encode(df_full)

    df_balanced = build_matchup_features(df_full)

    # Put target column 'result' at the end if present
    if "result" in df_balanced.columns:
        df_balanced = df_balanced[
            [col for col in df_balanced.columns if col != "result"] + ["result"]
        ]

    # Convert bools to ints for parquet compatibility / modelling
    bool_cols = df_balanced.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df_balanced[bool_cols] = df_balanced[bool_cols].astype(int)

    df_balanced.to_parquet("./data/training_data/dataset_v1.parquet", index=True)
    return df_balanced


def main() -> None:
    """Entry point for building dataset_v1."""
    build_dataset()
    


if __name__ == "__main__":
    main()