import numpy as np
import pandas as pd
import math

from datetime import datetime
import logging
from collections import defaultdict
from atp_forecaster.data.clean import get_cleaned_atp_matches

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_data():
    return get_cleaned_atp_matches()

def emwa(
    df: pd.DataFrame,
    feature_computers: dict,
    half_life: int = 25,
    break_gap: int = 20,
    max_dt: int = 3,
) -> pd.DataFrame:
    """
    Compute EWMA features for each player and return a new dataframe.
    """
    if not feature_computers:
        raise ValueError("Must provide at least one feature computer.")

    lambda_ = math.log(2) / half_life

    features_df = df[["order", "tourney_date", "id_a", "id_b"]].copy()

    # EWMA feature columns
    for feature in feature_computers:
        features_df[f"{feature}_a"] = 0.0
        features_df[f"{feature}_b"] = 0.0

    # player_id -> feature -> (value, last_date)
    recent = defaultdict(lambda: defaultdict(lambda: (0.0, datetime.min)))

    total_rows = len(df)
    if total_rows == 0:
        return features_df

    # Progress checkpoints at every 5%
    checkpoints = {int(total_rows * p / 20) for p in range(1, 20)}

    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        today_date = datetime.strptime(str(row["tourney_date"]), "%Y%m%d")

        for side in ["a", "b"]:
            player_id = row[f"id_{side}"]
            last_modified_date = None
            days = None

            for feature, compute_func in feature_computers.items():
                current_value, last_date = recent[player_id][feature]
                features_df.loc[idx, f"{feature}_{side}"] = current_value

                if last_modified_date is None:
                    last_modified_date = last_date
                    days = (today_date - last_modified_date).days
                    dt = min(max_dt, max(1.0, (days - break_gap) / 10.0))
                    decay = math.exp(-lambda_ * dt)

                game_value = compute_func(row, side)

                if last_date == datetime.min:
                    new_value = game_value
                else:
                    new_value = (1 - decay) * game_value + decay * current_value

                recent[player_id][feature] = (new_value, today_date)

        if i in checkpoints:
            logger.info(f"EMWA progress: {math.ceil(i / total_rows * 100)}% ({i}/{total_rows})")

    return features_df

def _opp(side):
    return "b" if side == "a" else "a"

def add_emwa_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_computers = {
        # Serve performance
        "p_ace": lambda row, side: getattr(row, f"ace_{side}") / (getattr(row, f"svpt_{side}") + 1e-4),
        
        "p_df": lambda row, side: getattr(row, f"df_{side}") / (getattr(row, f"svpt_{side}") + 1e-4),

        # Conditional win percentages
        "p_1stIn": lambda row, side: getattr(row, f"1stIn_{side}") / (getattr(row, f"svpt_{side}") + 1e-4),

        "p_1stWon": lambda row, side: getattr(row, f"1stWon_{side}") / (getattr(row, f"1stIn_{side}") + 1e-4),

        "p_2ndWon": lambda row, side: getattr(row, f"2ndWon_{side}") / 
            ((getattr(row, f"svpt_{side}") - getattr(row, f"1stIn_{side}")) + 1e-4
        ),

        "p_2ndWon_inPlay": lambda row, side: (
            row[f"2ndWon_{side}"]
            / (row[f"svpt_{side}"] - row[f"1stIn_{side}"] - row[f"df_{side}"] + 1e-4)
        ),

        # Break point defense
        "p_bpSaved": lambda row, side: getattr(row, f"bpSaved_{side}") / (getattr(row, f"bpFaced_{side}") + 1e-4),

        # Return performance
        "p_rpw": lambda row, side: (
            getattr(row, f"svpt_{_opp(side)}") 
            - getattr(row, f"1stWon_{_opp(side)}") 
            - getattr(row, f"2ndWon_{_opp(side)}")
        ) / (getattr(row, f"svpt_{_opp(side)}") + 1e-4),

        # Return: opponent aces per serve point
        "p_retAceAgainst": lambda row, side: getattr(row, f"ace_{_opp(side)}") / (getattr(row, f"svpt_{_opp(side)}") + 1e-4),

        # Return: points won on opponent 1st serve
        "p_ret1stWon": lambda row, side: (
            getattr(row, f"1stIn_{_opp(side)}") - getattr(row, f"1stWon_{_opp(side)}")
        ) / (getattr(row, f"1stIn_{_opp(side)}") + 1e-4),

        # Return: points won on opponent 2nd serve
        "p_ret2ndWon": lambda row, side: (
            row[f"svpt_{_opp(side)}"]
            - row[f"1stIn_{_opp(side)}"]
            - row[f"2ndWon_{_opp(side)}"]
        ) / (row[f"svpt_{_opp(side)}"] - row[f"1stIn_{_opp(side)}"] + 1e-4),

        # Return: points won on opponent 2nd serve excluding opp df
        "p_ret2ndWon_inPlay": lambda row, side: (
            (row[f"svpt_{_opp(side)}"] 
            - row[f"1stIn_{_opp(side)}"] 
            - row[f"df_{_opp(side)}"]) 
            - row[f"2ndWon_{_opp(side)}"]
        ) / (
            (row[f"svpt_{_opp(side)}"] 
            - row[f"1stIn_{_opp(side)}"] 
            - row[f"df_{_opp(side)}"]) + 1e-4
        ),

        # Break point conversion
        "p_bpConv": lambda row, side: (
            getattr(row, f"bpFaced_{_opp(side)}") - getattr(row, f"bpSaved_{_opp(side)}")
        ) / (getattr(row, f"bpFaced_{_opp(side)}") + 1e-4),

        # Total points won
        "p_totalPtsWon": lambda row, side: (
            row[f"1stWon_{side}"] 
            + row[f"2ndWon_{side}"]
            + (
                row[f"svpt_{_opp(side)}"]
                - row[f"1stWon_{_opp(side)}"]
                - row[f"2ndWon_{_opp(side)}"]
            )
        ) / (row[f"svpt_{side}"] + row[f"svpt_{_opp(side)}"] + 1e-4),


        # Dominance ratio: return% / serve lost%, capped at 3 for when serve lost% = 0
        "dominance_ratio": lambda row, side: min(3, 
        (
            (
                getattr(row, f"svpt_{_opp(side)}")
                - getattr(row, f"1stWon_{_opp(side)}")
                - getattr(row, f"2ndWon_{_opp(side)}")
            ) / (getattr(row, f"svpt_{_opp(side)}") + 1e-4)
        ) / (
            1 - (
                getattr(row, f"1stWon_{side}") + getattr(row, f"2ndWon_{side}")
            ) / (getattr(row, f"svpt_{side}") + 1e-4)
        )),
    }

    # return dataframe with specified features
    return emwa(df, feature_computers)


def main():
    logger.info("Starting player performance calculation... (this may take a while)")
    df = get_data()
    df_performance = add_emwa_features(df)
    df_performance.to_parquet("./data/features/base/player_performance.parquet", index=True)
    logger.info(f"Player performance calculation completed! Saved {len(df_performance)} rows to ./data/features/base/player_performance.parquet")

if __name__ == "__main__":
    main()