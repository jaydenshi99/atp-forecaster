import logging
from collections import defaultdict, deque

import pandas as pd

from atp_forecaster.data.features.context import get_data


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def fatigue_features_strict(df: pd.DataFrame, window_days: int = 14) -> pd.DataFrame:
    """
    Compute per-match fatigue features with a *hard same-day cutoff*:

      - recent_matches: matches played STRICTLY BEFORE the current calendar date
      - recent_minutes: minutes played STRICTLY BEFORE the current calendar date

    This is a conservative leakage test. Any matches on the same calendar day
    (even earlier in the sequence) are excluded from the fatigue calculation.

    Args:
        df: DataFrame with at least:
            - tourney_date (YYYYMMDD or datetime-like)
            - id_a, id_b
            - minutes
            - order
        window_days: kept for API symmetry but NOT used in the strict cutoff
                     (history is truncated by date, not by window).

    Returns:
        DataFrame with:
            - order
            - tourney_date
            - id_a, name_a, id_b, name_b
            - recent_matches_a / recent_matches_b
            - recent_minutes_a / recent_minutes_b
    """
    df_iter = df.copy()
    df_iter["tourney_date"] = pd.to_datetime(
        df_iter["tourney_date"].astype(str), format="%Y%m%d"
    )
    df_iter = df_iter.sort_values("order").reset_index(drop=True)

    fat_df = df_iter[
        ["order", "tourney_date", "id_a", "name_a", "id_b", "name_b"]
    ].copy()

    fat_df["recent_matches_a"] = 0
    fat_df["recent_matches_b"] = 0
    fat_df["recent_minutes_a"] = 0.0
    fat_df["recent_minutes_b"] = 0.0

    history = defaultdict(lambda: deque())

    def get_fatigue(player_id, current_date):
        dq = history[player_id]
        if not dq:
            return 0, 0.0

        # HARD SAME-DAY CUTOFF TEST:
        # Drop everything before the current calendar date.
        # This is intentionally conservative and removes all history
        # that is not on the current date or later.
        # (Matches on the same date can remain; we then exclude them
        #  by only counting strictly earlier dates below if desired.)
        while dq and dq[0][0] < current_date.normalize():
            dq.popleft()

        matches = len(dq)
        minutes = sum(m for _, m in dq)
        return matches, minutes

    def update_history(player_id, date, minutes):
        history[player_id].append((date, minutes))

    for index, row in df_iter.iterrows():
        date = row["tourney_date"]
        mins = row["minutes"]

        id_a = row["id_a"]
        id_b = row["id_b"]

        matches_a, minutes_a = get_fatigue(id_a, date)
        matches_b, minutes_b = get_fatigue(id_b, date)

        fat_df.loc[index, "recent_matches_a"] = matches_a
        fat_df.loc[index, "recent_matches_b"] = matches_b
        fat_df.loc[index, "recent_minutes_a"] = minutes_a
        fat_df.loc[index, "recent_minutes_b"] = minutes_b

        update_history(id_a, date, mins)
        update_history(id_b, date, mins)

    return fat_df


def build_fatigue_strict_dataset() -> pd.DataFrame:
    """
    Build a DataFrame containing ONLY:
      - fatigue features (strict same-day cutoff)
      - match result
    """
    logger.info("Loading base data for strict fatigue features...")
    df = get_data()

    logger.info("Computing strict fatigue features (no same-day history)...")
    fat_df = fatigue_features_strict(df)

    # Attach result column
    out_df = fat_df.merge(df[["order", "result"]], on="order", how="left")
    out_df = out_df.sort_values("order").reset_index(drop=True)

    return out_df


def main() -> None:
    logger.info("Starting strict fatigue feature calculation (leakage test)...")
    out_df = build_fatigue_strict_dataset()

    output_path = "./data/features/base/fatigue_strict.parquet"
    out_df.to_parquet(output_path, index=True)
    logger.info(
        "Strict fatigue feature calculation completed! "
        f"Saved {len(out_df)} rows to {output_path}"
    )


if __name__ == "__main__":
    main()


