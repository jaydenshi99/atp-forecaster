import logging
from typing import Hashable

import numpy as np
import pandas as pd
from glicko2 import Player

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_data():
    # Load with index to preserve row order for merging
    return pd.read_csv("./data/cleaned/atp_matches_cleaned.csv", index_col=0)

def add_glicko2_elo(
    df: pd.DataFrame,
    *,
    id_a_col: str = "id_a",
    id_b_col: str = "id_b",
    result_col: str = "result",
    date_col: str = "tourney_date",
    surface_col: str = "surface",
    # default Glicko-2 params for new players
    default_rating: float = 1500.0,
    default_rd: float = 350.0,
    default_vol: float = 0.06,
) -> pd.DataFrame:
    """
    Creates df with 'elo_a' and 'elo_b' columns with PRE-MATCH Glicko-2 ratings.

    df is assumed to be sorted by date.

    result_col: 1 = A wins, 0 = B wins, 0.5 = draw.
    
    Returns a dataframe with columns: id_a, id_b, tourney_date, 
    elo_a, elo_b, elo_surface_a, elo_surface_b.
    """
    
    # Create output dataframe preserving the index for merging
    glicko_df = df[[id_a_col, id_b_col, date_col]].copy()

    glicko_df["elo_a"] = np.nan
    glicko_df["elo_b"] = np.nan

    glicko_df["elo_surface_a"] = np.nan
    glicko_df["elo_surface_b"] = np.nan

    # (pool_key, player_id) -> Player()
    players: dict[tuple[Hashable, Hashable], Player] = {}

    def get_player(pool_key: Hashable, pid: Hashable) -> Player:
        key = (pool_key, pid)
        if key not in players:
            players[key] = Player(
                rating=default_rating,
                rd=default_rd,
                vol=default_vol,
            )
        return players[key]

    for idx, row in df.iterrows():
        # global
        p_a = get_player("Global", row[id_a_col])
        p_b = get_player("Global", row[id_b_col])

        # surface specific
        p_a_surface = get_player(row[surface_col], row[id_a_col])
        p_b_surface = get_player(row[surface_col], row[id_b_col])

        # pre-match ratings
        glicko_df.at[idx, "elo_a"] = p_a.getRating()
        glicko_df.at[idx, "elo_b"] = p_b.getRating()

        glicko_df.at[idx, "elo_surface_a"] = p_a_surface.getRating()
        glicko_df.at[idx, "elo_surface_b"] = p_b_surface.getRating()

        res = row[result_col]

        # The library's update_player takes lists of opponent ratings/RDs and scores.
        if res == 1:          # A wins
            p_a.update_player([p_b.getRating()], [p_b.getRd()], [1.0])
            p_b.update_player([p_a.getRating()], [p_a.getRd()], [0.0])

            p_a_surface.update_player([p_b_surface.getRating()], [p_b_surface.getRd()], [1.0])
            p_b_surface.update_player([p_a_surface.getRating()], [p_a_surface.getRd()], [0.0])
        elif res == 0:        # B wins
            p_a.update_player([p_b.getRating()], [p_b.getRd()], [0.0])
            p_b.update_player([p_a.getRating()], [p_a.getRd()], [1.0])

            p_a_surface.update_player([p_b_surface.getRating()], [p_b_surface.getRd()], [0.0])
            p_b_surface.update_player([p_a_surface.getRating()], [p_a_surface.getRd()], [1.0])
        else:
            raise ValueError(f"Unsupported result value in '{result_col}': {res}")

    return glicko_df

def main():
    logger.info("Starting Glicko-2 rating calculation...")
    df = get_data()
    glicko_df = add_glicko2_elo(df)
    # Save with index for merging
    glicko_df.to_csv('./data/features/glicko2_ratings.csv', index=True)
    logger.info(f"Glicko-2 rating calculation completed! Saved {len(glicko_df)} rows to ./data/features/glicko2_ratings.csv")

if __name__ == "__main__":
    main()