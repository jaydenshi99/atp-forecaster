import logging
from typing import Hashable

import numpy as np
import pandas as pd
from glicko2 import Player

from atp_forecaster.data.clean import get_cleaned_atp_matches

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_data():
    return get_cleaned_atp_matches()

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
    glicko_df = df[["order", id_a_col, id_b_col, date_col]].copy()

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

        # Cache pre-match ratings/RDs so updates use old values
        Ra_old, Rb_old = p_a.getRating(), p_b.getRating()
        RDa_old, RDb_old = p_a.getRd(), p_b.getRd()

        Ra_surf_old, Rb_surf_old = p_a_surface.getRating(), p_b_surface.getRating()
        RDa_surf_old, RDb_surf_old = p_a_surface.getRd(), p_b_surface.getRd()

        if res == 1:          # A wins
            p_a.update_player([Rb_old], [RDb_old], [1.0])
            p_b.update_player([Ra_old], [RDa_old], [0.0])

            p_a_surface.update_player([Rb_surf_old], [RDb_surf_old], [1.0])
            p_b_surface.update_player([Ra_surf_old], [RDa_surf_old], [0.0])
        elif res == 0:        # B wins
            p_a.update_player([Rb_old], [RDb_old], [0.0])
            p_b.update_player([Ra_old], [RDa_old], [1.0])

            p_a_surface.update_player([Rb_surf_old], [RDb_surf_old], [0.0])
            p_b_surface.update_player([Ra_surf_old], [RDa_surf_old], [1.0])
        else:
            raise ValueError(f"Unsupported result value in '{result_col}': {res}")

    return glicko_df

def main():
    logger.info("Starting Glicko-2 rating calculation...")
    df = get_data()
    glicko_df = add_glicko2_elo(df)
    # Save as parquet (preserves index and data types, faster and smaller)
    glicko_df.to_parquet('./data/features/base/glicko2_ratings.parquet', index=True)
    logger.info(f"Glicko-2 rating calculation completed! Saved {len(glicko_df)} rows to ./data/features/base/glicko2_ratings.parquet")

if __name__ == "__main__":
    main()