# holds the basic methods for the transitive model
import numpy as np
import pandas as pd

from atp_forecaster.scripts.point_to_match import point_to_match_dp

class TransitiveV2:
    def __init__(self, look_back: int = 120, base_spw: float = 0.6):
        self.history = None
        self.look_back = look_back  # look-back window in days
        self.base_spw = base_spw  # base serve win rate
        self.base_rpw = 1- base_spw  # base return win rate

    def fit(self, history: pd.DataFrame):
        """
        Fit the model to the history of matches.
        """
        self.history = history.copy()

        # convert YYYYMMDD to datetime
        self.history["tourney_date"] = pd.to_datetime(self.history["tourney_date"], format="%Y%m%d")
        base_cols = [
            "svpt_a", "1stWon_a", "2ndWon_a",
            "svpt_b", "1stWon_b", "2ndWon_b",
        ]
        missing = [c for c in base_cols if c not in self.history.columns]
        if missing:
            raise ValueError(f"Missing required columns to compute serve/return win rates: {missing}")

        # Serve win rates
        svpt_a = self.history["svpt_a"].replace(0, pd.NA)
        svpt_b = self.history["svpt_b"].replace(0, pd.NA)
        self.history["spw_a"] = (self.history["1stWon_a"] + self.history["2ndWon_a"]) / svpt_a
        self.history["spw_b"] = (self.history["1stWon_b"] + self.history["2ndWon_b"]) / svpt_b

        # Return win rates (opponent's serve points lost)
        self.history["rpw_a"] = (self.history["svpt_b"] - (self.history["1stWon_b"] + self.history["2ndWon_b"])) / svpt_b
        self.history["rpw_b"] = (self.history["svpt_a"] - (self.history["1stWon_a"] + self.history["2ndWon_a"])) / svpt_a

        # Replace div-by-zero NaNs with 0.0 for safety
        rate_cols = ["spw_a", "spw_b", "rpw_a", "rpw_b"]
        self.history[rate_cols] = self.history[rate_cols].astype(float).fillna(0.0)

    def add_match(self, match: pd.Series):
        """
        Add a match to the history.
        """
        base_cols = [
            "svpt_a", "1stWon_a", "2ndWon_a",
            "svpt_b", "1stWon_b", "2ndWon_b",
        ]
        missing = [c for c in base_cols if c not in match.index]
        if missing:
            raise ValueError(f"Missing required columns to compute serve/return win rates: {missing}")

        # make a single-row DataFrame
        row_df = match.to_frame().T.copy()
        row_df["tourney_date"] = pd.to_datetime(row_df["tourney_date"], format="%Y%m%d")

        svpt_a = row_df["svpt_a"].replace(0, pd.NA)
        svpt_b = row_df["svpt_b"].replace(0, pd.NA)
        row_df["spw_a"] = (row_df["1stWon_a"] + row_df["2ndWon_a"]) / svpt_a
        row_df["spw_b"] = (row_df["1stWon_b"] + row_df["2ndWon_b"]) / svpt_b

        row_df["rpw_a"] = (row_df["svpt_b"] - (row_df["1stWon_b"] + row_df["2ndWon_b"])) / svpt_b
        row_df["rpw_b"] = (row_df["svpt_a"] - (row_df["1stWon_a"] + row_df["2ndWon_a"])) / svpt_a

        rate_cols = ["spw_a", "spw_b", "rpw_a", "rpw_b"]
        row_df[rate_cols] = row_df[rate_cols].astype(float).fillna(0.0)

        if self.history is None:
            self.history = row_df
        else:
            self.history = pd.concat([self.history, row_df], ignore_index=True)

    def find_common_opponents(self, id_a, id_b, surface: str | None = None):
        """
        Find the common opponents between id_a and id_b.
        """
        if self.history is None:
            raise ValueError("Call fit() with match history before finding opponents.")

        # Use matches within the last look_back days if set
        recent = self.history.sort_values("tourney_date", ascending=False)
        if self.look_back:
            cutoff = recent["tourney_date"].max() - pd.Timedelta(days=self.look_back)
            recent = recent[recent["tourney_date"] >= cutoff]
        if surface is not None:
            recent = recent[recent["surface"] == surface]

        opponents_a = pd.concat([
            recent.loc[recent["id_a"] == id_a, "id_b"],
            recent.loc[recent["id_b"] == id_a, "id_a"],
        ])
        opponents_b = pd.concat([
            recent.loc[recent["id_a"] == id_b, "id_b"],
            recent.loc[recent["id_b"] == id_b, "id_a"],
        ])

        return set(opponents_a.unique()) & set(opponents_b.unique())

    def get_transitive_matches(self, id_a, id_b, surface: str | None = None):
        """
        Get per-opponent match histories for each player, restricted to common opponents.

        Returns:
            (dict, dict): ({opponent_id: DataFrame for id_a}, {opponent_id: DataFrame for id_b})
            Each DataFrame is trimmed to tourney_date, serve_win_rate, return_win_rate.
        """
        if self.history is None:
            raise ValueError("Call fit() with match history before fetching matches.")

        # Use matches within the last look_back days if set
        recent = self.history.sort_values("tourney_date", ascending=False)
        if self.look_back:
            cutoff = recent["tourney_date"].max() - pd.Timedelta(days=self.look_back)
            recent = recent[recent["tourney_date"] >= cutoff]
        if surface is not None:
            recent = recent[recent["surface"] == surface]

        common = self.find_common_opponents(id_a, id_b, surface=surface)

        required_cols = {
            "a": ("spw_a", "rpw_a"),
            "b": ("spw_b", "rpw_b"),
        }
        for side, cols in required_cols.items():
            missing = [c for c in cols if c not in recent.columns]
            if missing:
                raise ValueError(f"Missing expected columns for side {side}: {missing}")

        def group_by_opponent(player_id: int):
            played_as_a = recent.loc[recent["id_a"] == player_id].copy()
            played_as_a["opponent"] = played_as_a["id_b"]
            played_as_a["spw"] = played_as_a["spw_a"]
            played_as_a["rpw"] = played_as_a["rpw_a"]
            played_as_a["spw_w"] = played_as_a["svpt_a"]
            played_as_a["rpw_w"] = played_as_a["svpt_b"]

            played_as_b = recent.loc[recent["id_b"] == player_id].copy()
            played_as_b["opponent"] = played_as_b["id_a"]
            played_as_b["spw"] = played_as_b["spw_b"]
            played_as_b["rpw"] = played_as_b["rpw_b"]
            played_as_b["spw_w"] = played_as_b["svpt_b"]
            played_as_b["rpw_w"] = played_as_b["svpt_a"]

            combined = pd.concat([played_as_a, played_as_b], ignore_index=True)
            trimmed = combined[["tourney_date", "spw", "rpw", "spw_w", "rpw_w", "opponent"]]
            return {
                opp: grp.sort_values("tourney_date", ascending=False).reset_index(drop=True)
                for opp, grp in trimmed.groupby("opponent")
                if opp in common
            }

        return group_by_opponent(id_a), group_by_opponent(id_b)

    def predict_match(self, id_a, id_b, best_of: int = 3, surface: str | None = None, debug: bool = False):
        """
        Predict the outcome of a match between id_a and id_b. Returns the probability of id_a winning.
        """

        probs = []

        transitive_matches_a, transitive_matches_b = self.get_transitive_matches(id_a, id_b, surface=surface)
        for opponent in transitive_matches_a.keys():
            a_grp = transitive_matches_a[opponent].dropna(subset=["spw", "rpw", "spw_w", "rpw_w"])
            b_grp = transitive_matches_b[opponent].dropna(subset=["spw", "rpw", "spw_w", "rpw_w"])

            def wmean(series, weights):
                w = weights.replace(0, np.nan)
                num = (series * w).sum()
                den = w.sum()
                return num / den if den and not np.isnan(den) else np.nan

            spw_a = wmean(a_grp["spw"], a_grp["spw_w"])
            rpw_a = wmean(a_grp["rpw"], a_grp["rpw_w"])
            spw_b = wmean(b_grp["spw"], b_grp["spw_w"])
            rpw_b = wmean(b_grp["rpw"], b_grp["rpw_w"])

            if np.isnan(spw_a) or np.isnan(rpw_a) or np.isnan(spw_b) or np.isnan(rpw_b):
                continue
            # Serve probability for A vs B: base + (A serve strength minus base) - (B return strength minus base)
            p_srv = self.base_spw + (spw_a - self.base_spw) - (rpw_b - self.base_rpw)
            # Return probability for A vs B: base_return + (A return strength minus base_return) - (B serve strength minus base)
            p_ret = self.base_rpw + (rpw_a - self.base_rpw) - (spw_b - self.base_spw)

            p_srv = np.clip(p_srv, 0.0, 1.0)
            p_ret = np.clip(p_ret, 0.0, 1.0)

            p = point_to_match_dp(p_srv, p_ret, best_of=best_of)

            if debug:
                print(f"opp {opponent} spw_a={spw_a:.3f} rpw_a={rpw_a:.3f} spw_b={spw_b:.3f} rpw_b={rpw_b:.3f} p_srv={p_srv:.3f} p_ret={p_ret:.3f} p={p:.3f}")

            probs.append(p)
        if not probs:
            return point_to_match_dp(self.base_spw, self.base_rpw, best_of=best_of)

        return float(np.mean(probs))