import numpy as np
import pandas as pd

class KalmanFilterV1:
    def __init__(
        self,
        init_s: float = 0.0,
        init_P: float = 0.5,
        q: float = 0.001,
        k: float = 1.0,
        eps_R: float = 1e-4,
    ):
        """
        Initialise Kalman filter with hyperparameters.
        """
        self.init_s = init_s
        self.init_P = init_P
        self.q = q
        self.k = k
        self.eps_R = eps_R

    def sigmoid(self, x, k=None):
        """Compute sigmoid function."""
        if k is None:
            k = self.k
        return 1.0 / (1.0 + np.exp(-k * x))

    def generate_kalman_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Kalman filter features for a given DataFrame."""
        # Player state dictionaries
        s = {}  # skill
        P = {}  # uncertainty
        last_match_date = {}  # last match date

        # Add columns to df
        df = df.copy()
        df["skill_a"] = 0.0
        df["skill_b"] = 0.0
        df["skill_uncertainty_a"] = 0.0
        df["skill_uncertainty_b"] = 0.0

        df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")

        for idx, row in df.iterrows():
            id_a = row["id_a"]
            id_b = row["id_b"]

            if id_a not in s:
                s[id_a] = self.init_s
                P[id_a] = self.init_P
                last_match_date[id_a] = row["tourney_date"]
            if id_b not in s:
                s[id_b] = self.init_s
                P[id_b] = self.init_P
                last_match_date[id_b] = row["tourney_date"]

            # update df
            df.loc[idx, "skill_a"] = s[id_a]
            df.loc[idx, "skill_b"] = s[id_b]
            df.loc[idx, "skill_uncertainty_a"] = P[id_a]
            df.loc[idx, "skill_uncertainty_b"] = P[id_b]

            # update skill

            # build prior
            skill_drift_a = max(min((row["tourney_date"] - last_match_date[id_a]).days * self.q, 75 * self.q), 2 * self.q)
            skill_drift_b = max(min((row["tourney_date"] - last_match_date[id_b]).days * self.q, 75 * self.q), 2 * self.q)

            s_a_prior = s[id_a]
            s_b_prior = s[id_b]
            P_a_prior = P[id_a] + skill_drift_a
            P_b_prior = P[id_b] + skill_drift_b

            y_pred = self.sigmoid(s_a_prior - s_b_prior, k=self.k).item()
            y_pred = np.clip(y_pred, 1e-4, 1 - 1e-4)

            # build jacobian
            H = np.array([[1, -1]]) * self.k * y_pred * (1 - y_pred)
            P_mat = np.array([[P_a_prior, 0], [0, P_b_prior]])

            # compute kalman gain
            R = max(y_pred * (1 - y_pred), 1e-4)
            S = H @ P_mat @ H.T + R
            S = max(S, 1e-6)
            k_gain = P_mat @ H.T / S

            # update
            e = row["result"] - y_pred
            s_delta = k_gain * e
            new_P_mat = (np.eye(2) - k_gain @ H) @ P_mat
            new_P_mat += 1e-6 * np.eye(2)

            s[id_a] = s_a_prior + s_delta[0]
            s[id_b] = s_b_prior + s_delta[1]
            P[id_a] = new_P_mat[0, 0]
            P[id_b] = new_P_mat[1, 1]
            last_match_date[id_a] = row["tourney_date"]
            last_match_date[id_b] = row["tourney_date"]

        return df

    def evaluate_kalman_filter(self, df: pd.DataFrame) -> tuple:
        """Evaluate predictive power of kalman filter. Takes in a dataframe processed by generate_kalman_features."""
        correct = 0
        log_losses = []

        for idx, row in df.iterrows():
            # Compute prediction from stored skill values
            pred_prob = self.sigmoid(row["skill_a"] - row["skill_b"], k=self.k)
            result = row["result"]

            pred_prob_clipped = np.clip(pred_prob, 1e-6, 1 - 1e-6)
            log_loss = -result * np.log(pred_prob_clipped) - (1 - result) * np.log(1 - pred_prob_clipped)
            log_losses.append(log_loss)

            if (pred_prob > 0.5 and result == 1) or (pred_prob <= 0.5 and result == 0):
                correct += 1

        accuracy = correct / len(df) if len(df) > 0 else 0.0
        mean_log_loss = np.mean(log_losses)
        return accuracy, mean_log_loss
