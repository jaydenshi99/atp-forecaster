import numpy as np
import pandas as pd

class KalmanFilterV2:
    def __init__(
        self,
        phi: float = 0.995,
        q_g: float = 1e-3,
        q_d: float = 5e-4,
        k_elo: float = 1.0,
        eps_R: float = 1e-4,
        init_var_g: float = 0.50,
        init_var_d: float = 0.20,
    ):
        """
        Initialize Kalman filter with hyperparameters.
        
        Args:
            phi: Mean reversion rate for surface differentials (0 < phi <= 1)
            q_g: Global skill drift variance per day
            q_d: Surface offset drift variance per day
            k_elo: Observation scale factor (Elo scaling)
            eps_R: Minimum observation noise floor
            init_var_g: Initial variance for global skill
            init_var_d: Initial variance for surface offsets
        """
        self.phi = phi
        self.q_g = q_g
        self.q_d = q_d
        self.k_elo = k_elo
        self.eps_R = eps_R
        self.init_var_g = init_var_g
        self.init_var_d = init_var_d

    # compute sigmoid function
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def generate_kalman_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Kalman filter features for a given DataFrame.
        """

        # drop carpet matches
        df = df.copy()
        df = df[df["surface"] != "Carpet"]

        # x_i = [g, d_clay, d_grass, d_hard]^T
        IDX = {"g": 0, "clay": 1, "grass": 2, "hard": 3}

        # Selector row vectors d_s
        def d_selector(surface):
            if surface == "clay":
                return np.array([[1, 1, 0, 0, -1, -1, 0, 0]], dtype=float)
            if surface == "grass":
                return np.array([[1, 0, 1, 0, -1, 0, -1, 0]], dtype=float)
            # default hard
            return np.array([[1, 0, 0, 1, -1, 0, 0, -1]], dtype=float)

        x = {}
        P = {}
        last_date = {}

        df["g_a"] = 0.0
        df["g_b"] = 0.0
        df["eff_skill_a"] = 0.0
        df["eff_skill_b"] = 0.0
        df["eff_var_a"] = 0.0
        df["eff_var_b"] = 0.0
        df["p_pred"] = 0.0

        init_x = np.zeros(4, dtype=float)
        init_P = np.diag([self.init_var_g, self.init_var_d, self.init_var_d, self.init_var_d]).astype(float)

        # parameters
        phi = self.phi
        q_g = self.q_g
        q_d = self.q_d
        k_elo = self.k_elo
        eps_R = self.eps_R

        # effective skill calculation
        def eff_skill_and_var(xi, Pi, surface):
            v = np.zeros((4, 1))
            v[IDX["g"], 0] = 1.0
            v[IDX[surface], 0] = 1.0
            mu = (v.T @ xi.reshape(-1, 1)).item()
            var = (v.T @ Pi @ v).item()
            return mu, var

        for idx, row in df.iterrows():
            ida = row["id_a"]
            idb = row["id_b"]
            surface = str(row.get("surface", "hard")).lower()
            if surface not in ("clay", "grass", "hard"):
                surface = "hard"

            # init players
            if ida not in x:
                x[ida] = init_x.copy()
                P[ida] = init_P.copy()
                last_date[ida] = row["tourney_date"]
            if idb not in x:
                x[idb] = init_x.copy()
                P[idb] = init_P.copy()
                last_date[idb] = row["tourney_date"]

            # prediction step
            dta = max((row["tourney_date"] - last_date[ida]).days, 0)
            dtb = max((row["tourney_date"] - last_date[idb]).days, 0)

            Fa = np.diag([1.0, phi**dta, phi**dta, phi**dta])
            Fb = np.diag([1.0, phi**dtb, phi**dtb, phi**dtb])

            Qa = np.diag([q_g * dta, q_d * dta, q_d * dta, q_d * dta])
            Qb = np.diag([q_g * dtb, q_d * dtb, q_d * dtb, q_d * dtb])

            xa = Fa @ x[ida]
            xb = Fb @ x[idb]
            Pa = Fa @ P[ida] @ Fa.T + Qa
            Pb = Fb @ P[idb] @ Fb.T + Qb

            # log current state into df
            mu_a, var_a = eff_skill_and_var(xa, Pa, surface)
            mu_b, var_b = eff_skill_and_var(xb, Pb, surface)

            df.loc[idx, "g_a"] = xa[IDX["g"]]
            df.loc[idx, "g_b"] = xb[IDX["g"]]
            df.loc[idx, "eff_skill_a"] = mu_a
            df.loc[idx, "eff_skill_b"] = mu_b
            df.loc[idx, "eff_var_a"] = var_a
            df.loc[idx, "eff_var_b"] = var_b

            # build match-level state
            x8 = np.concatenate([xa, xb]).reshape(8, 1)
            P8 = np.zeros((8, 8), dtype=float)
            P8[:4, :4] = Pa
            P8[4:, 4:] = Pb

            # observation model
            ds = d_selector(surface)
            z = (k_elo * (ds @ x8)).item()
            p = float(self.sigmoid(z))
            p = float(np.clip(p, 1e-6, 1 - 1e-6))
            df.loc[idx, "p_pred"] = p

            # jacobian
            H = (k_elo * p * (1.0 - p)) * ds

            # approximate observation noise
            R = max(p * (1.0 - p), eps_R)

            # kalman gain
            S = (H @ P8 @ H.T).item() + R
            S = max(S, 1e-12)
            K = (P8 @ H.T) / S

            # update
            r = float(row["result"])
            e = r - p
            x8_post = x8 + K * e

            # use joseph covariance update for stability
            I8 = np.eye(8)
            KH = K @ H
            P8_post = (I8 - KH) @ P8 @ (I8 - KH).T + (K * R) @ K.T

            x[ida] = x8_post[:4, 0]
            x[idb] = x8_post[4:, 0]
            P[ida] = P8_post[:4, :4]
            P[idb] = P8_post[4:, 4:]
            last_date[ida] = row["tourney_date"]
            last_date[idb] = row["tourney_date"]

        return df
    
    def evaluate_kalman_filter(self, df: pd.DataFrame) -> tuple:
        """Evaluate predictive power of kalman filter. Takes in a dataframe processed by generate_kalman_features."""

        correct = 0
        log_losses = []

        for idx, row in df.iterrows():
            # Use the actual prediction from the kalman filter
            pred_prob = row["p_pred"]
            result = row["result"]

            pred_prob_clipped = np.clip(pred_prob, 1e-6, 1 - 1e-6)
            log_loss = -result * np.log(pred_prob_clipped) - (1 - result) * np.log(1 - pred_prob_clipped)
            log_losses.append(log_loss)

            if (pred_prob > 0.5 and result == 1) or (pred_prob <= 0.5 and result == 0):
                correct += 1

        accuracy = correct / len(df) if len(df) > 0 else 0.0
        mean_log_loss = np.mean(log_losses)
        return accuracy, mean_log_loss