import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb # Ensure lightgbm is in namespace

FEATURE_COLUMNS = [
    "log_ret_1s",
    "log_ret_5s",
    "log_ret_10s",
    "log_ret_30s",
    "ret_accel_10s",
    "time_of_day_sin",
    "time_of_day_cos",
    "rolling_std_60s",
    "volatility_ratio",
    "ema_fast_minus_slow",
    "body_1s",
    "range_1s",
    "close_position"
]

class LGBMSignalModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load(self):
        # We use joblib as it is more robust for sklearn/lgbm models
        self.model = joblib.load(self.model_path)


    def predict_probs(self, x: np.ndarray) -> dict:
        """
        Input: 13-dim feature vector (numpy array)
        Output: Dict with P_DOWN, P_UP, P_NONE
        """
        # Ensure x is 2D for DataFrame
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            
        # Create DataFrame with expected feature names to avoid sklearn warnings
        feat_df = pd.DataFrame(x, columns=FEATURE_COLUMNS)
        
        probs = self.model.predict_proba(feat_df)[0]
        return {
            "P_DOWN": float(probs[0]),
            "P_UP": float(probs[1]),
            "P_NONE": float(probs[2])
        }

