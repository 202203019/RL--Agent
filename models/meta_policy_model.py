from stable_baselines3 import PPO
import numpy as np

class MetaPolicyModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load(self):
        # We use stable_baselines3 which handles weights internally
        self.model = PPO.load(self.model_path)

    def predict_action(self, x: np.ndarray) -> int:
        """
        Input: 19-dim meta-feature vector
        Output: Action index (0=SKIP, 1=ENTER, 2=WAIT)
        """
        action, _ = self.model.predict(x, deterministic=True)
        return int(action)
