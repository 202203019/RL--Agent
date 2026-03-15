"""
Train RecurrentPPO (LSTM) V3 for NIFTY 1-second dataset.
Uses windowed observations and Variant D churn controls.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import RecurrentPPO
from build_tensortrade_ppo_env import DailyNiftyEnv

# Settings
WINDOW_SIZE = 40
REWARD_VARIANT = "D"
COOLDOWN_PARAMS = {'lockout': 5, 'flip_cooldown': 10}
TOTAL_TIMESTEPS = 300_000 # Increased for LSTM complexity
N_ENVS = 4

def make_env(df, variant, cp, window):
    def _init():
        return DailyNiftyEnv(df, reward_variant=variant, cooldown_params=cp, window_size=window)
    return _init

def train():
    # Load data
    DATA_DIR = Path("data/labeled")
    files = sorted(list(DATA_DIR.glob("*.parquet")))
    
    # Simple split: first 80% for training
    train_files = files[:int(0.8 * len(files))]
    print(f"INFO: Loading {len(train_files)} training files...")
    all_df = [pd.read_parquet(f) for f in train_files]
    train_df = pd.concat(all_df).reset_index(drop=True)
    
    window_sizes = [20, 40, 60]
    MODEL_DIR = Path("models/ppo_v3")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    for ws in window_sizes:
        print(f"\n--- Training RecurrentPPO V3 (Window={ws}, Variant={REWARD_VARIANT}) ---")
        
        # Create vectorized environment for each window size
        env = SubprocVecEnv([make_env(train_df, REWARD_VARIANT, COOLDOWN_PARAMS, ws) for _ in range(N_ENVS)])
        
        # RecurrentPPO Configuration
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs={
                "lstm_hidden_size": 128,
                "n_lstm_layers": 1,
                "shared_lstm": True,
                "enable_critic_lstm": False
            },
            device="auto"
        )
        
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        
        model_path = MODEL_DIR / f"recurrent_ppo_v3_ws{ws}.zip"
        model.save(model_path)
        print(f"INFO: Model saved to {model_path}")
        
        env.close()

    # Log config
    with open("reports/ppo_v3_training_config.md", "w") as f:
        f.write("# PPO V3 Training Configuration\n\n")
        f.write(f"- **Algorithm**: RecurrentPPO (LSTM)\n")
        f.write(f"- **Window Sizes Benchmark**: [20, 40, 60]\n")
        f.write(f"- **Reward Variant**: {REWARD_VARIANT}\n")
        f.write(f"- **Cooldown Params**: {COOLDOWN_PARAMS}\n")
        f.write(f"- **LSTM Hidden Size**: 128\n")
        f.write(f"- **Total Timesteps per model**: {TOTAL_TIMESTEPS}\n")

if __name__ == "__main__":
    train()
