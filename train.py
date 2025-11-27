import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import backtrader as bt
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from utils import compute_features
from gym_env import EpisodeProvider, ExitTrainingEnv
from features import compute_bt_indicators

# 1. Configuration
tickers = ['^BVSP', 'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA']
start_date = '2013-11-01'
end_date = '2023-11-01'
max_horizon = 50

# 2. Data Collection & Preparation
dfs = []
all_entries = []
current_offset = 0

print(f"Downloading data for {tickers}...")

for ticker in tickers:
    try:
        # Download data
        prices = yf.download(ticker, start=start_date, end=end_date, interval='1d', auto_adjust=True)
        if isinstance(prices.columns, pd.MultiIndex):
            prices = prices.droplevel(1, axis=1)
        df = prices.dropna()
        
        if len(df) < 200:
            print(f"Skipping {ticker}: too few data points ({len(df)})")
            continue

        # Compute indicators
        print(f"Computing indicators for {ticker}...")
        df_ind = compute_bt_indicators(df)
        
        # Join indicators
        # Ensure indices match
        df = df.join(df_ind).dropna()
        
        # Reset index to have integer indexing for the environment
        df = df.reset_index(drop=True)
        
        # Identify entry points (RSI < 30)
        # We must ensure entry is not too close to the end
        # valid_mask = (df["rsi"] < 30) & (df.index < len(df) - max_horizon)
        # entries = df.index[valid_mask].to_numpy()
        
        # Using the logic from notebook but adapted for safety
        # Notebook: entries = df.index[df["rsi"] < 30].tolist()
        # We add the horizon check
        entries = df.index[(df["rsi"] < 30) & (df.index < len(df) - max_horizon)].to_numpy()
        
        # Shift entries by current offset
        entries += current_offset
        
        dfs.append(df)
        all_entries.append(entries)
        
        current_offset += len(df)
        print(f"Processed {ticker}: {len(df)} rows, {len(entries)} entries.")
        
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

if not dfs:
    raise ValueError("No data downloaded!")

# Concatenate all dataframes
df_train = pd.concat(dfs, ignore_index=True)
train_entries = np.concatenate(all_entries)

print(f"Total training data: {len(df_train)} rows")
print(f"Total entry points: {len(train_entries)}")

# 3. Environment Setup
provider = EpisodeProvider(
    df=df_train,
    entry_indices=train_entries,
    max_horizon=max_horizon,
    stop_loss=-0.05,
    take_profit=0.10,
    feature_function=compute_features
)

env = ExitTrainingEnv(provider)

# Wrap the env for normalization
vec_env = DummyVecEnv([lambda: env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

# 4. Model Training
# Optional: logging folder
logger = configure("./logs_dqn", ["stdout", "csv", "tensorboard"])

print("Starting training...")
model = DQN(
    "MlpPolicy",
    vec_env,
    learning_rate=1e-4,
    buffer_size=100_000,
    batch_size=64,
    learning_starts=1_000,
    gamma=0.99,
    tau=1.0,
    train_freq=4,
    gradient_steps=1,
    exploration_fraction=0.3,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    verbose=1,
    tensorboard_log="./logs_dqn/"
)

model.set_logger(logger)
model.learn(total_timesteps=100_000)
model.save("exit_rl_dqn_improved")
print("Training finished. Model saved to exit_rl_dqn_improved.zip")
