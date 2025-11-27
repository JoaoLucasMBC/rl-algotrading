import yfinance as yf
import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from utils import compute_features
from gym_env import EpisodeProvider, ExitTrainingEnv
from features import compute_bt_indicators

# 1. Configuration (Same as train.py)
tickers = ['^BVSP', 'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA']
start_date = '2013-11-01'
end_date = '2023-11-01'
max_horizon = 50

# 2. Data Collection & Preparation (Simplified for debug, just need the env structure)
dfs = []
all_entries = []
current_offset = 0

print(f"Loading data for environment reconstruction...")

for ticker in tickers:
    try:
        prices = yf.download(ticker, start=start_date, end=end_date, interval='1d', auto_adjust=True)
        if isinstance(prices.columns, pd.MultiIndex):
            prices = prices.droplevel(1, axis=1)
        df = prices.dropna()
        
        if len(df) < 200: continue

        df_ind = compute_bt_indicators(df)
        df = df.join(df_ind).dropna()
        df = df.reset_index(drop=True)
        
        entries = df.index[(df["rsi"] < 30) & (df.index < len(df) - max_horizon)].to_numpy()
        entries += current_offset
        
        dfs.append(df)
        all_entries.append(entries)
        current_offset += len(df)
        
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

if not dfs:
    raise ValueError("No data downloaded!")

df_train = pd.concat(dfs, ignore_index=True)
train_entries = np.concatenate(all_entries)

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
# Important: We must wrap it exactly as in training to load the model correctly
vec_env = DummyVecEnv([lambda: env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

# 4. Load Model and Debug
model_path = "exit_rl_dqn_improved"
print(f"Loading model from {model_path}...")

try:
    model = DQN.load(model_path, env=vec_env)
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

print("\n--- Starting Debug Run (100 Episodes) ---")

n_episodes = 100
action_counts = {0: 0, 1: 0}
episode_lengths = []
episode_rewards = []

for i in range(n_episodes):
    obs = vec_env.reset()
    done = False
    length = 0
    total_reward = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        
        action_counts[action[0]] += 1
        length += 1
        total_reward += reward[0]
        
    episode_lengths.append(length)
    episode_rewards.append(total_reward)

# 5. Analysis
total_actions = sum(action_counts.values())
hold_pct = (action_counts[0] / total_actions) * 100
exit_pct = (action_counts[1] / total_actions) * 100
avg_len = np.mean(episode_lengths)
avg_rew = np.mean(episode_rewards)

print("\n--- Debug Results ---")
print(f"Total Actions Taken: {total_actions}")
print(f"Action 0 (HOLD): {action_counts[0]} ({hold_pct:.2f}%)")
print(f"Action 1 (EXIT): {action_counts[1]} ({exit_pct:.2f}%)")
print(f"Average Episode Length: {avg_len:.2f} (Max: {max_horizon})")
print(f"Average Reward: {avg_rew:.4f}")

if avg_len >= max_horizon * 0.95:
    print("\n[DIAGNOSIS] The agent is mostly HOLDING until timeout.")
elif avg_len <= 2:
    print("\n[DIAGNOSIS] The agent is mostly EXITING immediately.")
else:
    print("\n[DIAGNOSIS] The agent is taking mixed actions.")
