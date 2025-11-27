import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ======================================================
#  LAYER 1 — RLExitEnv
#  Environment for a SINGLE trade episode (exit-only RL)
# ======================================================
class RLExitEnv(gym.Env):
    """
    Environment where the agent decides when to EXIT a trade.
    Episode = life of one open trade.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        entry_index,
        max_horizon=50,
        stop_loss=-0.05,
        take_profit=0.10,
        feature_function=None,
    ):
        super(RLExitEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.entry_idx = int(entry_index)
        self.entry_price = float(self.df.loc[self.entry_idx, "Close"])

        self.max_horizon = max_horizon
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        # Feature extractor
        if feature_function is None:
            self.feature_function = self.default_features
        else:
            self.feature_function = feature_function

        # Action space: HOLD(0) or EXIT(1)
        self.action_space = spaces.Discrete(2)

        # Infer observation shape from a sample feature vector
        sample_features = np.array(self.feature_function(self.df, self.entry_idx, self.entry_idx))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=sample_features.shape,
            dtype=np.float32,
        )

        # Pointer during episode
        self.current_idx = self.entry_idx

        # Precompute R_max over the horizon
        end_idx = min(self.entry_idx + self.max_horizon, len(self.df) - 1)
        price_slice = self.df.loc[self.entry_idx:end_idx, "Close"].values
        max_price = price_slice.max()

        self.R_max = (max_price - self.entry_price) / self.entry_price

    # ------------------------------------------------------
    def default_features(self, df, idx, entry_idx):
        """Basic features: pnl, price_relative, time step"""
        price = df.loc[idx, "Close"]
        pnl = (price - df.loc[entry_idx, "Close"]) / df.loc[entry_idx, "Close"]
        time = idx - entry_idx

        return np.array([pnl, price / df.loc[entry_idx, "Close"], time], dtype=np.float32)

    # ------------------------------------------------------
    def compute_return(self, exit_idx):
        price_exit = self.df.loc[int(exit_idx), "Close"]
        return (price_exit - self.entry_price) / self.entry_price

    # ------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = self.entry_idx
        obs = self.feature_function(self.df, self.current_idx, self.entry_idx)
        return obs, {} 

    # ------------------------------------------------------
    def step(self, action):
        done = False
        reward = 0.0

        # ACTION 1 → EXIT NOW
        if action == 1:
            R_exit = self.compute_return(self.current_idx)
            reward = R_exit - self.R_max
            done = True
            return (
                self.feature_function(self.df, self.current_idx, self.entry_idx).astype(np.float32),
                float(reward),
                done,
                False,
                {},
            )

        # ACTION 0 → HOLD
        self.current_idx += 1
        
        # Time penalty for holding
        time_penalty = -0.0001
        reward += time_penalty

        # Forced exit at horizon end
        if self.current_idx >= self.entry_idx + self.max_horizon:
            R_exit = self.compute_return(self.current_idx)
            reward += (R_exit - self.R_max)
            done = True

        # Optional early exit (stop-loss or take-profit)
        else:
            current_ret = self.compute_return(self.current_idx)
            if current_ret <= self.stop_loss or current_ret >= self.take_profit:
                R_exit = current_ret
                reward += (R_exit - self.R_max)
                done = True

        return (
            self.feature_function(self.df, self.current_idx, self.entry_idx).astype(np.float32),
            float(reward),
            done,
            False,
            {},
        )

    # ------------------------------------------------------
    def render(self, mode="human"):
        print(
            f"Step {self.current_idx - self.entry_idx} | "
            f"Price: {self.df.loc[self.current_idx, 'Close']}"
        )


# ======================================================
#  LAYER 2 — EpisodeProvider
#  Picks a random entry index and creates an RLExitEnv
# ======================================================
class EpisodeProvider:
    """
    Stores possible entry points and samples a new RLExitEnv.
    """

    def __init__(
        self,
        df,
        entry_indices,
        max_horizon=50,
        stop_loss=-0.05,
        take_profit=0.10,
        feature_function=None,
    ):
        self.df = df
        self.entry_indices = entry_indices
        self.max_horizon = max_horizon
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.feature_function = feature_function

    def sample_env(self):
        """Returns one RLExitEnv randomly initialized at an entry index."""
        entry = int(np.random.choice(self.entry_indices))
        env = RLExitEnv(
            df=self.df,
            entry_index=entry,
            max_horizon=self.max_horizon,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            feature_function=self.feature_function,
        )
        return env


# ======================================================
#  LAYER 3 — ExitTrainingEnv
#  The real Gym env used by Stable-Baselines3
# ======================================================
class ExitTrainingEnv(gym.Env):
    """
    Stable-Baselines3-compatible environment.
    On reset(), it creates a NEW RLExitEnv for a NEW trade.
    """

    def __init__(self, episode_provider):
        super(ExitTrainingEnv, self).__init__()
        self.provider = episode_provider

        # Initialize first episode
        self.current_env = self.provider.sample_env()

        # Mirror spaces
        self.action_space = self.current_env.action_space
        self.observation_space = self.current_env.observation_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_env = self.provider.sample_env()
        obs, info = self.current_env.reset(seed=seed)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.current_env.step(action)
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        return self.current_env.render(mode=mode)
    