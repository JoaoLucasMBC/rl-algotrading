"""
Gym environment for exit strategy learning
Follows paper's optimal stopping formulation
"""

import gymnasium as gym
import numpy as np


class ExitEnv(gym.Env):
    """
    Environment for learning exit strategy
    
    State: [price_returns_15d, pnl%, days_held, rsi, volatility]
    Action: 0=HOLD, 1=SELL
    Reward: Paper's formulation - immediate payout on SELL, 0 on HOLD
    """
    
    def __init__(self, episode_data, entry_price, lookback=15, warmup_steps=0):
        super().__init__()
        self.episode_data = episode_data
        self.entry_price = float(entry_price)
        self.lookback = lookback
        self.warmup_steps = warmup_steps
        self.max_steps = len(episode_data)
        
        self.action_space = gym.spaces.Discrete(2)  # HOLD, SELL
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(lookback + 4,),  # 15 returns + 4 features
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.done = False
        return self._get_state()
    
    def _get_state(self):
        """
        State representation (Paper-aligned):
        - 15 recent prices (normalized by entry_price)
        - Current PnL %
        - Time remaining (normalized)
        - Current RSI (normalized)
        - Recent volatility
        """
        idx = self.current_step
        
        # Prices (last 15 days)
        prices = self.episode_data['Close'].values[:idx+1]
        
        # Ensure prices is 1D array
        if len(prices.shape) > 1:
            prices = prices.flatten()
        
        normalized_prices = np.zeros(self.lookback, dtype=np.float32)
        
        if len(prices) > 0:
            # Normalize by entry price
            # P_t / P_entry
            norm_p = prices / self.entry_price
            
            # Take last lookback prices
            n_prices = min(len(norm_p), self.lookback)
            if n_prices > 0:
                normalized_prices[-n_prices:] = norm_p[-n_prices:]
        
        # Current PnL %
        current_price = float(self.episode_data.iloc[idx]['Close'])
        pnl_pct = (current_price - self.entry_price) / self.entry_price
        
        # Time remaining (normalized)
        # T - t
        time_remaining = (self.max_steps - idx) / self.max_steps
        
        # RSI (normalized to 0-1)
        rsi = float(self.episode_data.iloc[idx]['RSI']) / 100.0
        
        # Volatility (std of last 15 returns)
        # We still calculate returns for volatility as it's a good feature
        returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else []
        volatility = float(np.std(returns)) if len(returns) > 0 else 0.0
                
        state = np.concatenate([
            normalized_prices,
            np.array([pnl_pct, time_remaining, rsi, volatility], dtype=np.float32)
        ])
        
        return state
    
    def step(self, action):
        """
        Execute action (Paper's formulation)
        action=0: HOLD (continue)
        action=1: SELL (exit position)
        """
        current_price = float(self.episode_data.iloc[self.current_step]['Close'])
        
        # Force HOLD during warmup period
        if self.current_step < self.warmup_steps:
            action = 0
            reward = 0
            self.current_step += 1
            # Check if done (shouldn't happen during warmup unless max_steps is tiny)
            if self.current_step >= self.max_steps:
                self.done = True
            
            next_state = self._get_state() if not self.done else np.zeros(self.observation_space.shape)
            info = {
                'exit_price': None,
                'days_held': self.current_step - self.warmup_steps,
                'pnl_pct': 0
            }
            return next_state, reward, self.done, info

        # Minimum hold period: can't sell in first 3 days AFTER warmup
        min_hold_days = 3
        if action == 1 and (self.current_step - self.warmup_steps) < min_hold_days:
            action = 0  # Force HOLD if trying to exit too early
        
        if action == 1:  # SELL
            # Paper's reward: immediate payout g_t(s_t)
            actual_return = (current_price - self.entry_price) / self.entry_price
            reward = actual_return  # No scaling (raw return)
            self.done = True
            
        else:  # HOLD
            # Paper's reward: 0 for continuing (value comes from future Q)
            reward = 0
            self.current_step += 1
            
            # Episode ends if we reach max steps
            if self.current_step >= self.max_steps - 1:
                # Forced exit at end
                final_price = float(self.episode_data.iloc[-1]['Close'])
                actual_return = (final_price - self.entry_price) / self.entry_price
                reward = actual_return  # No scaling
                self.done = True
        
        next_state = self._get_state() if not self.done else np.zeros(self.observation_space.shape)
        
        info = {
            'exit_price': current_price if self.done else None,
            'days_held': self.current_step,
            'pnl_pct': reward if self.done else 0  # Return original % for logging
        }
        
        return next_state, reward, self.done, info
