import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import os
import yaml
from tqdm import tqdm
import random

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def download_stock_data(tickers, start_date, end_date):
    data = {}
    print(f"Downloading data for {len(tickers)} tickers...")
    
    for ticker in tqdm(tickers):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                continue
            
            # Basic cleaning
            df = df.dropna()
            
            # Handle MultiIndex if present
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)
            
            # Calculate RSI
            df['RSI'] = calculate_rsi(df['Close'])
            
            # Drop initial NaNs
            df.dropna(subset=['RSI'], inplace=True)
            
            if len(df) > 100:  # Minimum length
                data[ticker] = df
                
        except Exception as e:
            print(f"Error {ticker}: {e}")
            
    return data

def create_random_episodes(data, num_episodes=15000, max_hold_days=20, lstm_warmup=12):
    """
    Create episodes by randomly sampling start dates from all available data.
    This mimics the paper's approach of training on random trajectories.
    """
    episodes = []
    tickers = list(data.keys())
    
    print(f"Generating {num_episodes} random episodes...")
    
    with tqdm(total=num_episodes) as pbar:
        while len(episodes) < num_episodes:
            # Pick random ticker
            ticker = random.choice(tickers)
            df = data[ticker]
            
            # Pick random valid start index
            # valid range: [lstm_warmup, len(df) - max_hold_days]
            min_idx = lstm_warmup
            max_idx = len(df) - max_hold_days - 1
            
            if max_idx <= min_idx:
                continue
                
            entry_idx = random.randint(min_idx, max_idx)
            
            # Create episode
            start_idx = entry_idx - lstm_warmup
            end_idx = entry_idx + max_hold_days
            
            episode_data = df.iloc[start_idx:end_idx].copy()
            entry_price = float(df.iloc[entry_idx]['Close'])
            
            episodes.append({
                'ticker': ticker,
                'data': episode_data,
                'entry_price': entry_price,
                'warmup_steps': lstm_warmup,
                'entry_date': df.index[entry_idx]
            })
            pbar.update(1)
            
    return episodes

def create_rsi_episodes(data, rsi_threshold=30, max_hold_days=20, lstm_warmup=12):
    """
    Create episodes specifically from RSI < 30 signals for Validation/Testing.
    """
    episodes = []
    
    for ticker, df in data.items():
        # Find signals
        for i in range(1, len(df) - max_hold_days):
            if df['RSI'].iloc[i] < rsi_threshold and df['RSI'].iloc[i-1] >= rsi_threshold:
                # Found signal
                entry_idx = i
                
                if entry_idx < lstm_warmup:
                    continue
                    
                start_idx = entry_idx - lstm_warmup
                end_idx = entry_idx + max_hold_days
                
                episode_data = df.iloc[start_idx:end_idx].copy()
                entry_price = float(df.iloc[entry_idx]['Close'])
                
                episodes.append({
                    'ticker': ticker,
                    'data': episode_data,
                    'entry_price': entry_price,
                    'warmup_steps': lstm_warmup,
                    'entry_date': df.index[entry_idx]
                })
                
    return episodes

def main():
    config = load_config()
    
    # Download data
    data = download_stock_data(
        config['tickers'],
        config['start_date'],
        config['end_date']
    )
    
    # 1. Generate Training Data (Random Sampling)
    # Target ~15k episodes for training
    train_episodes = create_random_episodes(
        data, 
        num_episodes=15000,
        max_hold_days=config['max_hold_days'],
        lstm_warmup=config['lstm_warmup']
    )
    
    # 2. Generate Validation/Test Data (RSI Signals ONLY)
    # We want to evaluate on the actual strategy conditions
    eval_episodes = create_rsi_episodes(
        data,
        rsi_threshold=config['rsi_entry_threshold'],
        max_hold_days=config['max_hold_days'],
        lstm_warmup=config['lstm_warmup']
    )
    
    # Split eval into val/test
    random.shuffle(eval_episodes)
    split_idx = int(len(eval_episodes) * 0.5)
    val_episodes = eval_episodes[:split_idx]
    test_episodes = eval_episodes[split_idx:]
    
    print(f"\nDataset Summary:")
    print(f"  Train (Random): {len(train_episodes)} episodes")
    print(f"  Val   (RSI<30): {len(val_episodes)} episodes")
    print(f"  Test  (RSI<30): {len(test_episodes)} episodes")
    
    # Save
    os.makedirs('data', exist_ok=True)
    with open('data/train_episodes.pkl', 'wb') as f:
        pickle.dump(train_episodes, f)
    with open('data/val_episodes.pkl', 'wb') as f:
        pickle.dump(val_episodes, f)
    with open('data/test_episodes.pkl', 'wb') as f:
        pickle.dump(test_episodes, f)
        
    print("\n✓ Data saved to data/ directory")

if __name__ == "__main__":
    main()
