"""
Data collection script for Brazilian stocks
Downloads historical data and identifies RSI entry signals
"""

import yfinance as yf
import pandas as pd
import numpy as np
import yaml
import os
from datetime import datetime
import pickle


def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def download_stock_data(ticker, start_date, end_date):
    """Download stock data from Yahoo Finance"""
    try:
        print(f"Downloading {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            print(f"  ⚠️  No data for {ticker}")
            return None
        df = df.dropna()
        df = df.droplevel(1, axis=1)
        print(f"  ✓ {ticker}: {len(df)} days")
        return df
    except Exception as e:
        print(f"  ✗ Error downloading {ticker}: {e}")
        return None


def find_entry_signals(df, rsi_threshold=30):
    """Find RSI entry signals (oversold)"""
    df['RSI'] = calculate_rsi(df['Close'])

    # Drop the NaN rows based on the RSI column
    df.dropna(subset=['RSI'], inplace=True)

    # Entry signal: RSI crosses below threshold
    entries = []
    for i in range(1, len(df)):
        if df['RSI'].iloc[i] < rsi_threshold and df['RSI'].iloc[i-1] >= rsi_threshold:
            entries.append(df.index[i])
    
    return entries


def create_episodes(df, ticker, entries, max_hold_days=20, lstm_warmup=12):
    """
    Create training episodes from entry signals
    Each episode: from (entry - warmup) to (entry + max_hold_days)
    """
    episodes = []
    
    for entry_date in entries:
        try:
            entry_idx = df.index.get_loc(entry_date)
            
            # Ensure we have enough history for warmup
            if entry_idx < lstm_warmup:
                continue
                
            start_idx = entry_idx - lstm_warmup
            end_idx = min(entry_idx + max_hold_days, len(df))
            
            # Need at least 5 days of holding period
            if end_idx - entry_idx < 5:
                continue
            
            episode_data = df.iloc[start_idx:end_idx].copy()
            
            # Store entry price (price at entry_idx, which is lstm_warmup steps into the episode)
            # We use the price at the actual entry signal for PnL calculations
            entry_price = df.iloc[entry_idx]['Close']
            episode_data['entry_price'] = entry_price
            
            episodes.append({
                'ticker': ticker,
                'entry_date': entry_date,
                'entry_price': entry_price,
                'data': episode_data,
                'warmup_steps': lstm_warmup
            })
        except Exception as e:
            print(f"  Error creating episode for {ticker} at {entry_date}: {e}")
            continue
    
    return episodes


def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    tickers = config['tickers']
    start_date = config['start_date']
    end_date = config['end_date']
    rsi_period = config['rsi_period']
    rsi_threshold = config['rsi_entry_threshold']
    max_hold_days = config['max_hold_days']
    lstm_warmup = config.get('lstm_warmup', 12)
    
    print("=" * 60)
    print("DATA COLLECTION - Brazilian Stocks")
    print("=" * 60)
    print(f"Tickers: {len(tickers)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"RSI Entry: < {rsi_threshold}")
    print(f"Max Hold: {max_hold_days} days")
    print(f"LSTM Warmup: {lstm_warmup} days")
    print("=" * 60)
    
    # Download data
    all_data = {}
    for ticker in tickers:
        df = download_stock_data(ticker, start_date, end_date)
        if df is not None and len(df) > 100:  # At least 100 days of data
            all_data[ticker] = df
    
    print(f"\n✓ Downloaded {len(all_data)} stocks successfully")
    
    # Save raw data
    print("\nSaving raw data...")
    for ticker, df in all_data.items():
        df.to_csv(f'data/raw/{ticker}.csv')
    
    # Find entry signals and create episodes
    print("\nFinding RSI entry signals...")
    all_episodes = []
    
    for ticker, df in all_data.items():
        entries = find_entry_signals(df, rsi_threshold)
        print(f"{ticker}: {len(entries)} entry signals")
        
        episodes = create_episodes(df, ticker, entries, max_hold_days, lstm_warmup)
        all_episodes.extend(episodes)
    
    print(f"\n✓ Created {len(all_episodes)} episodes total")
    
    # Split into train/val/test
    np.random.seed(42)
    np.random.shuffle(all_episodes)
    
    n_train = int(len(all_episodes) * config['train_split'])
    n_val = int(len(all_episodes) * config['val_split'])
    
    train_episodes = all_episodes[:n_train]
    val_episodes = all_episodes[n_train:n_train+n_val]
    test_episodes = all_episodes[n_train+n_val:]
    
    print(f"\nSplit:")
    print(f"  Train: {len(train_episodes)} episodes")
    print(f"  Val:   {len(val_episodes)} episodes")
    print(f"  Test:  {len(test_episodes)} episodes")
    
    # Save episodes
    print("\nSaving episodes...")
    with open('data/processed/train_episodes.pkl', 'wb') as f:
        pickle.dump(train_episodes, f)
    with open('data/processed/val_episodes.pkl', 'wb') as f:
        pickle.dump(val_episodes, f)
    with open('data/processed/test_episodes.pkl', 'wb') as f:
        pickle.dump(test_episodes, f)
    
    print("\n" + "=" * 60)
    print("✓ DATA COLLECTION COMPLETE")
    print("=" * 60)
    
    # Print some statistics
    returns = []
    for ep in all_episodes:
        final_price = ep['data']['Close'].iloc[-1]
        entry_price = ep['entry_price']
        ret = (final_price - entry_price) / entry_price
        returns.append(ret)
    
    print(f"\nEpisode Statistics:")
    print(f"  Avg Return: {np.mean(returns)*100:.2f}%")
    print(f"  Median Return: {np.median(returns)*100:.2f}%")
    print(f"  Std Return: {np.std(returns)*100:.2f}%")
    print(f"  Win Rate: {np.mean([r > 0 for r in returns])*100:.1f}%")
    print(f"  Best Return: {np.max(returns)*100:.2f}%")
    print(f"  Worst Return: {np.min(returns)*100:.2f}%")


if __name__ == '__main__':
    main()
