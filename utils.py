import numpy as np

def compute_features(df, idx, entry_idx):
    price = df.iloc[idx].Close
    entry_price = df.iloc[entry_idx].Close

    pnl = (price - entry_price) / entry_price
    price_relative = price / entry_price
    time_in_trade = idx - entry_idx

    return np.array([
        pnl,
        price_relative,
        time_in_trade,

        # Backtrader indicators (already in df)
        df.iloc[idx].rsi,
        df.iloc[idx].roc,
        df.iloc[idx].volatility,
        df.iloc[idx].macd_hist,
        df.iloc[idx].atr,

        (price - df.iloc[idx].ma) / df.iloc[idx].ma,
        (price - df.iloc[idx].ema) / df.iloc[idx].ema,
        df.iloc[idx].bb_perc,
    ], dtype=np.float32)
