import backtrader as bt
import pandas as pd

class IndicatorExporter(bt.Strategy):
    def __init__(self):
        # Create all indicators we want
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.roc = bt.indicators.ROC(self.data.close, period=14)
        self.vol = bt.indicators.StdDev(self.data.close, period=14)
        self.macd = bt.indicators.MACDHisto(self.data.close)
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.ma = bt.indicators.SMA(self.data.close, period=14)
        self.ema = bt.indicators.EMA(self.data.close, period=14)
        self.bbands = bt.indicators.BBands(self.data.close, period=14)

        # Storage list
        self.rows = []

    def next(self):
        self.rows.append({
            "datetime": self.data.datetime.datetime(),
            # Indicators
            "rsi": self.rsi[0],
            "roc": self.roc[0],
            "volatility": self.vol[0],
            "macd_hist": self.macd.histo[0],
            "atr": self.atr[0],
            "ma": self.ma[0],
            "ema": self.ema[0],
            
            # Bollinger %B = (price - lower) / (upper - lower)
            "bb_perc": (self.data.close[0] - self.bbands.bot[0]) /
                       (self.bbands.top[0] - self.bbands.bot[0])
        })

def compute_bt_indicators(df):
    data = bt.feeds.PandasData(dataname=df)
    cerebro = bt.Cerebro()
    cerebro.adddata(data)

    # Add strategy, but do NOT assign the result (it's just an int)
    cerebro.addstrategy(IndicatorExporter)

    # Run the engine -> returns list of strategies (instances)
    strategies = cerebro.run()
    exporter = strategies[0]   # THIS is your IndicatorExporter instance

    # Now exporter.rows exists
    df_ind = pd.DataFrame(exporter.rows)
    df_ind.set_index("datetime", inplace=True)
    return df_ind


