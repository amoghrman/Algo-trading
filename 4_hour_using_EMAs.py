from AlgorithmImports import *

class BitcoinMovingAverageCrossover(QCAlgorithm):

    def Initialize(self):
        # Define the backtest period
        self.SetStartDate(2024,1,1)
        
        self.SetCash(100)  # Set initial capital

        # Add BTCUSD pair - using daily resolution
        self.symbol = self.AddCrypto("SHIBUSD", Resolution.HOUR).Symbol
        
        # Define the moving average periods
        self.short_window = 12  # Short-term moving average (50 days)
        self.long_window = 48  # Long-term moving average (200 days)

        # Create Exponential Moving Averages (EMA)
        self.short_ma = self.EMA(self.symbol, self.short_window, Resolution.HOUR)
        self.long_ma = self.EMA(self.symbol, self.long_window, Resolution.HOUR)

        # Warm up the strategy to have enough data for the moving averages
        self.SetWarmUp(self.long_window*4)

    def OnData(self, data):
        # Ensure both moving averages are ready before making trades
        if not self.short_ma.IsReady or not self.long_ma.IsReady:
            return

        # Check if we are already invested in BTC
        invested = self.Portfolio[self.symbol].Invested

        # Buy signal: short-term EMA crosses above long-term EMA
        if self.short_ma.Current.Value > self.long_ma.Current.Value and not invested:
            self.SetHoldings(self.symbol, 1)  # Invest 100% of the portfolio in BTC
            self.Debug("Buy Signal Triggered")

        # Sell signal: short-term EMA crosses below long-term EMA
        elif self.short_ma.Current.Value < self.long_ma.Current.Value and invested:
            self.Liquidate(self.symbol)  # Sell all BTC holdings
            self.Debug("Sell Signal Triggered")