import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import time
import random

class StockDataProvider:
    def __init__(self, api_key=None):
        self.api_key = api_key or "demo"
        self.base_url = "https://www.alphavantage.co/query"
    
    def get_historical_data(self, symbol, period="1min", outputsize="compact"):
        """Fetch historical stock data from Alpha Vantage"""
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": period,
            "apikey": self.api_key,
            "outputsize": outputsize
        }
        
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        if "Time Series (1min)" in data:
            time_series = data["Time Series (1min)"]
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df = df.sort_index()
            return df
        else:
            print(f"Error fetching data: {data}")
            return None

class TradingSimulator:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.reset()
    
    def reset(self):
        self.capital = self.initial_capital
        self.positions = 0
        self.trades = []
        self.portfolio_value = []
        self.dates = []
    
    def buy(self, price, quantity, date):
        cost = price * quantity
        if cost <= self.capital:
            self.capital -= cost
            self.positions += quantity
            self.trades.append({
                'date': date,
                'action': 'BUY',
                'price': price,
                'quantity': quantity,
                'cost': cost
            })
            return True
        return False
    
    def sell(self, price, quantity, date):
        if quantity <= self.positions:
            revenue = price * quantity
            self.capital += revenue
            self.positions -= quantity
            self.trades.append({
                'date': date,
                'action': 'SELL',
                'price': price,
                'quantity': quantity,
                'revenue': revenue
            })
            return True
        return False
    
    def get_portfolio_value(self, current_price):
        return self.capital + (self.positions * current_price)
    
    def record_portfolio_value(self, price, date):
        portfolio_val = self.get_portfolio_value(price)
        self.portfolio_value.append(portfolio_val)
        self.dates.append(date)

class TradingStrategy:
    def __init__(self, name):
        self.name = name
    
    def should_buy(self, data, current_idx):
        raise NotImplementedError
    
    def should_sell(self, data, current_idx):
        raise NotImplementedError

class RandomStrategy(TradingStrategy):
    def __init__(self, buy_probability=0.1, sell_probability=0.1):
        super().__init__("Random Strategy")
        self.buy_prob = buy_probability
        self.sell_prob = sell_probability
    
    def should_buy(self, data, current_idx):
        return random.random() < self.buy_prob
    
    def should_sell(self, data, current_idx):
        return random.random() < self.sell_prob

class BuyAndHoldStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("Buy and Hold")
        self.bought = False
    
    def should_buy(self, data, current_idx):
        return not self.bought
    
    def should_sell(self, data, current_idx):
        return False

class SimpleMovingAverageStrategy(TradingStrategy):
    def __init__(self, short_window=5, long_window=20):
        super().__init__("Simple Moving Average")
        self.short_window = short_window
        self.long_window = long_window
    
    def should_buy(self, data, current_idx):
        if current_idx < self.long_window:
            return False
        
        short_ma = data['close'].iloc[current_idx-self.short_window:current_idx].mean()
        long_ma = data['close'].iloc[current_idx-self.long_window:current_idx].mean()
        
        return short_ma > long_ma
    
    def should_sell(self, data, current_idx):
        if current_idx < self.long_window:
            return False
        
        short_ma = data['close'].iloc[current_idx-self.short_window:current_idx].mean()
        long_ma = data['close'].iloc[current_idx-self.long_window:current_idx].mean()
        
        return short_ma < long_ma

class MomentumStrategy(TradingStrategy):
    def __init__(self, lookback_period=10, momentum_threshold=0.02):
        super().__init__("Momentum Strategy")
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold
    
    def should_buy(self, data, current_idx):
        if current_idx < self.lookback_period:
            return False
        
        # Calculate momentum as percentage change over lookback period
        current_price = data['close'].iloc[current_idx]
        past_price = data['close'].iloc[current_idx - self.lookback_period]
        momentum = (current_price - past_price) / past_price
        
        # Buy if positive momentum exceeds threshold
        return momentum > self.momentum_threshold
    
    def should_sell(self, data, current_idx):
        if current_idx < self.lookback_period:
            return False
        
        # Calculate momentum as percentage change over lookback period
        current_price = data['close'].iloc[current_idx]
        past_price = data['close'].iloc[current_idx - self.lookback_period]
        momentum = (current_price - past_price) / past_price
        
        # Sell if negative momentum exceeds threshold
        return momentum < -self.momentum_threshold

class RSIMomentumStrategy(TradingStrategy):
    def __init__(self, rsi_period=14, oversold_threshold=30, overbought_threshold=70):
        super().__init__("RSI Momentum Strategy")
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
    
    def calculate_rsi(self, prices):
        """Calculate RSI (Relative Strength Index)"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def should_buy(self, data, current_idx):
        if current_idx < self.rsi_period + 1:
            return False
        
        prices = data['close'].iloc[:current_idx + 1].values
        rsi = self.calculate_rsi(prices)
        
        # Buy when RSI indicates oversold condition (potential upward momentum)
        return rsi < self.oversold_threshold
    
    def should_sell(self, data, current_idx):
        if current_idx < self.rsi_period + 1:
            return False
        
        prices = data['close'].iloc[:current_idx + 1].values
        rsi = self.calculate_rsi(prices)
        
        # Sell when RSI indicates overbought condition (momentum reversing)
        return rsi > self.overbought_threshold

class BacktestEngine:
    def __init__(self, data_provider, initial_capital=10000):
        self.data_provider = data_provider
        self.initial_capital = initial_capital
    
    def run_backtest(self, symbol, strategy, shares_per_trade=10):
        """Run backtest for a given strategy"""
        data = self.data_provider.get_historical_data(symbol)
        if data is None:
            return None
        
        simulator = TradingSimulator(self.initial_capital)
        
        for i in range(len(data)):
            current_price = data['close'].iloc[i]
            current_date = data.index[i]
            
            # Check for buy signal
            if strategy.should_buy(data, i):
                simulator.buy(current_price, shares_per_trade, current_date)
            
            # Check for sell signal
            elif strategy.should_sell(data, i) and simulator.positions > 0:
                simulator.sell(current_price, min(shares_per_trade, simulator.positions), current_date)
            
            simulator.record_portfolio_value(current_price, current_date)
        
        return {
            'strategy': strategy.name,
            'final_value': simulator.get_portfolio_value(data['close'].iloc[-1]),
            'total_return': (simulator.get_portfolio_value(data['close'].iloc[-1]) - self.initial_capital) / self.initial_capital * 100,
            'trades': simulator.trades,
            'portfolio_values': simulator.portfolio_value,
            'dates': simulator.dates,
            'data': data
        }
    
    def compare_strategies(self, symbol, strategies):
        """Compare multiple strategies"""
        results = {}
        
        for strategy in strategies:
            print(f"Running backtest for {strategy.name}...")
            result = self.run_backtest(symbol, strategy)
            if result:
                results[strategy.name] = result
            time.sleep(1)  # Rate limiting for free API
        
        return results
    
    def plot_results(self, results):
        """Plot comparison of strategies"""
        plt.figure(figsize=(12, 8))
        
        for strategy_name, result in results.items():
            plt.plot(result['dates'], result['portfolio_values'], label=f"{strategy_name} ({result['total_return']:.2f}%)")
        
        plt.title('Trading Strategy Comparison')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def print_summary(self, results):
        """Print summary of results"""
        print("\n" + "="*50)
        print("TRADING STRATEGY COMPARISON RESULTS")
        print("="*50)
        
        for strategy_name, result in results.items():
            print(f"\n{strategy_name}:")
            print(f"  Initial Capital: ${self.initial_capital:,.2f}")
            print(f"  Final Value: ${result['final_value']:,.2f}")
            print(f"  Total Return: {result['total_return']:.2f}%")
            print(f"  Number of Trades: {len(result['trades'])}")

# Example usage
def main():
    # Initialize components
    data_provider = StockDataProvider()  # Uses demo key
    engine = BacktestEngine(data_provider)
    
    # Define strategies to test
    strategies = [
        RandomStrategy(buy_probability=0.05, sell_probability=0.05),
        BuyAndHoldStrategy(),
        SimpleMovingAverageStrategy(short_window=5, long_window=20),
        MomentumStrategy(lookback_period=10, momentum_threshold=0.02),
        RSIMomentumStrategy(rsi_period=14, oversold_threshold=30, overbought_threshold=70)
    ]
    
    # Run comparison
    symbol = "AAPL"  # Apple stock
    print(f"Running trading simulation for {symbol}...")
    
    results = engine.compare_strategies(symbol, strategies)
    
    if results:
        engine.print_summary(results)
        engine.plot_results(results)
    else:
        print("Failed to get data. Please check your API key or try again later.")

if __name__ == "__main__":
    main()