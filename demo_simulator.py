import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

class MockDataProvider:
    def __init__(self):
        pass
    
    def generate_mock_stock_data(self, days=100, initial_price=100):
        """Generate realistic mock stock data using geometric Brownian motion"""
        np.random.seed(42)  # For reproducible results
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                             end=datetime.now(), freq='H')[::6]  # Every 6 hours
        
        # Parameters for geometric Brownian motion
        dt = 1/252  # Daily time step
        mu = 0.05   # Expected annual return
        sigma = 0.2 # Annual volatility
        
        prices = [initial_price]
        for i in range(len(dates) - 1):
            # Generate random price movement
            drift = mu * dt
            shock = sigma * np.sqrt(dt) * np.random.normal()
            price = prices[-1] * np.exp(drift + shock)
            prices.append(price)
        
        # Create OHLC data
        data = []
        for i, price in enumerate(prices):
            noise = np.random.normal(0, 0.01)
            high = price * (1 + abs(noise))
            low = price * (1 - abs(noise))
            open_price = prices[i-1] if i > 0 else price
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': random.randint(100000, 1000000)
            })
        
        df = pd.DataFrame(data)
        df.index = dates[:len(df)]
        return df

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
        if not self.bought and current_idx == 0:
            self.bought = True
            return True
        return False
    
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
        prev_short_ma = data['close'].iloc[current_idx-self.short_window-1:current_idx-1].mean()
        prev_long_ma = data['close'].iloc[current_idx-self.long_window-1:current_idx-1].mean()
        
        # Golden cross: short MA crosses above long MA
        return short_ma > long_ma and prev_short_ma <= prev_long_ma
    
    def should_sell(self, data, current_idx):
        if current_idx < self.long_window:
            return False
        
        short_ma = data['close'].iloc[current_idx-self.short_window:current_idx].mean()
        long_ma = data['close'].iloc[current_idx-self.long_window:current_idx].mean()
        prev_short_ma = data['close'].iloc[current_idx-self.short_window-1:current_idx-1].mean()
        prev_long_ma = data['close'].iloc[current_idx-self.long_window-1:current_idx-1].mean()
        
        # Death cross: short MA crosses below long MA
        return short_ma < long_ma and prev_short_ma >= prev_long_ma

class BacktestEngine:
    def __init__(self, data_provider, initial_capital=10000):
        self.data_provider = data_provider
        self.initial_capital = initial_capital
    
    def run_backtest(self, data, strategy, shares_per_trade=10):
        """Run backtest for a given strategy"""
        simulator = TradingSimulator(self.initial_capital)
        
        for i in range(1, len(data)):
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
    
    def compare_strategies(self, data, strategies):
        """Compare multiple strategies"""
        results = {}
        
        for strategy in strategies:
            print(f"Running backtest for {strategy.name}...")
            result = self.run_backtest(data, strategy)
            results[strategy.name] = result
        
        return results
    
    def plot_results(self, results):
        """Plot comparison of strategies"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot portfolio values
        for strategy_name, result in results.items():
            ax1.plot(result['dates'], result['portfolio_values'], 
                    label=f"{strategy_name} ({result['total_return']:.2f}%)", linewidth=2)
        
        ax1.set_title('Trading Strategy Comparison - Portfolio Value Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot stock price
        data = list(results.values())[0]['data']
        ax2.plot(data.index, data['close'], label='Stock Price', color='black', alpha=0.7)
        ax2.set_title('Stock Price Over Time')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('trading_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def print_summary(self, results):
        """Print summary of results"""
        print("\n" + "="*60)
        print("TRADING STRATEGY COMPARISON RESULTS")
        print("="*60)
        
        # Sort by total return
        sorted_results = sorted(results.items(), key=lambda x: x[1]['total_return'], reverse=True)
        
        for i, (strategy_name, result) in enumerate(sorted_results):
            print(f"\n{i+1}. {strategy_name}:")
            print(f"   Initial Capital: ${self.initial_capital:,.2f}")
            print(f"   Final Value: ${result['final_value']:,.2f}")
            print(f"   Total Return: {result['total_return']:.2f}%")
            print(f"   Number of Trades: {len(result['trades'])}")
            
            if result['trades']:
                buy_trades = [t for t in result['trades'] if t['action'] == 'BUY']
                sell_trades = [t for t in result['trades'] if t['action'] == 'SELL']
                print(f"   Buy Trades: {len(buy_trades)}")
                print(f"   Sell Trades: {len(sell_trades)}")
        
        print("\n" + "="*60)
        best_strategy = sorted_results[0][0]
        best_return = sorted_results[0][1]['total_return']
        print(f"🏆 WINNER: {best_strategy} with {best_return:.2f}% return")
        print("="*60)

def main():
    # Initialize components
    data_provider = MockDataProvider()
    engine = BacktestEngine(data_provider)
    
    # Generate mock stock data
    print("Generating mock stock data...")
    stock_data = data_provider.generate_mock_stock_data(days=60, initial_price=150)
    
    # Define strategies to test
    strategies = [
        RandomStrategy(buy_probability=0.1, sell_probability=0.1),
        BuyAndHoldStrategy(),
        SimpleMovingAverageStrategy(short_window=5, long_window=15)
    ]
    
    print("Running trading simulation with mock data...")
    print(f"Data period: {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Starting price: ${stock_data['close'].iloc[0]:.2f}")
    print(f"Ending price: ${stock_data['close'].iloc[-1]:.2f}")
    
    results = engine.compare_strategies(stock_data, strategies)
    
    if results:
        engine.print_summary(results)
        engine.plot_results(results)
    else:
        print("Failed to run simulation.")

if __name__ == "__main__":
    main()