# Stock Trading AI Evaluation System

A comprehensive stock trading simulation system that allows you to evaluate AI trading strategies against random trading and other baselines using historical market data.

## Features

- **Historical Data Fetching**: Uses Alpha Vantage API for real stock market data
- **Multiple Trading Strategies**: Random, Buy-and-Hold, Moving Average, and extensible framework for AI strategies
- **Backtesting Engine**: Complete simulation with portfolio tracking and performance metrics
- **Comparison Tools**: Side-by-side strategy comparison with visualization
- **Evaluation Metrics**: Total return, number of trades, portfolio value over time

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the simulation:
```bash
python stock_simulator.py
```

3. For better results, get a free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key) and modify the code:
```python
data_provider = StockDataProvider(api_key="YOUR_API_KEY")
```

## Usage Example

```python
from stock_simulator import *

# Initialize
data_provider = StockDataProvider()
engine = BacktestEngine(data_provider, initial_capital=10000)

# Define strategies
strategies = [
    RandomStrategy(buy_probability=0.05, sell_probability=0.05),
    BuyAndHoldStrategy(),
    SimpleMovingAverageStrategy(short_window=5, long_window=20)
]

# Run comparison
results = engine.compare_strategies("AAPL", strategies)
engine.print_summary(results)
engine.plot_results(results)
```

## Adding Your AI Strategy

Create a custom strategy by extending the `TradingStrategy` class:

```python
class MyAIStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("My AI Strategy")
    
    def should_buy(self, data, current_idx):
        # Your AI logic here
        return your_buy_decision
    
    def should_sell(self, data, current_idx):
        # Your AI logic here
        return your_sell_decision
```

## API Limitations

- Free Alpha Vantage API: 5 calls per minute, 500 calls per day
- Demo key provides limited recent data
- For production use, consider upgrading to a paid plan