# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the System
- **Main simulator with Alpha Vantage API**: `python stock_simulator.py`
- **Demo simulator with mock data**: `python demo_simulator.py`
- **Install dependencies**: `pip install -r requirements.txt` or `uv pip install -r requirements.txt`

### Testing
- No formal test suite currently exists
- Verify functionality by running simulations and checking output plots/results

### Development Tracking & Workflow
- **Development Tracking**: `python -c "from dev_tracker import track_development_change; track_development_change('user_prompt', 'description')"`
- **Push to Remote**: `python -c "from dev_tracker import push_to_remote; push_to_remote()"`
- **Branch Workflow**: All changes automatically create feature branches (never direct commits to master)
- **Sound Notifications**: Completion sounds play when tasks finish (helps with multitasking)
- **Change Logs**: All development changes tracked in `change_logs/` with timestamps, session IDs, and user prompts

## Architecture Overview

This is a stock trading simulation and backtesting system with a modular strategy pattern:

### Core Components
1. **Data Providers**: 
   - `StockDataProvider` (stock_simulator.py) - Fetches real data from Alpha Vantage API
   - `MockDataProvider` (demo_simulator.py) - Generates synthetic stock data using geometric Brownian motion

2. **Trading Engine**:
   - `TradingSimulator` - Manages portfolio state, buy/sell operations, and trade tracking
   - `BacktestEngine` - Orchestrates backtesting across multiple strategies and generates comparison results

3. **Strategy Framework**:
   - `TradingStrategy` (abstract base class) - Defines `should_buy()` and `should_sell()` interface
   - Built-in strategies: `RandomStrategy`, `BuyAndHoldStrategy`, `SimpleMovingAverageStrategy`

### Key Design Patterns
- **Strategy Pattern**: All trading strategies inherit from `TradingStrategy` and implement buy/sell decision logic
- **Data Provider Pattern**: Abstracts data sources (real API vs mock data) behind common interface
- **Simulation Pattern**: `TradingSimulator` maintains portfolio state and processes trading decisions sequentially

### Data Flow
1. Data provider fetches/generates OHLC price data as pandas DataFrame
2. BacktestEngine runs strategies against historical data chronologically
3. For each time step, strategies evaluate `should_buy()` and `should_sell()`
4. TradingSimulator executes trades and tracks portfolio value
5. Results aggregated for comparison and visualization

### Extension Points
- **New Trading Strategies**: Extend `TradingStrategy` class and implement decision logic
- **New Data Sources**: Implement provider following `StockDataProvider` interface
- **Additional Metrics**: Extend result dictionary in `BacktestEngine.run_backtest()`

### Dependencies
- pandas/numpy for data manipulation
- matplotlib for visualization
- requests for API calls (Alpha Vantage)
- alpha-vantage package for market data

### API Limitations
- Alpha Vantage free tier: 5 calls/minute, 500 calls/day
- Demo mode uses "demo" API key with limited recent data
- Rate limiting implemented (1 second delay between strategy runs)