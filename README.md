# Advanced Quantitative Research Framework

A comprehensive, production-ready quantitative trading research framework featuring unconventional strategies, advanced risk management, and cross-market signal analysis.

## Features

### Unconventional Quantitative Strategies
- **Attention-Driven Strategy**: Exploits investor attention patterns and market psychology
- **Sentiment Regime Strategy**: Uses behavioral biases (herding, anchoring) for regime detection
- **Information Theory Strategy**: Applies entropy measures and transfer entropy
- **Complex Systems Strategy**: Network centrality and contagion analysis
- **Fractal Chaos Strategy**: Fractal dimension and Lyapunov exponents
- **Quantum-Inspired Strategy**: Quantum mechanics concepts for market analysis

### Advanced Ensemble System
- **Regime-Dependent Allocation**: Automatically adapts to market volatility, trend, and sentiment regimes
- **Risk Parity**: Equal risk contribution across strategies
- **Performance Momentum**: Boosts allocation to recently successful strategies
- **Dynamic Rebalancing**: Daily regime-based portfolio reweighting

### Comprehensive Backtesting Engine
- **Realistic Transaction Costs**: Commission and multiple slippage models
- **Position Sizing**: Risk-adjusted position sizing with constraints
- **Performance Metrics**: Sharpe ratio, Sortino, Calmar, VaR, CVaR, drawdown analysis
- **Walk-Forward Optimization**: Out-of-sample testing and parameter stability

### Advanced Risk Management
- **CVaR (Conditional VaR)**: Expected shortfall under extreme scenarios
- **Drawdown Control**: Maximum drawdown limits with recovery tracking
- **Kelly Criterion**: Optimal position sizing based on win probability
- **Stress Testing**: Portfolio testing under various market scenarios
- **Dynamic Risk Budgeting**: Risk parity allocation across strategies

### Cross-Market Signal Analysis
- **Inter-Market Correlations**: Relationships between equities, FX, commodities, crypto
- **Lead-Lag Analysis**: Detect which markets lead/follow others
- **FX Impact Analysis**: Currency movement effects on equity markets
- **Commodity Influence**: Sector impacts from commodity prices
- **Global Economic Signals**: GDP, inflation, interest rates

### Adaptive Parameter Optimization
- **Bayesian Optimization**: Efficient parameter search using Gaussian processes
- **Walk-Forward Analysis**: Out-of-sample parameter validation
- **Regime-Adaptive Parameters**: Different parameters for different market conditions
- **Genetic Algorithms**: Evolutionary parameter optimization

## Project Structure

```
research/
├── strategies.py                 # Original quantitative strategies
├── unconventional_strategies.py  # New behavioral & complex systems strategies
├── runner.py                     # Main research framework runner
├── backtesting_engine.py         # Advanced backtesting with costs
├── strategy_ensemble.py          # Multi-strategy ensemble system
├── risk_manager.py               # Advanced risk management
├── cross_market_signals.py       # Cross-market analysis
├── adaptive_optimizer.py         # Parameter optimization
├── correlation_analyzer.py       # Correlation analysis tools
└── __init__.py

trading/
├── strategies.py                 # Base strategy classes
├── algorithm.py                  # Trading algorithm implementation
├── backtesting.py                # Basic backtesting
├── data_fetcher.py               # Market data fetching
└── risk_manager.py               # Basic risk management
   ```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd advanced-quant-research

# Install dependencies
pip install -r requirements.txt

# Activate virtual environment (recommended)
python -m venv trading_env
source trading_env/bin/activate  # On Windows: trading_env\Scripts\activate
pip install -r requirements.txt
```

### 2. Basic Usage

```python
from research.runner import run_research_analysis

# Run comprehensive analysis on major tech stocks
symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
results = run_research_analysis(symbols, 'comprehensive')

# Analyze with specific unconventional strategy
attention_results = run_research_analysis(symbols, 'attention')
sentiment_results = run_research_analysis(symbols, 'sentiment')
fractal_results = run_research_analysis(symbols, 'fractal_chaos')
```

### 3. Advanced Ensemble Trading

```python
from research.strategy_ensemble import create_unconventional_ensemble
from research.backtesting_engine import BacktestingEngine

# Create and configure ensemble
ensemble = create_unconventional_ensemble()

# Setup backtesting with realistic costs
backtest_engine = BacktestingEngine(
    initial_capital=100000,
    commission_per_trade=0.001,
    slippage_bps=5
)

# Run ensemble backtest
price_data = {}  # Load your price data
signals = ensemble.generate_ensemble_signals(price_data, current_date)
results = backtest_engine.run_backtest(signals, price_data)
```

### 4. Risk Management

```python
from research.risk_manager import RiskManager

# Initialize risk manager
risk_manager = RiskManager(
    confidence_level=0.95,
    max_drawdown_limit=0.20
)

# Calculate portfolio risk metrics
var_metrics = risk_manager.calculate_var(portfolio_returns)
kelly_size = risk_manager.calculate_kelly_criterion(portfolio_returns)
drawdown = risk_manager.monitor_drawdown(portfolio_values)

# Generate comprehensive risk report
risk_report = risk_manager.generate_risk_report(portfolio_history)
```

### 5. Parameter Optimization

```python
from research.adaptive_optimizer import optimize_strategy_parameters
from research.unconventional_strategies import AttentionDrivenStrategy

# Optimize strategy parameters
best_params = optimize_strategy_parameters(
    AttentionDrivenStrategy,
    parameter_bounds={'attention_lookback': (10, 50), 'attention_threshold': (1.0, 2.0)},
    price_data_dict=price_data,
    start_date=start_date,
    end_date=end_date,
    optimization_method='bayesian'
)
```

## Available Strategies

### Traditional Strategies
- **Factor Momentum**: Cross-sectional factor analysis
- **Cross-Sectional Momentum**: Statistical momentum with volume filters
- **Volatility Regime**: Mean reversion vs momentum based on volatility
- **Liquidity Timing**: Order flow and volume-based timing
- **Statistical Process Control**: Control charts for regime detection

### Unconventional Strategies
- **Attention-Driven**: Investor attention patterns and spikes
- **Sentiment Regime**: Behavioral biases and market psychology
- **Information Theory**: Entropy and transfer entropy signals
- **Complex Systems**: Network effects and contagion analysis
- **Fractal Chaos**: Fractal geometry and chaos theory
- **Quantum-Inspired**: Quantum mechanics market analogies

## Strategy Analysis Types

```python
# Individual strategy analysis
run_research_analysis(symbols, 'attention')        # Attention-driven
run_research_analysis(symbols, 'sentiment')        # Sentiment regime
run_research_analysis(symbols, 'info_theory')      # Information theory
run_research_analysis(symbols, 'complex_systems')  # Complex systems
run_research_analysis(symbols, 'fractal_chaos')    # Fractal chaos
run_research_analysis(symbols, 'quantum')          # Quantum-inspired

# Traditional strategies
run_research_analysis(symbols, 'factor')           # Factor momentum
run_research_analysis(symbols, 'momentum')         # Cross-sectional momentum
run_research_analysis(symbols, 'volatility')       # Volatility regime
run_research_analysis(symbols, 'liquidity')        # Liquidity timing
run_research_analysis(symbols, 'spc')              # Statistical process control

# Comprehensive analysis
run_research_analysis(symbols, 'comprehensive')    # All strategies
run_research_analysis(symbols, 'correlation')      # Correlation analysis only
```

## Performance Metrics

The framework provides comprehensive performance analysis:

- **Risk-Adjusted Returns**: Sharpe, Sortino, Calmar ratios
- **Risk Metrics**: VaR, CVaR, maximum drawdown, recovery time
- **Position Sizing**: Kelly criterion, risk-adjusted sizing
- **Stress Testing**: Portfolio performance under various scenarios
- **Strategy Attribution**: Individual strategy contribution analysis

## Configuration

### Backtesting Parameters
```python
backtest_config = {
    'initial_capital': 100000,
    'commission_per_trade': 0.001,  # 0.1% commission
    'slippage_bps': 5,              # 5 basis points slippage
    'slippage_model': 'fixed',      # 'fixed', 'volume_based', 'adaptive'
    'max_position_size': 0.1,       # 10% max position
    'risk_free_rate': 0.02          # 2% risk-free rate
}
```

### Strategy Parameters
```python
# Attention-Driven Strategy
attention_params = {
    'attention_lookback': 21,
    'attention_threshold': 1.5,
    'volume_multiplier': 2.0
}

# Fractal Chaos Strategy
fractal_params = {
    'fractal_window': 200,
    'hurst_lookback': 100,
    'chaos_threshold': 0.1
}
```

## Risk Management Features

### Portfolio-Level Controls
- Maximum drawdown limits
- Portfolio volatility targets
- Position concentration limits
- Stress testing scenarios

### Strategy-Level Controls
- Individual strategy risk budgets
- Correlation-based diversification
- Dynamic position sizing
- Regime-based risk adjustments

## Cross-Market Integration

### Market Data Requirements
```python
# Add different market data
analyzer = CrossMarketAnalyzer()
analyzer.add_market_data('equity', 'AAPL', apple_data)
analyzer.add_market_data('fx', 'EURUSD', fx_data)
analyzer.add_market_data('commodity', 'GC=F', gold_data)
analyzer.add_market_data('crypto', 'BTC-USD', bitcoin_data)
```

### Cross-Market Signals
- **FX-Equity Relationships**: Currency impacts on stock performance
- **Commodity-Sector Links**: Oil/gold effects on energy/materials stocks
- **Crypto Correlations**: Digital asset relationships with traditional markets
- **Economic Indicators**: GDP, inflation, rates impact on equity markets

## Adaptive Optimization

### Bayesian Optimization
```python
from research.adaptive_optimizer import BayesianOptimizer

optimizer = BayesianOptimizer(parameter_bounds={
    'lookback': (10, 100),
    'threshold': (0.5, 3.0)
})

best_params = optimizer.optimize(objective_function)
```

### Walk-Forward Analysis
```python
from research.adaptive_optimizer import WalkForwardOptimizer

wf_optimizer = WalkForwardOptimizer(
    optimization_window=252,  # 1 year in-sample
    validation_window=63,     # 3 months out-of-sample
    step_size=21              # Monthly re-optimization
)

results = wf_optimizer.optimize_strategy(strategy_class, param_bounds, price_data)
```

## Results Interpretation

### Strategy Signals
- **1**: Strong buy signal
- **-1**: Strong sell signal
- **0**: No signal/neutral

### Ensemble Weights
- Dynamically adjusted based on market regime
- Risk parity allocation across strategies
- Performance momentum adjustments

### Risk Metrics
- **VaR**: Potential loss at confidence level
- **CVaR**: Expected loss beyond VaR
- **Kelly Fraction**: Optimal position size
- **Drawdown**: Peak-to-trough decline

## Development Workflow

1. **Research**: Analyze new strategies using the framework
2. **Backtest**: Test strategies with realistic costs and slippage
3. **Optimize**: Use Bayesian/walk-forward optimization for parameters
4. **Risk Manage**: Apply comprehensive risk controls
5. **Ensemble**: Combine strategies with regime-based allocation
6. **Deploy**: Implement live trading with risk limits

## Example Results

```python
# Comprehensive analysis results
{
    'strategy_performance': {
        'total_return': 0.234,      # 23.4% return
        'sharpe_ratio': 1.85,       # Excellent risk-adjusted return
        'max_drawdown': -0.123,     # 12.3% maximum drawdown
        'win_rate': 0.58           # 58% winning trades
    },
    'risk_metrics': {
        'var_95': -0.034,          # 3.4% VaR at 95% confidence
        'cvar_95': -0.052,         # 5.2% expected shortfall
        'kelly_fraction': 0.085    # 8.5% optimal position size
    },
    'ensemble_weights': {
        'attention_driven': 0.22,
        'sentiment_regime': 0.18,
        'fractal_chaos': 0.15,
        'information_theory': 0.12,
        'complex_systems': 0.10,
        'quantum_inspired': 0.08,
        'volatility_regime': 0.15
    }
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your strategy or enhancement
4. Add comprehensive tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This framework is for research and educational purposes. Not intended for live trading without thorough validation and risk management. Past performance does not guarantee future results.

---

**Ready to revolutionize your quantitative trading approach?**