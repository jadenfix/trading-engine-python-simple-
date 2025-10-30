#!/usr/bin/env python3
"""
Real Simulated Trading System
A continuous paper trading system that runs with live market data, simulates order execution,
maintains positions, and tracks P&L in real-time.
"""

import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import logging
from collections import defaultdict
from typing import Dict, List, Optional
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulated_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SimulatedTrading')

class SimulatedTradingSystem:
    """
    Real simulated trading system that runs continuously with live market data.
    Features:
    - Live market data streaming
    - Real-time strategy signal generation
    - Simulated order execution with realistic fills
    - Position management and P&L tracking
    - Risk management and position limits
    - Real-time performance monitoring
    """

    def __init__(self, symbols: List[str], initial_capital: float = 100000.0):
        """
        Initialize the simulated trading system

        Args:
            symbols: List of symbols to trade
            initial_capital: Starting capital
        """
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.cash = initial_capital

        # Portfolio tracking
        self.positions = defaultdict(lambda: {'quantity': 0, 'avg_price': 0.0, 'unrealized_pnl': 0.0})
        self.portfolio_value = initial_capital

        # Order management
        self.pending_orders = []
        self.filled_orders = []
        self.order_id_counter = 1

        # Market data
        self.market_data = {}
        self.last_update = {}

        # Trading parameters
        self.commission_per_trade = 0.001  # 0.1%
        self.slippage_bps = 5  # 5 basis points
        self.max_position_size = 0.1  # 10% of portfolio per position
        self.max_total_risk = 0.05  # 5% max drawdown

        # Performance tracking
        self.performance_history = []
        self.daily_pnl = []
        self.trade_log = []

        # Control flags
        self.is_running = False
        self.market_open = False

        # Strategy state
        self.strategy_signals = defaultdict(dict)

        logger.info(f"Initialized Simulated Trading System with ${initial_capital:,.2f} capital")
        logger.info(f"Trading symbols: {symbols}")

    def start_trading(self):
        """Start the simulated trading system"""
        if self.is_running:
            logger.warning("Trading system already running")
            return

        logger.info("🚀 Starting Simulated Trading System...")
        self.is_running = True

        # Start background threads
        market_thread = threading.Thread(target=self._market_data_thread, daemon=True)
        strategy_thread = threading.Thread(target=self._strategy_thread, daemon=True)
        execution_thread = threading.Thread(target=self._execution_thread, daemon=True)
        monitoring_thread = threading.Thread(target=self._monitoring_thread, daemon=True)

        market_thread.start()
        strategy_thread.start()
        execution_thread.start()
        monitoring_thread.start()

        logger.info("✅ All trading threads started successfully")
        logger.info("📊 Trading system is now running...")

    def stop_trading(self):
        """Stop the simulated trading system"""
        logger.info("🛑 Stopping Simulated Trading System...")
        self.is_running = False
        time.sleep(2)  # Allow threads to clean up
        self._generate_final_report()
        logger.info("✅ Trading system stopped")

    def _market_data_thread(self):
        """Background thread for live market data fetching"""
        logger.info("📡 Starting market data thread...")

        while self.is_running:
            try:
                # Check if market is open (simplified - weekdays 9:30 AM - 4:00 PM EST)
                now = datetime.now()
                market_open = (now.weekday() < 5 and  # Monday-Friday
                             (now.hour > 9 or (now.hour == 9 and now.minute >= 30)) and
                             (now.hour < 16))

                self.market_open = market_open

                if market_open:
                    # Fetch live data for all symbols
                    for symbol in self.symbols:
                        try:
                            ticker = yf.Ticker(symbol)
                            # Get latest minute data
                            data = ticker.history(period="1d", interval="1m")

                            if not data.empty:
                                latest = data.iloc[-1]
                                self.market_data[symbol] = {
                                    'bid': latest['Close'] * 0.9995,  # Simulate bid
                                    'ask': latest['Close'] * 1.0005,  # Simulate ask
                                    'last': latest['Close'],
                                    'volume': latest['Volume'],
                                    'timestamp': latest.name.timestamp(),
                                    'high': latest['High'],
                                    'low': latest['Low']
                                }
                                self.last_update[symbol] = time.time()

                        except Exception as e:
                            logger.warning(f"Failed to fetch data for {symbol}: {e}")

                    # Update portfolio values
                    self._update_portfolio_values()

                else:
                    logger.debug("Market is closed")

            except Exception as e:
                logger.error(f"Market data thread error: {e}")

            # Sleep based on market status
            sleep_time = 60 if market_open else 300  # 1 min when open, 5 min when closed
            time.sleep(sleep_time)

    def _strategy_thread(self):
        """Background thread for strategy signal generation"""
        logger.info("🎯 Starting strategy thread...")

        while self.is_running:
            try:
                if self.market_open:
                    # Generate signals for each symbol
                    for symbol in self.symbols:
                        if symbol in self.market_data:
                            signal = self._generate_trading_signal(symbol)
                            if signal:
                                self.strategy_signals[symbol] = signal

                                # Submit order if signal is strong enough
                                if abs(signal['strength']) > 0.7:
                                    self._submit_order_from_signal(symbol, signal)

                else:
                    logger.debug("Strategy thread: Market closed")

            except Exception as e:
                logger.error(f"Strategy thread error: {e}")

            time.sleep(30)  # Check every 30 seconds

    def _execution_thread(self):
        """Background thread for order execution"""
        logger.info("⚡ Starting execution thread...")

        while self.is_running:
            try:
                if self.market_open:
                    # Process pending orders
                    orders_to_remove = []

                    for i, order in enumerate(self.pending_orders):
                        if self._execute_order(order):
                            orders_to_remove.append(i)
                            self.filled_orders.append(order)

                    # Remove executed orders (in reverse order to maintain indices)
                    for i in reversed(orders_to_remove):
                        self.pending_orders.pop(i)

                else:
                    logger.debug("Execution thread: Market closed")

            except Exception as e:
                logger.error(f"Execution thread error: {e}")

            time.sleep(10)  # Check every 10 seconds

    def _monitoring_thread(self):
        """Background thread for performance monitoring"""
        logger.info("📊 Starting monitoring thread...")

        while self.is_running:
            try:
                # Record performance metrics
                current_time = datetime.now()
                portfolio_value = self._calculate_portfolio_value()

                performance_record = {
                    'timestamp': current_time.isoformat(),
                    'portfolio_value': portfolio_value,
                    'cash': self.cash,
                    'total_positions': len([p for p in self.positions.values() if p['quantity'] != 0]),
                    'pnl': portfolio_value - self.initial_capital,
                    'pnl_percent': (portfolio_value - self.initial_capital) / self.initial_capital * 100,
                    'pending_orders': len(self.pending_orders),
                    'filled_orders_today': len([o for o in self.filled_orders
                                               if o['timestamp'].date() == current_time.date()])
                }

                self.performance_history.append(performance_record)

                # Log summary every 5 minutes
                if len(self.performance_history) % 5 == 0:
                    logger.info(".2f"
                               f"Cash: ${self.cash:,.2f}, Positions: {performance_record['total_positions']}")

                # Check risk limits
                self._check_risk_limits()

            except Exception as e:
                logger.error(f"Monitoring thread error: {e}")

            time.sleep(60)  # Update every minute

    def _generate_trading_signal(self, symbol: str) -> Optional[Dict]:
        """Generate trading signals for a symbol"""
        try:
            if symbol not in self.market_data:
                return None

            data = self.market_data[symbol]
            current_price = data['last']

            # Simple momentum strategy
            # Look for price movement over last few minutes (if available)
            signal_strength = 0.0

            # Volume-based signal
            volume = data.get('volume', 0)
            avg_volume = getattr(self, f'{symbol}_avg_volume', volume)
            setattr(self, f'{symbol}_avg_volume', (avg_volume + volume) / 2)

            if volume > avg_volume * 1.5:  # High volume
                signal_strength += 0.3

            # Price momentum
            last_price = getattr(self, f'{symbol}_last_price', current_price)
            setattr(self, f'{symbol}_last_price', current_price)

            price_change = (current_price - last_price) / last_price

            if price_change > 0.005:  # >0.5% up
                signal_strength += 0.4
            elif price_change < -0.005:  # >0.5% down
                signal_strength -= 0.4

            # Current position bias
            position = self.positions[symbol]
            if position['quantity'] > 0:
                signal_strength -= 0.1  # Bias against adding to long positions
            elif position['quantity'] < 0:
                signal_strength += 0.1  # Bias against adding to short positions

            if abs(signal_strength) > 0.3:
                return {
                    'symbol': symbol,
                    'signal': 'BUY' if signal_strength > 0 else 'SELL',
                    'strength': abs(signal_strength),
                    'price': current_price,
                    'timestamp': datetime.now()
                }

        except Exception as e:
            logger.error(f"Signal generation error for {symbol}: {e}")

        return None

    def _submit_order_from_signal(self, symbol: str, signal: Dict):
        """Submit an order based on a trading signal"""
        try:
            # Calculate position size
            portfolio_value = self._calculate_portfolio_value()
            max_position_value = portfolio_value * self.max_position_size

            # Current position
            current_position = self.positions[symbol]['quantity']
            current_price = signal['price']

            # Determine order size
            if signal['signal'] == 'BUY':
                if current_position < 0:  # Close short position
                    quantity = abs(current_position)
                else:  # Open new long position
                    quantity = int(max_position_value / current_price * 0.1)  # Conservative sizing
            else:  # SELL
                if current_position > 0:  # Close long position
                    quantity = current_position
                else:  # Open new short position
                    quantity = int(max_position_value / current_price * 0.1)

            if quantity > 0:
                order = {
                    'order_id': self.order_id_counter,
                    'symbol': symbol,
                    'side': signal['signal'],
                    'quantity': quantity,
                    'price': current_price,
                    'order_type': 'MARKET',
                    'timestamp': datetime.now(),
                    'status': 'PENDING'
                }

                self.pending_orders.append(order)
                self.order_id_counter += 1

                logger.info(f"📋 Submitted {signal['signal']} order for {quantity} {symbol} @ ${current_price:.2f}")

        except Exception as e:
            logger.error(f"Order submission error: {e}")

    def _execute_order(self, order: Dict) -> bool:
        """Execute a pending order with simulated fills"""
        try:
            symbol = order['symbol']
            side = order['side']
            quantity = order['quantity']

            if symbol not in self.market_data:
                return False

            market_data = self.market_data[symbol]

            # Simulate slippage and execution price
            if side == 'BUY':
                execution_price = market_data['ask'] * (1 + self.slippage_bps / 10000)
            else:
                execution_price = market_data['bid'] * (1 - self.slippage_bps / 10000)

            # Calculate commission
            commission = execution_price * quantity * self.commission_per_trade

            # Update cash and positions
            total_cost = execution_price * quantity + commission

            if side == 'BUY':
                if self.cash >= total_cost:
                    self.cash -= total_cost
                    self._update_position(symbol, quantity, execution_price)
                else:
                    logger.warning(f"Insufficient cash for {symbol} buy order")
                    return False
            else:  # SELL
                if self.positions[symbol]['quantity'] >= quantity:
                    self.cash += execution_price * quantity - commission
                    self._update_position(symbol, -quantity, execution_price)
                else:
                    logger.warning(f"Insufficient position for {symbol} sell order")
                    return False

            # Mark order as filled
            order['status'] = 'FILLED'
            order['execution_price'] = execution_price
            order['commission'] = commission
            order['fill_time'] = datetime.now()

            # Log trade
            self.trade_log.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': execution_price,
                'commission': commission,
                'pnl': 0.0  # Will be calculated later
            })

            logger.info(f"✅ Filled {side} order: {quantity} {symbol} @ ${execution_price:.2f} "
                       f"(Commission: ${commission:.2f})")

            return True

        except Exception as e:
            logger.error(f"Order execution error: {e}")
            return False

    def _update_position(self, symbol: str, quantity: int, price: float):
        """Update position after a trade"""
        position = self.positions[symbol]

        if position['quantity'] == 0:
            # Opening new position
            position['quantity'] = quantity
            position['avg_price'] = price
        else:
            # Adding to existing position
            total_quantity = position['quantity'] + quantity
            total_cost = position['quantity'] * position['avg_price'] + quantity * price
            position['avg_price'] = total_cost / total_quantity
            position['quantity'] = total_quantity

    def _update_portfolio_values(self):
        """Update unrealized P&L for all positions"""
        for symbol, position in self.positions.items():
            if position['quantity'] != 0 and symbol in self.market_data:
                current_price = self.market_data[symbol]['last']
                position['unrealized_pnl'] = position['quantity'] * (current_price - position['avg_price'])

    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        portfolio_value = self.cash

        for symbol, position in self.positions.items():
            if position['quantity'] != 0 and symbol in self.market_data:
                current_price = self.market_data[symbol]['last']
                portfolio_value += position['quantity'] * current_price

        return portfolio_value

    def _check_risk_limits(self):
        """Check and enforce risk limits"""
        portfolio_value = self._calculate_portfolio_value()
        pnl = portfolio_value - self.initial_capital
        pnl_percent = pnl / self.initial_capital

        # Check drawdown limit
        if pnl_percent < -self.max_total_risk:
            logger.warning(".2f")
            # Could implement position reduction logic here

    def get_status(self) -> Dict:
        """Get current system status"""
        portfolio_value = self._calculate_portfolio_value()

        positions_count = len([p for p in self.positions.values() if p['quantity'] != 0])

        return {
            'is_running': self.is_running,
            'market_open': self.market_open,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'total_pnl': portfolio_value - self.initial_capital,
            'pnl_percent': (portfolio_value - self.initial_capital) / self.initial_capital * 100,
            'positions': dict(self.positions),
            'positions_count': positions_count,
            'pending_orders': len(self.pending_orders),
            'total_trades': len(self.trade_log),
            'last_update': datetime.now().isoformat()
        }

    def _generate_final_report(self):
        """Generate final performance report"""
        if not self.performance_history:
            logger.info("No performance data to report")
            return

        # Calculate final metrics
        final_value = self.performance_history[-1]['portfolio_value']
        total_pnl = final_value - self.initial_capital
        total_return = total_pnl / self.initial_capital * 100

        # Calculate daily returns
        if len(self.performance_history) > 1:
            daily_returns = []
            prev_value = self.initial_capital

            for record in self.performance_history:
                daily_pnl = record['portfolio_value'] - prev_value
                daily_return = daily_pnl / prev_value
                daily_returns.append(daily_return)
                prev_value = record['portfolio_value']

            if daily_returns:
                avg_daily_return = np.mean(daily_returns)
                daily_volatility = np.std(daily_returns)
                sharpe_ratio = avg_daily_return / daily_volatility * np.sqrt(252) if daily_volatility > 0 else 0

        # Generate report
        report = {
            'simulation_summary': {
                'start_time': self.performance_history[0]['timestamp'],
                'end_time': self.performance_history[-1]['timestamp'],
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_pnl': total_pnl,
                'total_return_percent': total_return,
                'total_trades': len(self.trade_log),
                'win_rate': len([t for t in self.trade_log if t.get('pnl', 0) > 0]) / max(len(self.trade_log), 1),
                'sharpe_ratio': sharpe_ratio if 'sharpe_ratio' in locals() else 0,
            },
            'symbols_traded': self.symbols,
            'final_positions': dict(self.positions),
            'performance_history': self.performance_history[-100:],  # Last 100 records
            'trade_log': self.trade_log[-50:]  # Last 50 trades
        }

        # Save to file
        with open('simulated_trading_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("📊 Final performance report saved to simulated_trading_report.json")
        logger.info(".2f"
                   f"   Sharpe Ratio: {report['simulation_summary']['sharpe_ratio']:.2f}")
        logger.info(f"   Total Trades: {report['simulation_summary']['total_trades']}")

def main():
    """Main function to run the simulated trading system"""
    print("🤖 Real Simulated Trading System")
    print("=" * 50)
    print("A continuous paper trading system with live market data")
    print("=" * 50)

    # Configuration
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN']
    initial_capital = 100000.0
    run_duration_hours = 1  # Run for 1 hour for demo

    print(f"📊 Trading Symbols: {symbols}")
    print(f"💰 Initial Capital: ${initial_capital:,.2f}")
    print(f"⏱️  Run Duration: {run_duration_hours} hour(s)")
    print()

    # Initialize trading system
    trader = SimulatedTradingSystem(symbols, initial_capital)

    try:
        # Start trading
        trader.start_trading()

        # Run for specified duration
        end_time = time.time() + (run_duration_hours * 3600)

        print("🚀 Trading system is running...")
        print("Press Ctrl+C to stop early")

        while time.time() < end_time and trader.is_running:
            try:
                # Print status every 5 minutes
                status = trader.get_status()
                print(f"\r📊 Portfolio: ${status['portfolio_value']:,.2f} | "
                      f"P&L: ${status['total_pnl']:,.2f} ({status['pnl_percent']:+.2f}%) | "
                      f"Positions: {status['positions_count']} | "
                      f"Trades: {status['total_trades']} | "
                      f"Orders: {status['pending_orders']}", end="", flush=True)

                time.sleep(300)  # Update every 5 minutes

            except KeyboardInterrupt:
                print("\n⚠️  Received interrupt signal...")
                break

        print("\n\n🛑 Stopping trading system...")

    except Exception as e:
        print(f"\n❌ Error during trading: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Stop trading and generate report
        trader.stop_trading()

        # Final status
        final_status = trader.get_status()
        print("\n🏁 Final Results:")
        print(".2f")
        print(".2f")
        print(f"   Total Trades: {final_status['total_trades']}")
        print(f"   Open Positions: {final_status['positions_count']}")

if __name__ == "__main__":
    main()
