"""
Risk management for trading algorithm
"""
import numpy as np


class RiskManager:
    """Manages risk and position sizing for trading"""

    def __init__(self, max_risk_per_trade=0.02, max_position_size=0.1, stop_loss_pct=0.05):
        """
        Initialize risk manager

        Args:
            max_risk_per_trade (float): Maximum risk per trade as fraction of total capital (0.02 = 2%)
            max_position_size (float): Maximum position size as fraction of total capital (0.1 = 10%)
            stop_loss_pct (float): Stop loss percentage (0.05 = 5%)
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct

    def calculate_position_size(self, capital, current_price, volatility=None):
        """
        Calculate position size based on risk management rules

        Args:
            capital (float): Total available capital
            current_price (float): Current asset price
            volatility (float, optional): Asset volatility for adjustment

        Returns:
            int: Number of shares to buy/sell
        """
        # Base position size
        max_position_value = capital * self.max_position_size
        base_position_size = max_position_value / current_price

        # Adjust for risk per trade
        risk_adjusted_size = (capital * self.max_risk_per_trade) / (current_price * self.stop_loss_pct)

        # Use the more conservative of the two
        position_size = min(base_position_size, risk_adjusted_size)

        # Adjust for volatility if provided
        if volatility:
            volatility_adjustment = 1.0 / (1.0 + volatility)
            position_size *= volatility_adjustment

        return int(position_size)

    def calculate_stop_loss(self, entry_price, side):
        """
        Calculate stop loss price

        Args:
            entry_price (float): Entry price of the trade
            side (str): 'long' or 'short'

        Returns:
            float: Stop loss price
        """
        if side == 'long':
            return entry_price * (1 - self.stop_loss_pct)
        else:  # short
            return entry_price * (1 + self.stop_loss_pct)

    def calculate_take_profit(self, entry_price, side, risk_reward_ratio=2.0):
        """
        Calculate take profit price

        Args:
            entry_price (float): Entry price of the trade
            side (str): 'long' or 'short'
            risk_reward_ratio (float): Risk to reward ratio

        Returns:
            float: Take profit price
        """
        if side == 'long':
            return entry_price * (1 + (self.stop_loss_pct * risk_reward_ratio))
        else:  # short
            return entry_price * (1 - (self.stop_loss_pct * risk_reward_ratio))

    def should_close_position(self, current_price, entry_price, side, stop_loss=None, take_profit=None):
        """
        Check if position should be closed based on stop loss or take profit

        Args:
            current_price (float): Current market price
            entry_price (float): Entry price of the position
            side (str): 'long' or 'short'
            stop_loss (float, optional): Custom stop loss price
            take_profit (float, optional): Custom take profit price

        Returns:
            str: 'stop_loss', 'take_profit', or None
        """
        if stop_loss is None:
            stop_loss = self.calculate_stop_loss(entry_price, side)
        if take_profit is None:
            take_profit = self.calculate_take_profit(entry_price, side)

        if side == 'long':
            if current_price <= stop_loss:
                return 'stop_loss'
            elif current_price >= take_profit:
                return 'take_profit'
        else:  # short
            if current_price >= stop_loss:
                return 'stop_loss'
            elif current_price <= take_profit:
                return 'take_profit'

        return None

    def calculate_portfolio_risk(self, positions, current_prices, capital):
        """
        Calculate total portfolio risk

        Args:
            positions (dict): Current positions {symbol: shares}
            current_prices (dict): Current prices {symbol: price}
            capital (float): Total capital

        Returns:
            float: Total portfolio risk as fraction of capital
        """
        total_risk = 0.0

        for symbol, shares in positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position_value = abs(shares) * current_price

                # Risk is position value * stop loss percentage
                position_risk = position_value * self.stop_loss_pct
                total_risk += position_risk

        return total_risk / capital if capital > 0 else 0.0
