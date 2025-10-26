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


class RiskProfile:
    """Risk profile system for different risk tolerance levels"""

    def __init__(self, profile='medium'):
        """
        Initialize risk profile

        Args:
            profile (str): Risk level ('very_low', 'low', 'medium', 'high', 'very_high')
        """
        self.profile = profile.lower()
        self._set_profile_parameters()

    def _set_profile_parameters(self):
        """Set parameters based on risk profile"""
        profiles = {
            'very_low': {
                'max_risk_per_trade': 0.005,  # 0.5%
                'max_position_size': 0.05,    # 5%
                'stop_loss_pct': 0.02,        # 2%
                'take_profit_ratio': 3.0,
                'min_volatility_filter': 0.005,
                'max_drawdown_limit': 0.05
            },
            'low': {
                'max_risk_per_trade': 0.01,   # 1%
                'max_position_size': 0.08,    # 8%
                'stop_loss_pct': 0.03,        # 3%
                'take_profit_ratio': 2.5,
                'min_volatility_filter': 0.008,
                'max_drawdown_limit': 0.08
            },
            'medium': {
                'max_risk_per_trade': 0.02,   # 2%
                'max_position_size': 0.15,    # 15%
                'stop_loss_pct': 0.05,        # 5%
                'take_profit_ratio': 2.0,
                'min_volatility_filter': 0.01,
                'max_drawdown_limit': 0.12
            },
            'high': {
                'max_risk_per_trade': 0.05,   # 5%
                'max_position_size': 0.25,    # 25%
                'stop_loss_pct': 0.08,        # 8%
                'take_profit_ratio': 1.5,
                'min_volatility_filter': 0.015,
                'max_drawdown_limit': 0.20
            },
            'very_high': {
                'max_risk_per_trade': 0.10,   # 10%
                'max_position_size': 0.40,    # 40%
                'stop_loss_pct': 0.12,        # 12%
                'take_profit_ratio': 1.2,
                'min_volatility_filter': 0.02,
                'max_drawdown_limit': 0.30
            }
        }

        if self.profile not in profiles:
            self.profile = 'medium'

        self.params = profiles[self.profile]

    def get_position_size_multiplier(self):
        """Get position size multiplier for this risk profile"""
        return self.params['max_position_size']

    def get_stop_loss_pct(self):
        """Get stop loss percentage for this risk profile"""
        return self.params['stop_loss_pct']

    def should_trade_symbol(self, volatility):
        """Check if symbol meets minimum volatility requirement"""
        return volatility >= self.params['min_volatility_filter']

    def get_max_drawdown_limit(self):
        """Get maximum allowed drawdown for this risk profile"""
        return self.params['max_drawdown_limit']


class AdvancedRiskManager(RiskManager):
    """Advanced risk manager with profile-based risk management"""

    def __init__(self, max_risk_per_trade=0.02, max_position_size=0.1, stop_loss_pct=0.05, risk_profile='medium'):
        """
        Initialize advanced risk manager

        Args:
            max_risk_per_trade (float): Maximum risk per trade as fraction of total capital
            max_position_size (float): Maximum position size as fraction of total capital
            stop_loss_pct (float): Stop loss percentage
            risk_profile (str): Risk tolerance level
        """
        super().__init__(max_risk_per_trade, max_position_size, stop_loss_pct)
        self.risk_profile = RiskProfile(risk_profile)
        self._update_parameters_from_profile()

    def _update_parameters_from_profile(self):
        """Update parameters based on risk profile"""
        self.max_risk_per_trade = self.risk_profile.params['max_risk_per_trade']
        self.max_position_size = self.risk_profile.params['max_position_size']
        self.stop_loss_pct = self.risk_profile.params['stop_loss_pct']

    def calculate_position_size(self, capital, current_price, volatility=None, symbol=None):
        """
        Calculate position size with risk profile adjustments

        Args:
            capital (float): Total available capital
            current_price (float): Current asset price
            volatility (float, optional): Asset volatility for adjustment
            symbol (str, optional): Symbol for volatility filtering

        Returns:
            int: Number of shares to buy/sell
        """
        # Check minimum volatility requirement
        if volatility is not None and not self.risk_profile.should_trade_symbol(volatility):
            return 0

        # Get base position size
        base_size = super().calculate_position_size(capital, current_price, volatility)

        # Apply risk profile multiplier
        profile_multiplier = self.risk_profile.get_position_size_multiplier()
        adjusted_size = int(base_size * profile_multiplier * capital / current_price)

        return max(0, adjusted_size)

    def calculate_stop_loss(self, entry_price, side):
        """
        Calculate stop loss with risk profile adjustments

        Args:
            entry_price (float): Entry price of the trade
            side (str): 'long' or 'short'

        Returns:
            float: Stop loss price
        """
        profile_stop_loss = self.risk_profile.get_stop_loss_pct()
        return super().calculate_stop_loss(entry_price, side)

    def calculate_take_profit(self, entry_price, side, risk_reward_ratio=None):
        """
        Calculate take profit with risk profile adjustments

        Args:
            entry_price (float): Entry price of the trade
            side (str): 'long' or 'short'
            risk_reward_ratio (float, optional): Custom risk-reward ratio

        Returns:
            float: Take profit price
        """
        if risk_reward_ratio is None:
            risk_reward_ratio = self.risk_profile.params['take_profit_ratio']

        return super().calculate_take_profit(entry_price, side, risk_reward_ratio)

    def get_portfolio_heat(self, positions, current_prices, capital):
        """
        Calculate portfolio heat (risk exposure) based on risk profile

        Args:
            positions (dict): Current positions {symbol: shares}
            current_prices (dict): Current prices {symbol: price}
            capital (float): Total capital

        Returns:
            dict: Risk metrics including heat, drawdown, and risk level
        """
        total_risk = self.calculate_portfolio_risk(positions, current_prices, capital)
        max_drawdown = self.risk_profile.get_max_drawdown_limit()

        return {
            'total_risk': total_risk,
            'risk_level': 'high' if total_risk > max_drawdown * 0.8 else 'medium' if total_risk > max_drawdown * 0.5 else 'low',
            'max_drawdown_limit': max_drawdown,
            'risk_utilization': total_risk / max_drawdown if max_drawdown > 0 else 0
        }
