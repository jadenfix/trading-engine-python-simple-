"""
Cross-Market Signal Analysis System

This module analyzes signals across different asset classes and markets to find:
- Inter-market correlations and lead-lag relationships
- FX impacts on equities
- Commodity influences on stocks
- Crypto market signals
- Global economic indicators
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')


class CrossMarketAnalyzer:
    """Analyzes relationships between different markets and asset classes"""

    def __init__(self, correlation_window=252, lead_lag_max_lags=10):
        """
        Initialize cross-market analyzer

        Args:
            correlation_window (int): Window for correlation analysis
            lead_lag_max_lags (int): Maximum lags for lead-lag analysis
        """
        self.correlation_window = correlation_window
        self.lead_lag_max_lags = lead_lag_max_lags

        # Market data containers
        self.equity_data = {}
        self.fx_data = {}
        self.commodity_data = {}
        self.crypto_data = {}
        self.economic_data = {}

    def add_market_data(self, market_type, symbol, data):
        """
        Add market data for analysis

        Args:
            market_type (str): 'equity', 'fx', 'commodity', 'crypto', 'economic'
            symbol (str): Market symbol/ticker
            data (pd.DataFrame): Price/return data
        """
        if market_type == 'equity':
            self.equity_data[symbol] = data
        elif market_type == 'fx':
            self.fx_data[symbol] = data
        elif market_type == 'commodity':
            self.commodity_data[symbol] = data
        elif market_type == 'crypto':
            self.crypto_data[symbol] = data
        elif market_type == 'economic':
            self.economic_data[symbol] = data

    def analyze_inter_market_correlations(self, current_date, min_correlation=0.3):
        """
        Analyze correlations between different markets

        Args:
            current_date (pd.Timestamp): Current date for analysis
            min_correlation (float): Minimum correlation threshold

        Returns:
            dict: Inter-market correlation analysis
        """
        # Get all market data
        all_markets = {
            'equity': self.equity_data,
            'fx': self.fx_data,
            'commodity': self.commodity_data,
            'crypto': self.crypto_data,
            'economic': self.economic_data
        }

        correlations = {}
        significant_relationships = []

        # Calculate correlations between all market pairs
        market_types = list(all_markets.keys())

        for i, market1_type in enumerate(market_types):
            for market2_type in market_types[i+1:]:
                market1_data = all_markets[market1_type]
                market2_data = all_markets[market2_type]

                # Find common symbols and date range
                market1_symbols = set(market1_data.keys())
                market2_symbols = set(market2_data.keys())

                for symbol1 in market1_symbols:
                    for symbol2 in market2_symbols:
                        try:
                            data1 = market1_data[symbol1]
                            data2 = market2_data[symbol2]

                            # Get overlapping date range
                            common_dates = data1.index.intersection(data2.index)
                            if len(common_dates) < self.correlation_window:
                                continue

                            recent_data1 = data1.loc[common_dates].tail(self.correlation_window)
                            recent_data2 = data2.loc[common_dates].tail(self.correlation_window)

                            # Calculate returns if price data
                            if 'Close' in recent_data1.columns:
                                returns1 = recent_data1['Close'].pct_change().dropna()
                            else:
                                returns1 = recent_data1.iloc[:, 0].pct_change().dropna()

                            if 'Close' in recent_data2.columns:
                                returns2 = recent_data2['Close'].pct_change().dropna()
                            else:
                                returns2 = recent_data2.iloc[:, 0].pct_change().dropna()

                            # Align data
                            common_return_dates = returns1.index.intersection(returns2.index)
                            if len(common_return_dates) < 30:
                                continue

                            aligned_returns1 = returns1.loc[common_return_dates]
                            aligned_returns2 = returns2.loc[common_return_dates]

                            # Calculate correlation
                            corr, p_value = pearsonr(aligned_returns1, aligned_returns2)

                            correlations[f"{market1_type}_{symbol1}_{market2_type}_{symbol2}"] = {
                                'correlation': corr,
                                'p_value': p_value,
                                'sample_size': len(aligned_returns1),
                                'significant': abs(corr) > min_correlation and p_value < 0.05
                            }

                            if abs(corr) > min_correlation and p_value < 0.05:
                                significant_relationships.append({
                                    'market1': f"{market1_type}_{symbol1}",
                                    'market2': f"{market2_type}_{symbol2}",
                                    'correlation': corr,
                                    'p_value': p_value,
                                    'direction': 'positive' if corr > 0 else 'negative'
                                })

                        except Exception as e:
                            continue

        return {
            'all_correlations': correlations,
            'significant_relationships': significant_relationships,
            'summary': {
                'total_relationships_analyzed': len(correlations),
                'significant_relationships': len(significant_relationships),
                'strongest_positive': max([r['correlation'] for r in significant_relationships], default=0),
                'strongest_negative': min([r['correlation'] for r in significant_relationships], default=0)
            }
        }

    def analyze_lead_lag_relationships(self, current_date):
        """
        Analyze lead-lag relationships between markets

        Args:
            current_date (pd.Timestamp): Current date for analysis

        Returns:
            dict: Lead-lag analysis results
        """
        lead_lag_results = {}

        # Analyze relationships between key market pairs
        key_pairs = [
            ('equity', 'fx'),
            ('equity', 'commodity'),
            ('fx', 'commodity'),
            ('equity', 'crypto'),
            ('commodity', 'crypto')
        ]

        for market1_type, market2_type in key_pairs:
            market1_data = getattr(self, f"{market1_type}_data")
            market2_data = getattr(self, f"{market2_type}_data")

            pair_results = []

            for symbol1 in market1_data.keys():
                for symbol2 in market2_data.keys():
                    try:
                        data1 = market1_data[symbol1]
                        data2 = market2_data[symbol2]

                        # Get overlapping returns
                        returns1 = data1['Close'].pct_change().dropna()
                        returns2 = data2['Close'].pct_change().dropna()

                        common_dates = returns1.index.intersection(returns2.index)
                        if len(common_dates) < self.correlation_window:
                            continue

                        aligned_returns1 = returns1.loc[common_dates].tail(self.correlation_window)
                        aligned_returns2 = returns2.loc[common_dates].tail(self.correlation_window)

                        # Test lead-lag relationships
                        max_corr = -1
                        best_lag = 0
                        leading_market = None

                        for lag in range(-self.lead_lag_max_lags, self.lead_lag_max_lags + 1):
                            if lag < 0:
                                # Market 2 leads market 1
                                lagged_returns2 = aligned_returns2.shift(-lag)
                                common_idx = aligned_returns1.index.intersection(lagged_returns2.dropna().index)
                                if len(common_idx) > 30:
                                    corr, _ = pearsonr(aligned_returns1.loc[common_idx], lagged_returns2.loc[common_idx])
                                    if abs(corr) > abs(max_corr):
                                        max_corr = corr
                                        best_lag = lag
                                        leading_market = f"{market2_type}_{symbol2}"
                            elif lag > 0:
                                # Market 1 leads market 2
                                lagged_returns1 = aligned_returns1.shift(lag)
                                common_idx = aligned_returns2.index.intersection(lagged_returns1.dropna().index)
                                if len(common_idx) > 30:
                                    corr, _ = pearsonr(aligned_returns2.loc[common_idx], lagged_returns1.loc[common_idx])
                                    if abs(corr) > abs(max_corr):
                                        max_corr = corr
                                        best_lag = lag
                                        leading_market = f"{market1_type}_{symbol1}"

                        if abs(max_corr) > 0.3:  # Significant lead-lag relationship
                            pair_results.append({
                                'pair': f"{market1_type}_{symbol1} vs {market2_type}_{symbol2}",
                                'leading_market': leading_market,
                                'lag_days': best_lag,
                                'correlation': max_corr,
                                'direction': 'positive' if max_corr > 0 else 'negative'
                            })

                    except Exception as e:
                        continue

            lead_lag_results[f"{market1_type}_vs_{market2_type}"] = pair_results

        return lead_lag_results

    def generate_cross_market_signals(self, current_date, equity_signals):
        """
        Generate enhanced signals incorporating cross-market analysis

        Args:
            current_date (pd.Timestamp): Current date
            equity_signals (dict): Original equity signals to enhance

        Returns:
            dict: Enhanced signals with cross-market information
        """
        enhanced_signals = equity_signals.copy()

        # Analyze current market conditions
        correlations = self.analyze_inter_market_correlations(current_date)
        lead_lag = self.analyze_lead_lag_relationships(current_date)

        # Enhance signals based on cross-market relationships
        for symbol in list(enhanced_signals.keys()):
            if symbol in self.equity_data:
                signal_enhancements = []

                # Check FX relationships
                for fx_symbol in self.fx_data.keys():
                    fx_relationships = [r for r in correlations['significant_relationships']
                                      if f"equity_{symbol}" in r['market1'] and f"fx_{fx_symbol}" in r['market2']]

                    if fx_relationships:
                        strongest_fx = max(fx_relationships, key=lambda x: abs(x['correlation']))
                        signal_enhancements.append({
                            'type': 'fx_correlation',
                            'market': fx_symbol,
                            'correlation': strongest_fx['correlation'],
                            'signal_boost': 0.1 if strongest_fx['correlation'] > 0.5 else 0
                        })

                # Check commodity relationships
                for comm_symbol in self.commodity_data.keys():
                    comm_relationships = [r for r in correlations['significant_relationships']
                                        if f"equity_{symbol}" in r['market1'] and f"commodity_{comm_symbol}" in r['market2']]

                    if comm_relationships:
                        strongest_comm = max(comm_relationships, key=lambda x: abs(x['correlation']))
                        signal_enhancements.append({
                            'type': 'commodity_correlation',
                            'market': comm_symbol,
                            'correlation': strongest_comm['correlation'],
                            'signal_boost': 0.1 if abs(strongest_comm['correlation']) > 0.4 else 0
                        })

                # Check crypto relationships
                for crypto_symbol in self.crypto_data.keys():
                    crypto_relationships = [r for r in correlations['significant_relationships']
                                          if f"equity_{symbol}" in r['market1'] and f"crypto_{crypto_symbol}" in r['market2']]

                    if crypto_relationships:
                        strongest_crypto = max(crypto_relationships, key=lambda x: abs(x['correlation']))
                        signal_enhancements.append({
                            'type': 'crypto_correlation',
                            'market': crypto_symbol,
                            'correlation': strongest_crypto['correlation'],
                            'signal_boost': 0.05 if abs(strongest_crypto['correlation']) > 0.3 else 0
                        })

                # Apply signal enhancements
                total_boost = sum([enh['signal_boost'] for enh in signal_enhancements])

                if total_boost > 0:
                    # Enhance bullish signals
                    if enhanced_signals[symbol] > 0:
                        enhanced_signals[symbol] = min(enhanced_signals[symbol] * (1 + total_boost), 1)
                    # Dampen bearish signals if cross-market is positive
                    elif enhanced_signals[symbol] < 0 and total_boost > 0.2:
                        enhanced_signals[symbol] = enhanced_signals[symbol] * (1 - total_boost * 0.5)

                # Store enhancement info
                enhanced_signals[f"{symbol}_enhancements"] = signal_enhancements

        return enhanced_signals


class FXImpactAnalyzer:
    """Analyzes FX rate impacts on equity markets"""

    def __init__(self, currency_pairs=None):
        """
        Initialize FX impact analyzer

        Args:
            currency_pairs (list): List of currency pairs to analyze
        """
        self.currency_pairs = currency_pairs or ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']
        self.fx_impacts = {}

    def analyze_fx_equity_relationships(self, equity_data, fx_data, current_date):
        """
        Analyze how FX movements impact equity markets

        Args:
            equity_data (dict): Equity price data
            fx_data (dict): FX rate data
            current_date (pd.Timestamp): Current date

        Returns:
            dict: FX impact analysis
        """
        impacts = {}

        for equity_symbol in equity_data.keys():
            equity_returns = equity_data[equity_symbol]['Close'].pct_change().dropna()

            fx_impacts = []

            for fx_pair in self.currency_pairs:
                if fx_pair in fx_data:
                    fx_returns = fx_data[fx_pair]['Close'].pct_change().dropna()

                    # Find overlapping period
                    common_dates = equity_returns.index.intersection(fx_returns.index)
                    if len(common_dates) < 60:
                        continue

                    aligned_equity = equity_returns.loc[common_dates].tail(252)
                    aligned_fx = fx_returns.loc[common_dates].tail(252)

                    # Calculate relationship
                    corr, p_value = pearsonr(aligned_equity, aligned_fx)

                    if abs(corr) > 0.2 and p_value < 0.05:
                        fx_impacts.append({
                            'currency_pair': fx_pair,
                            'correlation': corr,
                            'p_value': p_value,
                            'impact_strength': abs(corr)
                        })

            impacts[equity_symbol] = {
                'fx_impacts': sorted(fx_impacts, key=lambda x: x['impact_strength'], reverse=True),
                'primary_fx_driver': fx_impacts[0]['currency_pair'] if fx_impacts else None,
                'net_fx_impact': sum([impact['correlation'] for impact in fx_impacts]) / len(fx_impacts) if fx_impacts else 0
            }

        return impacts


class CommodityInfluenceAnalyzer:
    """Analyzes commodity price impacts on equity markets"""

    def __init__(self, key_commodities=None):
        """
        Initialize commodity influence analyzer

        Args:
            key_commodities (list): Key commodities to analyze
        """
        self.key_commodities = key_commodities or ['GC=F', 'CL=F', 'SI=F', 'HG=F']  # Gold, Oil, Silver, Copper
        self.sector_mappings = {
            'technology': ['AAPL', 'MSFT', 'GOOGL'],
            'energy': ['XOM', 'CVX'],
            'materials': ['LIN', 'APD'],
            'financials': ['JPM', 'BAC']
        }

    def analyze_commodity_equity_relationships(self, equity_data, commodity_data, current_date):
        """
        Analyze commodity impacts on equity sectors

        Args:
            equity_data (dict): Equity price data
            commodity_data (dict): Commodity price data
            current_date (pd.Timestamp): Current date

        Returns:
            dict: Commodity influence analysis
        """
        sector_impacts = {}

        for sector, sector_stocks in self.sector_mappings.items():
            sector_returns = []

            # Get sector returns
            for stock in sector_stocks:
                if stock in equity_data:
                    returns = equity_data[stock]['Close'].pct_change().dropna()
                    if len(returns) > 30:
                        sector_returns.append(returns)

            if not sector_returns:
                continue

            # Average sector returns
            sector_avg = pd.concat(sector_returns, axis=1).mean(axis=1)

            commodity_impacts = []

            for commodity in self.key_commodities:
                if commodity in commodity_data:
                    comm_returns = commodity_data[commodity]['Close'].pct_change().dropna()

                    # Find overlapping period
                    common_dates = sector_avg.index.intersection(comm_returns.index)
                    if len(common_dates) < 60:
                        continue

                    aligned_sector = sector_avg.loc[common_dates].tail(252)
                    aligned_comm = comm_returns.loc[common_dates].tail(252)

                    # Calculate relationship
                    corr, p_value = pearsonr(aligned_sector, aligned_comm)

                    if abs(corr) > 0.15 and p_value < 0.10:
                        commodity_impacts.append({
                            'commodity': commodity,
                            'correlation': corr,
                            'p_value': p_value,
                            'impact_strength': abs(corr)
                        })

            sector_impacts[sector] = {
                'commodity_impacts': sorted(commodity_impacts, key=lambda x: x['impact_strength'], reverse=True),
                'primary_commodity': commodity_impacts[0]['commodity'] if commodity_impacts else None,
                'net_commodity_impact': sum([impact['correlation'] for impact in commodity_impacts]) / len(commodity_impacts) if commodity_impacts else 0
            }

        return sector_impacts


class GlobalEconomicSignalGenerator:
    """Generates signals from global economic indicators"""

    def __init__(self):
        """Initialize economic signal generator"""
        self.economic_indicators = {}
        self.signal_thresholds = {
            'interest_rate_change': 0.0025,  # 25bps
            'gdp_growth': 0.005,
            'inflation': 0.01,
            'unemployment': 0.5
        }

    def add_economic_data(self, indicator_name, data):
        """
        Add economic indicator data

        Args:
            indicator_name (str): Name of economic indicator
            data (pd.Series): Economic data series
        """
        self.economic_indicators[indicator_name] = data

    def generate_economic_signals(self, current_date):
        """
        Generate trading signals based on economic indicators

        Args:
            current_date (pd.Timestamp): Current date

        Returns:
            dict: Economic signals
        """
        signals = {}

        for indicator, data in self.economic_indicators.items():
            if current_date in data.index:
                current_value = data.loc[current_date]

                # Get recent trend
                recent_data = data.loc[:current_date].tail(12)  # Last 12 months

                if len(recent_data) >= 3:
                    trend = (recent_data.iloc[-1] - recent_data.iloc[0]) / recent_data.iloc[0]

                    # Generate signals based on indicator type
                    if 'interest_rate' in indicator.lower():
                        if abs(trend) > self.signal_thresholds['interest_rate_change']:
                            signals[f"{indicator}_signal"] = 1 if trend < 0 else -1  # Rate cuts bullish, hikes bearish

                    elif 'gdp' in indicator.lower():
                        if trend < -self.signal_thresholds['gdp_growth']:
                            signals[f"{indicator}_signal"] = -1  # GDP slowdown bearish
                        elif trend > self.signal_thresholds['gdp_growth']:
                            signals[f"{indicator}_signal"] = 1  # GDP acceleration bullish

                    elif 'inflation' in indicator.lower():
                        if trend > self.signal_thresholds['inflation']:
                            signals[f"{indicator}_signal"] = -1  # Rising inflation bearish
                        elif trend < -self.signal_thresholds['inflation']:
                            signals[f"{indicator}_signal"] = 1  # Falling inflation bullish

                    elif 'unemployment' in indicator.lower():
                        if trend > self.signal_thresholds['unemployment']:
                            signals[f"{indicator}_signal"] = -1  # Rising unemployment bearish
                        elif trend < -self.signal_thresholds['unemployment']:
                            signals[f"{indicator}_signal"] = 1  # Falling unemployment bullish

        # Aggregate signals
        bullish_signals = sum([1 for s in signals.values() if s == 1])
        bearish_signals = sum([1 for s in signals.values() if s == -1])

        overall_signal = 0
        if bullish_signals > bearish_signals:
            overall_signal = 1
        elif bearish_signals > bullish_signals:
            overall_signal = -1

        return {
            'individual_signals': signals,
            'overall_signal': overall_signal,
            'bullish_count': bullish_signals,
            'bearish_count': bearish_signals,
            'confidence': min(1.0, (bullish_signals + bearish_signals) / len(self.economic_indicators))
        }


# Example usage and integration
def create_cross_market_analyzer():
    """
    Create a fully configured cross-market analyzer

    Returns:
        CrossMarketAnalyzer: Configured analyzer
    """
    analyzer = CrossMarketAnalyzer()

    # In a real implementation, you would load actual market data
    # For demo purposes, we'll just return the analyzer structure

    return analyzer


if __name__ == "__main__":
    print("Cross-Market Signal Analysis System")
    print("=" * 45)

    # Test cross-market analyzer
    analyzer = create_cross_market_analyzer()

    print("Cross-market analyzer initialized with:")
    print("• Inter-market correlation analysis")
    print("• Lead-lag relationship detection")
    print("• FX impact analysis")
    print("• Commodity influence analysis")
    print("• Global economic signal generation")

    print("\nCross-market signal system ready!")
    print("Integrates signals across equities, FX, commodities, crypto, and economic indicators.")
