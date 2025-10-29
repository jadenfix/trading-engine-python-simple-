"""
Advanced Correlation Analysis for Finding Statistical Edges

This module implements sophisticated correlation analysis techniques to find
relationships between assets and identify potential trading opportunities.
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import mutual_info_score
from statsmodels.tsa.stattools import grangercausalitytests, coint
import warnings
warnings.filterwarnings('ignore')


class CorrelationAnalyzer:
    """Advanced correlation analysis for finding statistical edges between assets"""

    def __init__(self, min_correlation=0.1, min_significance=0.05):
        """
        Initialize correlation analyzer

        Args:
            min_correlation (float): Minimum correlation threshold for analysis
            min_significance (float): Minimum p-value threshold for statistical significance
        """
        self.min_correlation = min_correlation
        self.min_significance = min_significance

    def analyze_cross_sectional_correlations(self, price_data_dict, method='pearson'):
        """
        Analyze cross-sectional correlations between multiple assets

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            method (str): Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            dict: Comprehensive correlation analysis results
        """
        symbols = list(price_data_dict.keys())

        # Calculate returns for all symbols
        returns_dict = {}
        for symbol, data in price_data_dict.items():
            returns_dict[symbol] = data['Close'].pct_change().dropna()

        # Align all return series to common dates
        common_dates = self._find_common_dates(returns_dict)
        aligned_returns = {symbol: returns.loc[common_dates] for symbol, returns in returns_dict.items()}

        # Create correlation matrix
        returns_df = pd.DataFrame(aligned_returns)
        correlation_matrix = returns_df.corr(method=method)

        # Find significant correlations
        significant_pairs = self._find_significant_correlations(correlation_matrix, symbols)

        # Calculate cointegration relationships
        cointegration_pairs = self._find_cointegration_pairs(aligned_returns, symbols)

        # Calculate information coefficients
        info_coefficients = self._calculate_information_coefficients(aligned_returns)

        # Find lead-lag relationships
        lead_lag_analysis = self._analyze_lead_lag_relationships(aligned_returns)

        return {
            'correlation_matrix': correlation_matrix,
            'significant_pairs': significant_pairs,
            'cointegration_pairs': cointegration_pairs,
            'information_coefficients': info_coefficients,
            'lead_lag_analysis': lead_lag_analysis,
            'returns_data': aligned_returns,
            'symbols': symbols
        }

    def find_statistical_arbitrage_opportunities(self, price_data_dict, lookback_period=60):
        """
        Find statistical arbitrage opportunities using advanced correlation techniques

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            lookback_period (int): Period for calculating statistics

        Returns:
            dict: Statistical arbitrage opportunities
        """
        symbols = list(price_data_dict.keys())

        # Calculate returns and build correlation structure
        returns_dict = {}
        for symbol, data in price_data_dict.items():
            returns_dict[symbol] = data['Close'].pct_change().dropna()

        # Find common dates
        common_dates = self._find_common_dates(returns_dict)
        aligned_returns = {symbol: returns.loc[common_dates] for symbol, returns in returns_dict.items()}

        # Find pairs with strong correlations and potential mean reversion
        arb_opportunities = []

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]

                # Test for cointegration
                try:
                    coint_result = coint(aligned_returns[symbol1], aligned_returns[symbol2])
                    coint_t_stat, p_value, critical_values = coint_result[0], coint_result[1], coint_result[2]

                    if p_value < self.min_significance:
                        # Calculate hedge ratio and spread
                        hedge_ratio = self._calculate_hedge_ratio(aligned_returns[symbol1], aligned_returns[symbol2])

                        # Calculate spread statistics
                        spread = aligned_returns[symbol2] - hedge_ratio * aligned_returns[symbol1]
                        spread_mean = spread.rolling(window=lookback_period).mean()
                        spread_std = spread.rolling(window=lookback_period).std()

                        # Calculate half-life of mean reversion
                        half_life = self._calculate_half_life(spread)

                        if 5 <= half_life <= 50:  # Reasonable half-life range
                            arb_opportunities.append({
                                'pair': (symbol1, symbol2),
                                'hedge_ratio': hedge_ratio,
                                'half_life': half_life,
                                'cointegration_p_value': p_value,
                                'correlation': aligned_returns[symbol1].corr(aligned_returns[symbol2]),
                                'spread_mean': spread_mean.iloc[-1] if len(spread_mean) > 0 else 0,
                                'spread_std': spread_std.iloc[-1] if len(spread_std) > 0 else 1,
                                'z_score': (spread.iloc[-1] - spread_mean.iloc[-1]) / spread_std.iloc[-1] if len(spread_std) > 0 and spread_std.iloc[-1] != 0 else 0
                            })
                except:
                    continue

        # Sort by statistical significance and mean reversion strength
        arb_opportunities.sort(key=lambda x: (x['cointegration_p_value'], -abs(x['correlation']), x['half_life']))

        return {
            'arbitrage_opportunities': arb_opportunities,
            'total_pairs_tested': len(symbols) * (len(symbols) - 1) // 2,
            'significant_pairs': len([opp for opp in arb_opportunities if opp['cointegration_p_value'] < 0.05]),
            'best_opportunity': arb_opportunities[0] if arb_opportunities else None
        }

    def find_factor_momentum_opportunities(self, price_data_dict, factors=None):
        """
        Find factor momentum opportunities using cross-sectional analysis

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            factors (dict): Factor definitions (size, value, quality, etc.)

        Returns:
            dict: Factor momentum analysis results
        """
        if factors is None:
            # Default factor definitions
            factors = {
                'size': lambda returns: -returns.rolling(252).mean(),  # Negative momentum as proxy for size
                'value': lambda returns: returns.rolling(21).std() / returns.rolling(252).std(),  # Volatility as value proxy
                'quality': lambda returns: returns.rolling(63).mean() / returns.rolling(252).std(),  # Sharpe-like ratio
                'momentum': lambda returns: returns.rolling(252).mean()
            }

        symbols = list(price_data_dict.keys())

        # Calculate returns for all symbols
        returns_dict = {}
        for symbol, data in price_data_dict.items():
            returns_dict[symbol] = data['Close'].pct_change().dropna()

        # Align returns
        common_dates = self._find_common_dates(returns_dict)
        aligned_returns = {symbol: returns.loc[common_dates] for symbol, returns in returns_dict.items()}

        factor_scores = {}
        for factor_name, factor_func in factors.items():
            factor_scores[factor_name] = {}
            for symbol in symbols:
                try:
                    factor_scores[factor_name][symbol] = factor_func(aligned_returns[symbol])
                except:
                    factor_scores[factor_name][symbol] = pd.Series(0, index=common_dates)

        # Calculate factor momentum (momentum of factors)
        factor_momentum = {}
        for factor_name in factors.keys():
            factor_cross_section = pd.DataFrame(factor_scores[factor_name])
            factor_returns = factor_cross_section.mean(axis=1)  # Average factor score across stocks
            factor_momentum[factor_name] = factor_returns.rolling(21).mean()  # Factor momentum

        # Find which factors have the strongest momentum
        latest_factor_momentum = {factor: momentum.iloc[-1] for factor, momentum in factor_momentum.items()}

        # Identify stocks with strong exposure to high-momentum factors
        factor_exposure = {}
        for symbol in symbols:
            exposures = []
            for factor_name, factor_score in factor_scores.items():
                if len(factor_score[symbol].dropna()) > 30:
                    # Calculate correlation between stock returns and factor momentum
                    correlation = aligned_returns[symbol].corr(factor_momentum[factor_name])
                    exposures.append((factor_name, correlation, latest_factor_momentum[factor_name]))
            factor_exposure[symbol] = exposures

        return {
            'factor_momentum': latest_factor_momentum,
            'factor_exposure': factor_exposure,
            'best_factor': max(latest_factor_momentum.items(), key=lambda x: abs(x[1])),
            'factor_correlation_matrix': pd.DataFrame([
                [aligned_returns[s1].corr(aligned_returns[s2]) for s2 in symbols]
                for s1 in symbols
            ], index=symbols, columns=symbols)
        }

    def find_seasonal_patterns(self, price_data_dict, min_observations=5):
        """
        Find statistically significant seasonal patterns in asset returns

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            min_observations (int): Minimum observations for statistical significance

        Returns:
            dict: Seasonal pattern analysis results
        """
        symbols = list(price_data_dict.keys())

        # Calculate returns
        returns_dict = {}
        for symbol, data in price_data_dict.items():
            returns_dict[symbol] = data['Close'].pct_change().dropna()

        seasonal_patterns = {}

        for symbol in symbols:
            returns = returns_dict[symbol]

            # Day of week patterns
            dow_returns = []
            dow_pvalues = []

            for day in range(5):  # Monday to Friday
                day_returns = returns[returns.index.dayofweek == day]
                if len(day_returns) >= min_observations:
                    dow_returns.append(day_returns.mean())
                    # Test if significantly different from overall mean
                    t_stat, p_value = stats.ttest_1samp(day_returns, returns.mean())
                    dow_pvalues.append(p_value)
                else:
                    dow_returns.append(0)
                    dow_pvalues.append(1.0)

            # Month patterns
            month_returns = []
            month_pvalues = []

            for month in range(1, 13):
                month_rets = returns[returns.index.month == month]
                if len(month_rets) >= min_observations:
                    month_returns.append(month_rets.mean())
                    t_stat, p_value = stats.ttest_1samp(month_rets, returns.mean())
                    month_pvalues.append(p_value)
                else:
                    month_returns.append(0)
                    month_pvalues.append(1.0)

            # Intraday patterns (if intraday data available)
            hour_patterns = self._analyze_intraday_patterns(price_data_dict[symbol])

            seasonal_patterns[symbol] = {
                'day_of_week': {
                    'returns': dow_returns,
                    'p_values': dow_pvalues,
                    'significant_days': [day for day, p in enumerate(dow_pvalues) if p < self.min_significance]
                },
                'monthly': {
                    'returns': month_returns,
                    'p_values': month_pvalues,
                    'significant_months': [month for month, p in enumerate(month_pvalues, 1) if p < self.min_significance]
                },
                'intraday': hour_patterns
            }

        return {
            'seasonal_patterns': seasonal_patterns,
            'summary': self._summarize_seasonal_patterns(seasonal_patterns)
        }

    def find_network_correlations(self, price_data_dict, min_correlation=0.3):
        """
        Find network-based correlations and clusters of related assets

        Args:
            price_data_dict (dict): Dictionary with symbols as keys and price DataFrames as values
            min_correlation (float): Minimum correlation for network connections

        Returns:
            dict: Network correlation analysis
        """
        symbols = list(price_data_dict.keys())

        # Calculate returns
        returns_dict = {}
        for symbol, data in price_data_dict.items():
            returns_dict[symbol] = data['Close'].pct_change().dropna()

        # Align returns
        common_dates = self._find_common_dates(returns_dict)
        aligned_returns = {symbol: returns.loc[common_dates] for symbol, returns in returns_dict.items()}

        # Create correlation matrix
        correlation_matrix = pd.DataFrame([
            [aligned_returns[s1].corr(aligned_returns[s2]) for s2 in symbols]
            for s1 in symbols
        ], index=symbols, columns=symbols)

        # Find clusters using hierarchical clustering
        correlation_distances = 1 - correlation_matrix.abs()
        linkage_matrix = linkage(correlation_distances.values, method='average')

        # Identify strong correlation clusters
        clusters = self._identify_correlation_clusters(correlation_matrix, min_correlation)

        # Find central nodes (most connected assets)
        centrality_scores = self._calculate_centrality_scores(correlation_matrix)

        return {
            'correlation_matrix': correlation_matrix,
            'linkage_matrix': linkage_matrix,
            'clusters': clusters,
            'centrality_scores': centrality_scores,
            'network_summary': {
                'total_assets': len(symbols),
                'strong_connections': (correlation_matrix.abs() > min_correlation).sum().sum() // 2,
                'avg_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix, k=1)].mean(),
                'max_correlation': correlation_matrix.values.max(),
                'min_correlation': correlation_matrix.values.min()
            }
        }

    def _find_common_dates(self, returns_dict):
        """Find common dates across all return series"""
        if not returns_dict:
            return pd.DatetimeIndex([])

        common_dates = None
        for returns in returns_dict.values():
            if common_dates is None:
                common_dates = returns.index
            else:
                common_dates = common_dates.intersection(returns.index)

        return common_dates

    def _find_significant_correlations(self, correlation_matrix, symbols):
        """Find statistically significant correlations"""
        significant_pairs = []

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                corr_value = correlation_matrix.iloc[i, j]

                if abs(corr_value) >= self.min_correlation:
                    # Test statistical significance
                    n = len(correlation_matrix)
                    t_stat = corr_value * np.sqrt((n - 2) / (1 - corr_value**2))
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

                    if p_value < self.min_significance:
                        significant_pairs.append({
                            'pair': (symbols[i], symbols[j]),
                            'correlation': corr_value,
                            'p_value': p_value,
                            'significance_level': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*'
                        })

        return significant_pairs

    def _find_cointegration_pairs(self, aligned_returns, symbols):
        """Find cointegrated pairs"""
        cointegration_pairs = []

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]

                try:
                    coint_result = coint(aligned_returns[symbol1], aligned_returns[symbol2])
                    coint_t_stat, p_value, critical_values = coint_result[0], coint_result[1], coint_result[2]

                    if p_value < self.min_significance:
                        hedge_ratio = self._calculate_hedge_ratio(aligned_returns[symbol1], aligned_returns[symbol2])
                        half_life = self._calculate_half_life(
                            aligned_returns[symbol2] - hedge_ratio * aligned_returns[symbol1]
                        )

                        cointegration_pairs.append({
                            'pair': (symbol1, symbol2),
                            'hedge_ratio': hedge_ratio,
                            'half_life': half_life,
                            'cointegration_p_value': p_value,
                            'correlation': aligned_returns[symbol1].corr(aligned_returns[symbol2])
                        })
                except:
                    continue

        return cointegration_pairs

    def _calculate_hedge_ratio(self, price1, price2):
        """Calculate optimal hedge ratio using linear regression"""
        model = stats.linregress(price1, price2)
        return model.slope

    def _calculate_half_life(self, spread):
        """Calculate half-life of mean reversion"""
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag
        spread_lag = spread_lag.dropna()
        spread_diff = spread_diff.dropna()

        if len(spread_lag) < 10:
            return float('inf')

        model = stats.linregress(spread_lag, spread_diff)
        if model.slope >= 0:
            return float('inf')

        return -np.log(2) / model.slope

    def _calculate_information_coefficients(self, aligned_returns):
        """Calculate information coefficients between assets"""
        symbols = list(aligned_returns.keys())
        info_coeffs = {}

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]

                # Calculate mutual information
                try:
                    # Discretize returns for mutual information calculation
                    returns1_discrete = pd.qcut(aligned_returns[symbol1].rank(), q=10, duplicates='drop')
                    returns2_discrete = pd.qcut(aligned_returns[symbol2].rank(), q=10, duplicates='drop')

                    mi_score = mutual_info_score(returns1_discrete, returns2_discrete)
                    info_coeffs[(symbol1, symbol2)] = mi_score
                except:
                    info_coeffs[(symbol1, symbol2)] = 0

        return info_coeffs

    def _analyze_lead_lag_relationships(self, aligned_returns):
        """Analyze lead-lag relationships between assets"""
        symbols = list(aligned_returns.keys())
        lead_lag_results = {}

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]

                # Test Granger causality in both directions
                try:
                    # Test if symbol1 Granger-causes symbol2
                    granger_12 = grangercausalitytests(
                        pd.DataFrame({'y': aligned_returns[symbol2], 'x': aligned_returns[symbol1]}),
                        maxlag=5, verbose=False
                    )

                    # Test if symbol2 Granger-causes symbol1
                    granger_21 = grangercausalitytests(
                        pd.DataFrame({'y': aligned_returns[symbol1], 'x': aligned_returns[symbol2]}),
                        maxlag=5, verbose=False
                    )

                    # Find best lag for each direction
                    best_lag_12 = min(granger_12.keys(),
                                    key=lambda k: granger_12[k][0]['ssr_ftest'][1])
                    best_lag_21 = min(granger_21.keys(),
                                    key=lambda k: granger_21[k][0]['ssr_ftest'][1])

                    lead_lag_results[(symbol1, symbol2)] = {
                        'symbol1_causes_symbol2': {
                            'best_lag': best_lag_12,
                            'p_value': granger_12[best_lag_12][0]['ssr_ftest'][1]
                        },
                        'symbol2_causes_symbol1': {
                            'best_lag': best_lag_21,
                            'p_value': granger_21[best_lag_21][0]['ssr_ftest'][1]
                        }
                    }
                except:
                    lead_lag_results[(symbol1, symbol2)] = None

        return lead_lag_results

    def _analyze_intraday_patterns(self, data):
        """Analyze intraday patterns if data is available"""
        # Placeholder for intraday analysis
        return {'hourly_patterns': {}, 'significant_hours': []}

    def _summarize_seasonal_patterns(self, seasonal_patterns):
        """Summarize seasonal patterns across all assets"""
        summary = {
            'strongest_dow_patterns': {},
            'strongest_monthly_patterns': {},
            'assets_with_seasonality': []
        }

        for symbol, patterns in seasonal_patterns.items():
            # Check for significant day patterns
            dow_sig = patterns['day_of_week']['significant_days']
            if dow_sig:
                summary['assets_with_seasonality'].append(symbol)
                summary['strongest_dow_patterns'][symbol] = dow_sig

            # Check for significant month patterns
            month_sig = patterns['monthly']['significant_months']
            if month_sig:
                summary['strongest_monthly_patterns'][symbol] = month_sig

        return summary

    def _identify_correlation_clusters(self, correlation_matrix, min_correlation):
        """Identify clusters of highly correlated assets"""
        # Simple threshold-based clustering
        clusters = []

        # Find assets with correlation above threshold
        high_corr_pairs = []
        symbols = correlation_matrix.index.tolist()

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                if abs(correlation_matrix.iloc[i, j]) > min_correlation:
                    high_corr_pairs.append((symbols[i], symbols[j], correlation_matrix.iloc[i, j]))

        # Group connected components
        clusters = self._find_connected_components(high_corr_pairs, symbols)

        return {
            'clusters': clusters,
            'high_correlation_pairs': high_corr_pairs,
            'num_clusters': len(clusters)
        }

    def _find_connected_components(self, pairs, all_symbols):
        """Find connected components in correlation network"""
        # Simple implementation of connected components
        components = []
        visited = set()

        for symbol in all_symbols:
            if symbol not in visited:
                component = self._dfs_find_component(symbol, pairs, visited)
                if len(component) > 1:  # Only include clusters with multiple assets
                    components.append(component)

        return components

    def _dfs_find_component(self, start_symbol, pairs, visited):
        """Depth-first search to find connected component"""
        component = set([start_symbol])
        stack = [start_symbol]

        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)

                for symbol1, symbol2, corr in pairs:
                    if symbol1 == current and symbol2 not in component:
                        component.add(symbol2)
                        stack.append(symbol2)
                    elif symbol2 == current and symbol1 not in component:
                        component.add(symbol1)
                        stack.append(symbol1)

        return list(component)

    def _calculate_centrality_scores(self, correlation_matrix):
        """Calculate centrality scores for assets in correlation network"""
        centrality = {}

        for symbol in correlation_matrix.index:
            # Degree centrality (number of strong correlations)
            strong_connections = (correlation_matrix.abs().loc[symbol] > self.min_correlation).sum() - 1
            centrality[symbol] = strong_connections

        return centrality

