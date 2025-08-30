import os

import holidays
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis, entropy
from statsmodels.tsa.stattools import acf
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.trend import PSARIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange

FEATURE_NAME = 'T10YIE'


class FeatureGenerator:
    def __init__(self, path='/Users/arynaartsiukevich/PycharmProjects/inflation_rate_time_series'):
        self.path = path
        self.data_path = os.path.join(path, 'input', f'{FEATURE_NAME}_original.csv')
        self.plots_path = os.path.join(path, 'data', 'plots')
        self._df = pd.read_csv(self.data_path, parse_dates=True, index_col='observation_date')
        self.df_features = self._df.copy()
        self.feature_name = 'T10YIE'

    def generate_lag_features(self, simple_lags=20, seasonal_period=20, seasonal_lags=3, log_lags=10):
        df_features = self._df.copy()

        for lag in range(1, simple_lags + 1):
            df_features[f'lag_{lag}'] = df_features[f'{FEATURE_NAME}'].shift(lag)

        for i in range(1, seasonal_lags + 1):
            df_features[f'seasonal_lag_{i * seasonal_period}'] = df_features[f'{FEATURE_NAME}'].shift(
                i * seasonal_period)

        df_features['diff_1'] = df_features[f'{FEATURE_NAME}'].diff()
        df_features['diff_2'] = df_features['diff_1'].diff()

        df_features['diff_1_lag_1'] = df_features['diff_1'].shift(1)
        df_features['diff_1_lag_2'] = df_features['diff_1'].shift(2)

        df_features[f'{FEATURE_NAME}_clipped'] = df_features[f'{FEATURE_NAME}'].clip(lower=1e-6)
        for lag in range(1, log_lags + 1):
            df_features[f'log_lag_{lag}'] = np.log(df_features[f'{FEATURE_NAME}_clipped'].shift(lag))

        df_features = df_features.drop(columns=[f'{FEATURE_NAME}_clipped'])

        self.df_features = df_features.dropna()
        print(f"Generated lag- features.")
        return self.df_features

    def generate_sliding_features(self, window_sizes=(6, 8, 12, 14, 18)):
        df_features = self.df_features.copy()
        for w in window_sizes:
            df_features[f'ma_{w}'] = df_features[f'{FEATURE_NAME}'].rolling(window=w).mean()

            df_features[f'median_{w}'] = df_features[f'{FEATURE_NAME}'].rolling(window=w).median()

            df_features[f'std_{w}'] = df_features[f'{FEATURE_NAME}'].rolling(window=w).std()

            df_features[f'min_{w}'] = df_features[f'{FEATURE_NAME}'].rolling(window=w).min()

            df_features[f'max_{w}'] = df_features[f'{FEATURE_NAME}'].rolling(window=w).max()

            df_features[f'range_{w}'] = df_features[f'max_{w}'] - df_features[f'min_{w}']

            df_features[f'var_{w}'] = df_features[f'{FEATURE_NAME}'].rolling(window=w).var()

            df_features[f'skew_{w}'] = df_features[f'{FEATURE_NAME}'].rolling(window=w).apply(skew, raw=True)

            df_features[f'kurt_{w}'] = df_features[f'{FEATURE_NAME}'].rolling(window=w).apply(kurtosis, raw=True)

            df_features[f'q25_{w}'] = df_features[f'{FEATURE_NAME}'].rolling(window=w).quantile(0.25)
            df_features[f'q75_{w}'] = df_features[f'{FEATURE_NAME}'].rolling(window=w).quantile(0.75)

            df_features[f'iqr_{w}'] = df_features[f'q75_{w}'] - df_features[f'q25_{w}']

        self.df_features = df_features.dropna()
        print(f"Generated sliding features.")
        return self.df_features

    def generate_ema_features(self, ema_alphas=(0.1, 0.2, 0.3, 0.5, 0.7, 0.9)):
        df_features = self.df_features.copy()
        for alpha in ema_alphas:
            df_features[f'ema_{alpha}'] = df_features[f'{FEATURE_NAME}'].ewm(alpha=alpha, adjust=False).mean()

            ema = df_features[f'{FEATURE_NAME}'].ewm(alpha=alpha, adjust=False).mean()
            ema_ema = ema.ewm(alpha=alpha, adjust=False).mean()
            df_features[f'dema_{alpha}'] = 2 * ema - ema_ema

            ema_ema_ema = ema_ema.ewm(alpha=alpha, adjust=False).mean()
            df_features[f'tema_{alpha}'] = 3 * ema - 3 * ema_ema + ema_ema_ema

        diff = df_features[f'{FEATURE_NAME}'].diff().abs()
        sigma = diff.std() + 1e-6
        alpha_t = 2 / (1 + np.exp(-diff / sigma))
        adaptive_ema = np.zeros(len(df_features))
        adaptive_ema[0] = df_features[f'{FEATURE_NAME}'].iloc[0]
        for t in range(1, len(df_features)):
            adaptive_ema[t] = alpha_t.iloc[t] * df_features[f'{FEATURE_NAME}'].iloc[t] + (1 - alpha_t.iloc[t]) * \
                              adaptive_ema[
                                  t - 1]
        df_features.loc[:, 'adaptive_ema'] = adaptive_ema

        self.df_features = df_features.dropna()
        print(f"Generated ema features.")
        return self.df_features

    def generate_technical_features(self):
        df_features = self.df_features.copy()
        for period in [14, 21, 30]:
            rsi = RSIIndicator(close=df_features[f'{FEATURE_NAME}'], window=period).rsi()
            df_features.loc[:, f'rsi_{period}'] = rsi
        stochastic = StochasticOscillator(close=df_features[f'{FEATURE_NAME}'], high=df_features[f'{FEATURE_NAME}'],
                                          low=df_features[f'{FEATURE_NAME}'],
                                          window=14, smooth_window=3)
        df_features.loc[:, 'stoch_k'] = stochastic.stoch()
        df_features.loc[:, 'stoch_d'] = stochastic.stoch_signal()
        williams = WilliamsRIndicator(high=df_features[f'{FEATURE_NAME}'], low=df_features[f'{FEATURE_NAME}'],
                                      close=df_features[f'{FEATURE_NAME}'],
                                      lbp=14)
        df_features.loc[:, 'williams_r'] = williams.williams_r()

        for period in [10, 20, 30]:
            roc = ROCIndicator(close=df_features[f'{FEATURE_NAME}'], window=period).roc()
            df_features.loc[:, f'roc_{period}'] = roc
            df_features.loc[:, f'momentum_{period}'] = df_features[f'{FEATURE_NAME}'] - df_features[
                f'{FEATURE_NAME}'].shift(period)

        macd = MACD(close=df_features[f'{FEATURE_NAME}'], window_slow=26, window_fast=12, window_sign=9)
        df_features.loc[:, 'macd'] = macd.macd()
        df_features.loc[:, 'macd_signal'] = macd.macd_signal()
        df_features.loc[:, 'macd_hist'] = macd.macd_diff()
        psar = PSARIndicator(high=df_features[f'{FEATURE_NAME}'], low=df_features[f'{FEATURE_NAME}'],
                             close=df_features[f'{FEATURE_NAME}'], step=0.02,
                             max_step=0.2)
        df_features.loc[:, 'psar'] = psar.psar()
        adx = ADXIndicator(high=df_features[f'{FEATURE_NAME}'], low=df_features[f'{FEATURE_NAME}'],
                           close=df_features[f'{FEATURE_NAME}'], window=14)
        df_features.loc[:, 'adx'] = adx.adx()

        bb = BollingerBands(close=df_features[f'{FEATURE_NAME}'], window=20, window_dev=2)
        df_features.loc[:, 'bb_upper'] = bb.bollinger_hband()
        df_features.loc[:, 'bb_middle'] = bb.bollinger_mavg()
        df_features.loc[:, 'bb_lower'] = bb.bollinger_lband()
        df_features.loc[:, 'bb_width'] = (df_features['bb_upper'] - df_features['bb_lower']) / df_features['bb_middle']
        df_features.loc[:, 'bb_position'] = (df_features[f'{FEATURE_NAME}'] - df_features['bb_lower']) / (
                df_features['bb_upper'] - df_features['bb_lower'])
        atr = AverageTrueRange(high=df_features[f'{FEATURE_NAME}'], low=df_features[f'{FEATURE_NAME}'],
                               close=df_features[f'{FEATURE_NAME}'], window=14)
        df_features.loc[:, 'atr'] = atr.average_true_range()
        std_14 = df_features[f'{FEATURE_NAME}'].rolling(window=14).std()
        std_50 = df_features[f'{FEATURE_NAME}'].rolling(window=50).std()
        df_features.loc[:, 'vol_ratio'] = std_14 / std_50

        self.df_features = df_features.dropna()
        print(f"Generated technical features.")
        return self.df_features

    def generate_date_features(self):
        df_features = self.df_features.copy()
        df_features.loc[:, 'dayofmonth'] = df_features.index.day
        df_features.loc[:, 'quarter'] = df_features.index.quarter
        df_features.loc[:, 'dayofyear'] = df_features.index.dayofyear
        df_features.loc[:, 'weekofyear'] = df_features.index.isocalendar().week
        for day in range(5):
            df_features.loc[:, f'dayofweek_{day}'] = (df_features.index.weekday == day).astype(int)
        for month in range(1, 13):
            df_features.loc[:, f'month_{month}'] = (df_features.index.month == month).astype(int)

        df_features.loc[:, 'sin_dayofweek'] = np.sin(2 * np.pi * df_features.index.weekday / 7)
        df_features.loc[:, 'cos_dayofweek'] = np.cos(2 * np.pi * df_features.index.weekday / 7)
        df_features.loc[:, 'sin_month'] = np.sin(2 * np.pi * df_features.index.month / 12)
        df_features.loc[:, 'cos_month'] = np.cos(2 * np.pi * df_features.index.month / 12)
        df_features.loc[:, 'sin_dayofyear'] = np.sin(2 * np.pi * df_features.index.dayofyear / 365)
        df_features.loc[:, 'cos_dayofyear'] = np.cos(2 * np.pi * df_features.index.dayofyear / 365)

        us_holidays = holidays.US(years=range(2018, 2026))
        df_features.loc[:, 'is_holiday'] = df_features.index.isin(us_holidays).astype(int)
        days_in_month = df_features.index.to_series().dt.days_in_month
        df_features.loc[:, 'is_start_month'] = (df_features.index.day <= 3).astype(int)
        df_features.loc[:, 'is_end_month'] = (df_features.index.day >= days_in_month - 3).astype(int)

        self.df_features = df_features.dropna()
        print(f"Generated date features.")
        return self.df_features

    def generate_stats_features(self):
        df_features = self.df_features.copy()
        for lag in [1, 2, 3, 7, 14, 30]:
            acf_values = df_features[f'{FEATURE_NAME}'].rolling(window=50).apply(
                lambda x: acf(x, nlags=lag, fft=False)[-1] if len(x) > lag else np.nan, raw=True)
            df_features.loc[:, f'acf_lag_{lag}'] = acf_values

        for w in [10, 20, 30]:
            df_features.loc[:, f'skew_{w}'] = df_features[f'{FEATURE_NAME}'].rolling(window=w).apply(skew, raw=True)
            df_features.loc[:, f'kurt_{w}'] = df_features[f'{FEATURE_NAME}'].rolling(window=w).apply(kurtosis, raw=True)
            df_features.loc[:, f'shannon_entropy_{w}'] = df_features[f'{FEATURE_NAME}'].rolling(window=w).apply(
                lambda x: entropy(np.histogram(x, bins=10, density=True)[0]) if np.sum(
                    np.histogram(x, bins=10, density=True)[0]) > 0 else np.nan, raw=True)
            df_features.loc[:, f'mad_{w}'] = df_features[f'{FEATURE_NAME}'].rolling(window=w).apply(
                lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
            df_features.loc[:, f'coeff_var_{w}'] = df_features[f'{FEATURE_NAME}'].rolling(window=w).std() / (
                    df_features[f'{FEATURE_NAME}'].rolling(window=w).mean() + 1e-6)
            df_features.loc[:, f'z_score_diff_{w}'] = df_features['z_score_rate'].rolling(window=w).apply(
                lambda x: x[-1] - x[0] if len(x) == w else np.nan, raw=True)

        df_features.loc[:, 'var_ratio_5_20'] = df_features[f'{FEATURE_NAME}'].rolling(window=5).var() / (
                df_features[f'{FEATURE_NAME}'].rolling(window=20).var() + 1e-6)
        df_features.loc[:, 'var_ratio_10_50'] = df_features[f'{FEATURE_NAME}'].rolling(window=10).var() / (
                df_features[f'{FEATURE_NAME}'].rolling(window=50).var() + 1e-6)

        for lag in [1, 7, 14]:
            df_features.loc[:, f'corr_lag_{lag}'] = df_features[f'{FEATURE_NAME}'].rolling(window=20).corr(
                df_features[f'{FEATURE_NAME}'].shift(lag))

        self.df_features = df_features.dropna()
        print(f"Generated stats features.")
        return self.df_features

    def generate_mixed_features(self):
        df_features = self.df_features.copy()

        df_features.loc[:, 'ema_ratio_0.5_0.1'] = df_features['ema_0.5'] / (df_features['ema_0.1'] + 1e-6)
        df_features.loc[:, 'ema_ratio_0.7_0.3'] = df_features['ema_0.7'] / (df_features['ema_0.3'] + 1e-6)
        df_features.loc[:, 'ema_diff_0.5_0.1'] = df_features['ema_0.5'] - df_features['ema_0.1']

        df_features.loc[:, 'dist_to_ema_0.3'] = df_features[f'{FEATURE_NAME}'] - df_features['ema_0.3']
        df_features.loc[:, 'dist_to_ema_0.5'] = df_features[f'{FEATURE_NAME}'] - df_features['ema_0.5']
        df_features.loc[:, 'dist_to_ema_0.7'] = df_features[f'{FEATURE_NAME}'] - df_features['ema_0.7']

        rsi_norm = (df_features['rsi_14'] - df_features['rsi_14'].min()) / (
                df_features['rsi_14'].max() - df_features['rsi_14'].min() + 1e-6)
        stoch_k_norm = (df_features['stoch_k'] - df_features['stoch_k'].min()) / (
                df_features['stoch_k'].max() - df_features['stoch_k'].min() + 1e-6)
        williams_norm = (df_features['williams_r'] - df_features['williams_r'].min()) / (
                df_features['williams_r'].max() - df_features['williams_r'].min() + 1e-6)
        df_features.loc[:, 'momentum_composite'] = (rsi_norm + stoch_k_norm + williams_norm) / 3

        df_features.loc[:, 'ema_crossover'] = ((df_features['ema_0.5'] > df_features['ema_0.1']) & (
                df_features['ema_0.5'].shift(1) <= df_features['ema_0.1'].shift(1))).astype(int) - \
                                              ((df_features['ema_0.5'] < df_features['ema_0.1']) & (
                                                      df_features['ema_0.5'].shift(1) >= df_features[
                                                  'ema_0.1'].shift(1))).astype(int)

        df_features.loc[:, 'z_score_rate'] = (df_features[f'{FEATURE_NAME}'] - df_features[f'{FEATURE_NAME}'].rolling(
            window=20).mean()) / (df_features[f'{FEATURE_NAME}'].rolling(window=20).std() + 1e-6)
        df_features.loc[:, 'z_score_rsi'] = (df_features['rsi_14'] - df_features['rsi_14'].rolling(
            window=20).mean()) / (df_features['rsi_14'].rolling(window=20).std() + 1e-6)

        df_features.loc[:, 'is_peak'] = (
                df_features[f'{FEATURE_NAME}'] == df_features[f'{FEATURE_NAME}'].rolling(window=5,
                                                                                         center=True).max()).astype(
            int)
        df_features.loc[:, 'is_trough'] = (
                df_features[f'{FEATURE_NAME}'] == df_features[f'{FEATURE_NAME}'].rolling(window=5,
                                                                                         center=True).min()).astype(
            int)
        df_features.loc[:, 'trend_direction'] = np.sign(df_features[f'{FEATURE_NAME}'].rolling(window=20).mean().diff())

        self.df_features = df_features.dropna()
        print(f"Generated mixed features.")
        return self.df_features

    def get_feature_statistics(self):
        desc_stats = self.df_features.describe(percentiles=[0.25, 0.5, 0.75]).round(4)
        print("Feature Statistics:")
        print(desc_stats)

    def plot_feature_correlation(self):
        plt.figure(figsize=(15, 10))
        corr = self.df_features.corr()
        sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
        plt.title("Correlation Heatmap of Features")
        plt.savefig(os.path.join(self.plots_path, 'feature_correlation.png'))
        plt.show()

    def generate_features(self):
        self.generate_lag_features()
        self.generate_sliding_features()
        self.generate_ema_features()
        self.generate_technical_features()
        self.generate_date_features()
        self.generate_mixed_features()
        self.generate_stats_features()

        self.get_feature_statistics()
        self.plot_feature_correlation()

    def save_expanded_df(self, file_name=f'{FEATURE_NAME}.csv'):
        output_path = os.path.join(self.path, 'input', file_name)
        self.df_features.to_csv(output_path)


if __name__ == '__main__':
    np.random.seed(42)
    feature_generator = FeatureGenerator()
    feature_generator.generate_features()
