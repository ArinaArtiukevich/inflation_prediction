import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro
from scipy.stats import zscore
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf

FEATURE_NAME = 'T10YIE'


class DataExplorer:
    def __init__(self, path='/Users/arynaartsiukevich/PycharmProjects/inflation_rate_time_series'):
        self.data_path = os.path.join(path, 'input', 'T10YIE_original.csv')
        self.plots_path = os.path.join(path, 'data', 'plots')

        self.df = self._get_data()
        self._trend = None
        self._seasonal = None
        self._residual = None

    def _get_data(self):
        self.df = pd.read_csv(self.data_path, parse_dates=True, index_col='observation_date')
        self.df.dropna(subset=[f'{FEATURE_NAME}'], inplace=True)
        return self.df

    def get_statistics(self):
        desc_stats = self.df[f'{FEATURE_NAME}'].describe(percentiles=[0.25, 0.5, 0.75]).round(4)
        print("Descriptive Statistics:")
        skewness = self.df[f'{FEATURE_NAME}'].skew().round(4)
        kurtosis = self.df[f'{FEATURE_NAME}'].kurtosis().round(4)
        desc_stats['skewness'] = skewness
        desc_stats['kurtosis'] = kurtosis
        print(desc_stats)

    def plot_distribution(self):
        plt.figure(figsize=(8, 5))
        sns.histplot(self.df[f'{FEATURE_NAME}'], bins=30, kde=True)
        plt.title(f"Histogram of {FEATURE_NAME} (10-Year Inflation Expectations)")
        plt.xlabel(f"{FEATURE_NAME} (%)")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(self.plots_path, 'histogram.png'))

        sm.qqplot(self.df[f'{FEATURE_NAME}'].dropna(), line='s')
        plt.title(f"QQ Plot of {FEATURE_NAME}")
        plt.savefig(os.path.join(self.plots_path, 'qq_plot.png'))

    def detect_outliers(self):
        self.plot_boxplot()

        self.df["z_score"] = zscore(self.df[f'{FEATURE_NAME}'], nan_policy='omit')
        self.df["z_score"] = self.df["z_score"].round(4)

        outliers = self.df[abs(self.df["z_score"]) > 3]
        print("Number of Outliers:", len(outliers))
        # print("Sample Outliers:\n", outliers.round(4))

        self.plot_scatter_outliers(outliers)

    def plot_boxplot(self):
        plt.figure(figsize=(6, 5))
        sns.boxplot(y=self.df[f'{FEATURE_NAME}'])
        plt.title(f"Box Plot of {FEATURE_NAME}")
        plt.ylabel(f"{FEATURE_NAME} (%)")
        plt.savefig(os.path.join(self.plots_path, 'box_plot.png'))

    def plot_scatter_outliers(self, outliers):
        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df[f'{FEATURE_NAME}'], label=f'{FEATURE_NAME}')
        plt.scatter(outliers.index, outliers[f'{FEATURE_NAME}'], color="red", label="Outliers")

        plt.legend()
        plt.title(f"{FEATURE_NAME} with Outliers Highlighted")
        plt.xlabel("Date")
        plt.ylabel(f"{FEATURE_NAME} (%)")
        plt.savefig(os.path.join(self.plots_path, 'scatter_with_outliers.png'))

    def decompose(self):
        decomposition = seasonal_decompose(self.df[f'{FEATURE_NAME}'], model='additive', period=365)
        self._trend = decomposition.trend
        self._seasonal = decomposition.seasonal
        self._residual = decomposition.resid

        plt.figure(figsize=(12, 10))
        plt.subplot(4, 1, 1)
        plt.plot(self.df.index, self.df[f'{FEATURE_NAME}'], color='blue', label='Original')
        plt.title('Original Time Series')
        plt.ylabel('Rate (%)')
        plt.grid(True)
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(self.df.index, self._trend, color='green', label='Trend')
        plt.title('Trend Component')
        plt.ylabel('Rate (%)')
        plt.grid(True)
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(self.df.index, self._seasonal, color='orange', label='Seasonal')
        plt.title('Seasonal Component')
        plt.ylabel('Rate (%)')
        plt.grid(True)
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(self.df.index, self._residual, color='red', label='Residual')
        plt.title('Residual Component')
        plt.ylabel('Rate (%)')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, 'decomposed.png'))

        print("Decomposition Statistics:")
        print(f"Trend Mean: {self._trend.mean():.4f}, Std: {self._trend.std():.4f}")
        print(f"Seasonal Mean: {self._seasonal.mean():.4f}, Std: {self._seasonal.std():.4f}")
        print(f"Residual Mean: {self._residual.mean():.4f}, Std: {self._residual.std():.4f}")

    def plot_acf(self):
        plot_acf(self.df[f'{FEATURE_NAME}'], lags=40, alpha=0.05)
        plt.title('Autocorrelation Function (ACF)')
        plt.xlabel('Lag')
        plt.ylabel('ACF')
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_path, 'acf.png'))

        acf_values = acf(self.df[f'{FEATURE_NAME}'], nlags=40, fft=False)
        print("ACF Values (first 40 lags):")
        for lag, value in enumerate(acf_values[:10]):
            print(f"Lag {lag}: {value:.4f}")

    def plot_pacf(self):
        plot_pacf(self.df[f'{FEATURE_NAME}'], lags=40, alpha=0.05, method='ols')
        plt.title('Partial Autocorrelation Function (PACF)')
        plt.xlabel('Lag')
        plt.ylabel('PACF')
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_path, 'pacf.png'))

        pacf_values = pacf(self.df[f'{FEATURE_NAME}'], nlags=40, method='ols')
        print("PACF Values (first 40 lags):")
        for lag, value in enumerate(pacf_values[:10]):
            print(f"Lag {lag}: {value:.4f}")

    def get_statistical_tests(self):
        residual_clean = self._residual.dropna()
        adf_result = adfuller(self.df[f'{FEATURE_NAME}'])
        print("\nAugmented Dickey-Fuller Test for Stationarity:")
        print(f"ADF Statistic: {adf_result[0]:.4f}")
        print(f"p-value: {adf_result[1]:.4f}")
        print("Critical Values:")
        for key, value in adf_result[4].items():
            print(f"  {key}: {value:.4f}")
        print("Conclusion: Stationary" if adf_result[1] < 0.05 else "Conclusion: Non-stationary")

        shapiro_stat, shapiro_p = shapiro(residual_clean)
        print("\nShapiro-Wilk Test for Normality of Residuals:")
        print(f"Statistic: {shapiro_stat:.4f}")
        print(f"p-value: {shapiro_p:.4f}")
        print("Conclusion: Normal" if shapiro_p > 0.05 else "Conclusion: Non-normal")

        lb_test = acorr_ljungbox(residual_clean, lags=40, return_df=True)
        print("\nLjung-Box Test for Autocorrelation of Residuals (first 40 lags):")
        for lag, row in lb_test.iterrows():
            print(f"Lag {int(lag)}: Statistic={row['lb_stat']:.4f}, p-value={row['lb_pvalue']:.4f}")
        print("Conclusion: No autocorrelation" if all(
            p > 0.05 for p in lb_test['lb_pvalue']) else "Conclusion: Autocorrelation present")


if __name__ == '__main__':
    np.random.seed(42)
    data_explorer = DataExplorer()
    data_explorer.get_statistics()
    data_explorer.plot_distribution()
    data_explorer.detect_outliers()
    data_explorer.decompose()
    data_explorer.plot_acf()
    data_explorer.plot_pacf()
    data_explorer.get_statistical_tests()
