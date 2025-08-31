import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from data.feature_selector import FeatureSelector


class ModelBuilder:
    def __init__(self, path='/Users/arynaartsiukevich/PycharmProjects/inflation_rate_time_series',
                 file_name='T10YIE.csv'):
        self.FEATURE_NAME = 'T10YIE'

        self.path = path
        self.file_name = file_name
        self.data_path = os.path.join(path, 'input', self.file_name)
        self.plots_path = os.path.join(path, 'data', 'plots')

        # self.selected_features = self._get_best_features()
        self.selected_features = ['T10YIE', 'is_peak', 'is_trough', 'lag_1', 'month_2', 'lag_4', 'month_12', 'tema_0.5', 'stoch_k', 'adx', 'max_14', 'dayofweek_3', 'q75_18', 'ema_ratio_0.7_0.3', 'stoch_d', 'kurt_20', 'lag_14', 'lag_13']
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self._get_train_test_val_split(
            train_size=0.6, val_size=0.2)

    def _get_best_features(self):
        feature_selector = FeatureSelector(path=self.path, file_name=self.file_name)
        return feature_selector.get_selected_features(method='forward')

    def _get_train_test_val_split(self, train_size=0.7, val_size=0.15):
        df_features = pd.read_csv(self.data_path, parse_dates=True, index_col='observation_date')
        df_features = df_features.sort_index()

        df_features[f'{self.FEATURE_NAME}_next_day'] = df_features[f'{self.FEATURE_NAME}'].shift(-1)
        df_features = df_features.dropna(subset=[f'{self.FEATURE_NAME}_next_day'])

        total_size = len(df_features)
        train_idx = int(total_size * train_size)
        val_idx = train_idx + int(total_size * val_size)

        self.X_train = df_features[self.selected_features].iloc[:train_idx]
        self.X_val = df_features[self.selected_features].iloc[train_idx:val_idx]
        self.X_test = df_features[self.selected_features].iloc[val_idx:]
        self.y_train = df_features[f'{self.FEATURE_NAME}_next_day'].iloc[:train_idx]
        self.y_val = df_features[f'{self.FEATURE_NAME}_next_day'].iloc[train_idx:val_idx]
        self.y_test = df_features[f'{self.FEATURE_NAME}_next_day'].iloc[val_idx:]

        # print(self.X_train.index)
        # print(self.X_val.index)
        # print(self.X_test.index)

        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def compute_metrics(self, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.inf

        denominator = np.abs(y_true) + np.abs(y_pred)
        smape = np.mean(2 * np.abs(y_true - y_pred) / denominator) * 100 if (denominator > 0).sum() > 0 else np.inf

        return {'R²': r2, 'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'SMAPE': smape}

    def print_metrics(self, metrics):
        print(f"Test R²: {metrics['R²']:.4f}")
        print(f"Test MSE: {metrics['MSE']:.4f}")
        print(f"Test MAE: {metrics['MAE']:.4f}")
        print(f"Test RMSE: {metrics['RMSE']:.4f}")
        print(f"Test MAPE: {metrics['MAPE']:.4f}%")
        print(f"Test SMAPE: {metrics['SMAPE']:.4f}%")

    def plot_results(self, y_pred, y_true, model_name):
        start_idx = len(self.y_test) - len(y_pred)
        dates_aligned = self.y_test.index[start_idx:]

        plt.figure(figsize=(12, 6))
        plt.plot(dates_aligned, y_true, label=f'Actual Day {self.FEATURE_NAME}', color='b')
        plt.plot(dates_aligned, y_pred, label=f'Predicted Day {self.FEATURE_NAME}', color='r', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel(f'{self.FEATURE_NAME}')
        plt.title(f'{model_name}: Actual vs Predicted {self.FEATURE_NAME} on Test Set')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_path, f'{model_name}_predictions.png'))

    def explore_residuals(self, y_pred, y_true, model_name):
        residuals = y_true - y_pred

        start_idx = len(self.y_test) - len(y_pred)
        time_index = self.y_test.index[start_idx:]

        self.plot_scatter_residuals(residuals, model_name)
        self.plot_time_residuals(time_index, residuals, model_name)
        self.plot_hist_residuals(residuals, model_name)
        self.get_largest_errors_period(time_index, residuals, model_name)

    def plot_scatter_residuals(self, residuals, model_name):
        plt.figure(figsize=(12, 6))
        plt.scatter(range(len(residuals)), residuals, color='b', label='Residuals')
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Residual (Actual - Predicted)')
        plt.title(f'Residual Plot: {model_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_path, f'{model_name}_residuals.png'))

    def plot_time_residuals(self, time_index, residuals, model_name):
        plt.figure(figsize=(12, 6))
        plt.plot(time_index, residuals, label='Residuals')
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f'Residuals over Time for {model_name}')
        plt.xlabel('Time')
        plt.ylabel('Residual')
        plt.legend()
        plt.savefig(os.path.join(self.plots_path, f'{model_name}_residuals_time.png'))

    def plot_hist_residuals(self, residuals, model_name):
        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=30, edgecolor='black')
        plt.title(f'Residual Distribution for {model_name}')
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.plots_path, f'{model_name}_residuals_histogram.png'))

    def get_largest_errors_period(self, time_index, residuals, model_name):
        error_df = pd.DataFrame({
            'time': time_index,
            'abs_residual': np.abs(residuals)
        })
        error_df['date'] = error_df['time'].dt.date
        daily_errors = error_df.groupby('date')['abs_residual'].mean().sort_values(ascending=False).head(10)

        print(f"\nPeriods with largest errors (top 10 days by mean absolute error) for {model_name}:")
        print(daily_errors)

        # daily_errors.to_csv(os.path.join(self.plots_path, f'{model_name}_high_error_periods.csv'))
        self.plot_largest_errors_period(daily_errors, model_name)
        return daily_errors

    def plot_largest_errors_period(self, daily_errors, model_name):
        plt.figure(figsize=(10, 6))
        daily_errors.plot(kind='bar')
        plt.title(f'Top 10 Days with Largest Errors for {model_name}')
        plt.xlabel('Date')
        plt.ylabel('Mean Absolute Error')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, f'{model_name}_high_error_periods.png'))

if __name__ == '__main__':
    np.random.seed(42)
    model_builder = ModelBuilder()
