import os

import numpy as np
import pandas as pd

from model.gradient_boosting_builder import GradientBoostingRegressor
from model.linear_regression_builder import LinearRegressionBuilder
from model.lstm_builder import LSTM_Builder

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

class ModelExecutor:

    def __init__(self, path='/Users/arynaartsiukevich/PycharmProjects/inflation_rate_time_series'):
        self.GB_NAME = 'gradient_boosting'
        self.LR_NAME = 'linear_regression'
        self.LSTM_NAME = 'lstm'

        self.models = {self.GB_NAME: GradientBoostingRegressor(), self.LR_NAME: LinearRegressionBuilder(),
                       self.LSTM_NAME: LSTM_Builder()}
        self.metrics = {}

        self.plots_path = os.path.join(path, 'data', 'plots')

    def train_models(self):
        self.models[self.GB_NAME].train_gradient_boosting_regressor()
        self.metrics[self.GB_NAME] = self.models[self.GB_NAME].evaluate_gradient_boosting_regressor()

        self.models[self.LSTM_NAME].tune_lstm_model(early_stopping_rounds=20, n_trials=15)
        self.metrics[self.LSTM_NAME] = self.models[self.LSTM_NAME].evaluate_lstm_model()

        self.models[self.LR_NAME].train_linear_regression_model()
        self.metrics[self.LR_NAME] = self.models[self.LR_NAME].evaluate_linear_regression_model()

    def compare_models(self):
        self.train_models()
        metrics_df = pd.DataFrame({
            'Model': [f'{self.LSTM_NAME}', f'{self.LR_NAME}', f'{self.GB_NAME}'],
            'R²': [self.metrics.get(f'{self.LSTM_NAME}', {}).get('R²', np.nan),
                   self.metrics.get(f'{self.LR_NAME}', {}).get('R²', np.nan),
                   self.metrics.get(f'{self.GB_NAME}', {}).get('R²', np.nan)],
            'MSE': [self.metrics.get(f'{self.LSTM_NAME}', {}).get('MSE', np.nan),
                    self.metrics.get(f'{self.LR_NAME}', {}).get('MSE', np.nan),
                    self.metrics.get(f'{self.GB_NAME}', {}).get('MSE', np.nan)],
            'MAE': [self.metrics.get(f'{self.LSTM_NAME}', {}).get('MAE', np.nan),
                    self.metrics.get(f'{self.LR_NAME}', {}).get('MAE', np.nan),
                    self.metrics.get(f'{self.GB_NAME}', {}).get('MAE', np.nan)],
            'RMSE': [self.metrics.get(f'{self.LSTM_NAME}', {}).get('RMSE', np.nan),
                     self.metrics.get(f'{self.LR_NAME}', {}).get('RMSE', np.nan),
                     self.metrics.get(f'{self.GB_NAME}', {}).get('RMSE', np.nan)],
            'MAPE': [self.metrics.get(f'{self.LSTM_NAME}', {}).get('MAPE', np.nan),
                     self.metrics.get(f'{self.LR_NAME}', {}).get('MAPE', np.nan),
                     self.metrics.get(f'{self.GB_NAME}', {}).get('MAPE', np.nan)],
            'SMAPE': [self.metrics.get(f'{self.LSTM_NAME}', {}).get('SMAPE', np.nan),
                      self.metrics.get(f'{self.LR_NAME}', {}).get('SMAPE', np.nan),
                      self.metrics.get(f'{self.GB_NAME}', {}).get('SMAPE', np.nan)]
        })
        print("\nModel Comparison Metrics:")
        print(metrics_df)
        metrics_df.to_csv(os.path.join(self.plots_path, 'model_comparison.csv'), index=False)

        best_model = metrics_df.loc[metrics_df['R²'].idxmax()]['Model']
        print(f"\nBest Model (by R²): {best_model}")



if __name__ == '__main__':
    np.random.seed(42)
    model_executor = ModelExecutor()
    model_executor.train_models()
    model_executor.compare_models()
