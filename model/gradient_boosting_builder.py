import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from xgboost import XGBRegressor

from model.model_builder import ModelBuilder


class GradientBoostingRegressor(ModelBuilder):
    def __init__(self):
        super().__init__()
        self.model = None
        self.best_params = None
        self.best_r2 = None
        self.mse = None

        self.model_name = 'gradient_boosting'

    def train_gradient_boosting_regressor(self, param_grid=None, cv_folds=5, early_stopping_rounds=50):
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 300, 500, 1000],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }

        tscv = TimeSeriesSplit(n_splits=cv_folds)

        model = XGBRegressor(eval_metric='rmse', random_state=42)

        grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='r2', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        self.best_params = grid_search.best_params_
        self.best_r2 = grid_search.best_score_

        self.model = XGBRegressor(**self.best_params, eval_metric='rmse', random_state=42,
                                  early_stopping_rounds=early_stopping_rounds)
        self.model.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], verbose=False)

        mse_scores = -cross_val_score(grid_search.best_estimator_, self.X_train, self.y_train, cv=tscv,
                                      scoring='neg_mean_squared_error')
        self.mse = mse_scores.mean()

        print(f"Best parameters: {self.best_params}")
        print(f"Cross-Validated R²: {self.best_r2:.4f}")
        print(f"Cross-Validated MSE: {self.mse:.4f}")

    def evaluate_gradient_boosting_regressor(self):
        y_pred = self.model.predict(self.X_test)

        metrics = self.compute_metrics(self.y_test, y_pred)

        self.print_metrics(metrics)
        self.plot_results(y_pred=y_pred, y_true=self.y_test, model_name=self.model_name)
        self.explore_residuals(y_pred=y_pred, y_true=self.y_test, model_name=self.model_name)
        self.plot_feature_importance()
        return metrics

    def plot_feature_importance(self):
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]
        print([self.selected_features[i] for i in indices])
        print(importance)
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(self.selected_features)), importance[indices], align='center')
        plt.xticks(range(len(self.selected_features)),
                   [self.selected_features[i] for i in indices], rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Gradient Boosting Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, 'gbr_feature_importance.png'))


if __name__ == '__main__':
    np.random.seed(42)
    custom_xgboost = GradientBoostingRegressor()
    custom_xgboost.train_gradient_boosting_regressor()
    custom_xgboost.evaluate_gradient_boosting_regressor()
