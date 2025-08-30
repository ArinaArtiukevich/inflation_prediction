import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score

from model.model_builder import ModelBuilder


class LinearRegressionBuilder(ModelBuilder):
    def __init__(self):
        super().__init__()

        self.model = None
        self.best_alpha = None
        self.best_r2 = None
        self.mse = None
        self.coefficients = None

        self.model_name = 'linear_regression'

    def train_linear_regression_model(self, alpha_grid=(0.01, 0.1, 1, 10, 100, 1000), cv_folds=5):
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        self.model = Ridge()

        grid_search = GridSearchCV(self.model, {'alpha': list(alpha_grid)}, cv=tscv, scoring='r2', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        self.model = grid_search.best_estimator_
        self.best_alpha = grid_search.best_params_['alpha']
        self.best_r2 = grid_search.best_score_

        mse_scores = -cross_val_score(self.model, self.X_train, self.y_train, cv=tscv, scoring='neg_mean_squared_error')
        self.mse = mse_scores.mean()

        self.coefficients = pd.Series(self.model.coef_, index=self.selected_features)

        print(f"Best alpha: {self.best_alpha}")
        print(f"Cross-Validated R²: {self.best_r2:.4f}")
        print(f"Cross-Validated MSE: {self.mse:.4f}")
        print("Feature Coefficients:")
        print(self.coefficients)

    def evaluate_linear_regression_model(self):
        if not self.model:
            self.train_linear_regression_model()

        y_pred = self.model.predict(self.X_test)
        metrics = self.compute_metrics(self.y_test, y_pred)
        self.print_metrics(metrics)
        self.plot_results(y_pred=y_pred, y_true=self.y_test, model_name=self.model_name)
        self.explore_residuals(y_pred=y_pred, y_true=self.y_test, model_name=self.model_name)
        return metrics


if __name__ == '__main__':
    np.random.seed(42)
    custom_lr = LinearRegressionBuilder()
    custom_lr.train_linear_regression_model()
    custom_lr.evaluate_linear_regression_model()
