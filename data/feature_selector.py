import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from data.feature_generator import FeatureGenerator

FEATURE_NAME = 'T10YIE'


class FeatureSelector:
    def __init__(self, path='/Users/arynaartsiukevich/PycharmProjects/inflation_rate_time_series',
                 file_name='T10YIE.csv'):
        self.file_name = file_name
        self.data_path = os.path.join(path, 'input', self.file_name)
        self.plots_path = os.path.join(path, 'data', 'plots')
        self.df_features = pd.read_csv(self.data_path, parse_dates=True, index_col='observation_date')

        self.forward_selected_features = []
        self.forward_r2_scores = []
        self.backward_selected_features = []
        self.backward_r2_scores = []

    def forward_selection(self, max_features=20, improvement_threshold=0.005, cv_folds=5):
        target = pd.Series(self.df_features[f'{FEATURE_NAME}']).shift(-1)
        valid_indices = self.df_features.index.intersection(target.dropna().index)
        X = self.df_features.loc[valid_indices]
        y = target.loc[valid_indices].values

        available_features = list(X.columns)
        current_r2 = -float('inf')

        print("Starting Forward Selection...")
        for i in range(max_features):
            best_r2 = -float('inf')
            best_feature = None
            scores = []

            for feature in available_features:
                trial_features = self.forward_selected_features + [feature]
                X_trial = X[trial_features]

                model = LinearRegression()
                r2_scores = cross_val_score(model, X_trial, y, cv=cv_folds, scoring='r2')
                r2 = r2_scores.mean()
                scores.append((feature, r2))

                if r2 > best_r2:
                    best_r2 = r2
                    best_feature = feature

            if i == 0:
                improvement = float('inf')
            else:
                improvement = (best_r2 - current_r2) / (1 - current_r2 + 1e-6)

            if improvement < improvement_threshold and i > 0:
                print(f"Stopping: Improvement {improvement * 100:.2f}% < {improvement_threshold * 100}%")
                break

            if best_feature is None:
                print("No further improvement possible.")
                break

            self.forward_selected_features.append(best_feature)
            self.forward_r2_scores.append(best_r2)
            current_r2 = best_r2
            available_features.remove(best_feature)
            print(
                f"Iteration {i + 1}: Added {best_feature}, R² = {best_r2:.4f}, Improvement = {improvement * 100:.2f}%")

        print(f"Selected {len(self.forward_selected_features)} features: {self.forward_selected_features}")
        print(f"Final R²: {self.forward_r2_scores[-1]:.4f}")

    def plot_forward_selection(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.forward_r2_scores) + 1), self.forward_r2_scores, marker='o', linestyle='-',
                 color='b')
        plt.xlabel('Number of Features')
        plt.ylabel('Cross-Validated R²')
        plt.title('Forward Selection: R² vs Number of Features')
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_path, 'forward_selection_r2.png'))
        plt.show()

    def backward_selection(self, min_features=20, degradation_threshold=0.001, cv_folds=5):
        target = pd.Series(self.df_features[f'{FEATURE_NAME}']).shift(-1)
        valid_indices = self.df_features.index.intersection(target.dropna().index)
        X = self.df_features.loc[valid_indices]
        y = target.loc[valid_indices].values

        available_features = list(X.columns)
        self.backward_selected_features = available_features.copy()

        model = LinearRegression()
        initial_r2_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
        current_r2 = initial_r2_scores.mean()
        self.backward_r2_scores.append(current_r2)
        print("Starting Backward Selection...")
        print(f"Initial R² with all {len(available_features)} features: {current_r2:.6f}")

        while len(self.backward_selected_features) > min_features:
            best_r2 = -float('inf')
            best_feature_to_remove = None
            scores = []

            for feature in self.backward_selected_features:
                trial_features = [f for f in self.backward_selected_features if f != feature]
                X_trial = X[trial_features]

                model = LinearRegression()
                r2_scores = cross_val_score(model, X_trial, y, cv=cv_folds, scoring='r2')
                r2 = r2_scores.mean()
                scores.append((feature, r2))

                if r2 > best_r2:
                    best_r2 = r2
                    best_feature_to_remove = feature

            degradation = (current_r2 - best_r2) / (1 - best_r2 + 1e-6)

            if degradation > degradation_threshold:
                print(f"Stopping: Degradation {degradation * 100:.2f}% > {degradation_threshold * 100}%")
                break

            if best_feature_to_remove is None:
                print("No feature can be removed without significant degradation.")
                break

            self.backward_selected_features.remove(best_feature_to_remove)
            self.backward_r2_scores.append(best_r2)
            current_r2 = best_r2
            print(f"Iteration {len(available_features) - len(self.backward_selected_features) + 1}: "
                  f"Removed {best_feature_to_remove}, R² = {best_r2:.4f}, Degradation = {degradation * 100:.2f}%")

        print(
            f"Backward Selection: Selected {len(self.backward_selected_features)} features: {self.backward_selected_features}")
        print(f"Final R²: {self.backward_r2_scores[-1]:.4f}")

    def plot_backward_selection(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.backward_r2_scores) + 1), self.backward_r2_scores, marker='o', linestyle='-',
                 color='b')
        plt.xlabel('Number of Features')
        plt.ylabel('Cross-Validated R²')
        plt.title('Backward Selection: R² vs Number of Features')
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_path, 'backward_selection_r2.png'))
        plt.show()

    def get_selected_features(self, method='forward'):
        if method == 'forward':
            self.forward_selection()
            self.plot_forward_selection()
            return self.forward_selected_features
        elif method == 'backward':
            self.backward_selection()
            self.plot_backward_selection()
            return self.backward_selected_features
        else:
            self.forward_selection()
            self.backward_selection()
            return self.forward_selected_features if self.forward_r2_scores[-1] > self.backward_r2_scores[
                -1] else self.backward_selected_features

    def compare_feature_sets(self):
        forward_set = set(self.forward_selected_features)
        backward_set = set(self.backward_selected_features)
        all_features = forward_set | backward_set
        intersection = forward_set & backward_set
        data = {
            'Feature': list(forward_set | backward_set),
            'Forward Selection': [1 if f in forward_set else 0 for f in all_features],
            'Backward Selection': [1 if f in backward_set else 0 for f in all_features]
        }
        df = pd.DataFrame(data)
        print("\nFeature Set Comparison:")
        print(df)
        df.to_csv(os.path.join(self.plots_path, 'feature_set_comparison.csv'), index=False)

        print("\nCommon Features:")
        print(intersection)

        self.plot_correlated_features(method='forward')
        self.plot_correlated_features(method='backward')
        self.plot_correlated_features(method='common')

        return df

    def plot_correlated_features(self, method='forward'):
        if method == 'forward':
            selected_features = self.forward_selected_features
        elif method == 'backward':
            selected_features = self.backward_selected_features
        else:
            selected_features = set(self.forward_selected_features) & set(self.forward_selected_features)

        corr_matrix = self.df_features.corr(method='pearson')
        feature_corr = corr_matrix.loc[selected_features, selected_features]
        print(f'\n {method} Selection Correlation Matrix:')
        print(feature_corr.round(4))
        feature_corr.to_csv(os.path.join(self.plots_path, f'{method}_correlations.csv'))
        plt.figure(figsize=(10, 8))
        sns.heatmap(feature_corr, annot=False, cmap='coolwarm')
        plt.title(f'Correlation Heatmap: {method} Selection Features')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, f'{method}_correlations.png'))
        plt.show()


if __name__ == '__main__':
    np.random.seed(42)
    #
    # feature_generator = FeatureGenerator()
    # feature_generator.generate_features()
    # feature_generator.save_expanded_df()

    feature_selector = FeatureSelector()
    feature_selector.forward_selection()
    feature_selector.plot_forward_selection()
    feature_selector.backward_selection()
    feature_selector.plot_backward_selection()
    feature_selector.compare_feature_sets()
