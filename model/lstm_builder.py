import numpy as np
import optuna
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sympy.printing.pytorch import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from model.model_builder import ModelBuilder


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.dropout(lstm_out)
        out = self.dense(out)
        return out


class LSTM_Builder(ModelBuilder):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.best_params = None
        self.best_r2 = None
        self.mse = None

        self.train_losses = []
        self.val_losses = []

        self.model_name = 'lstm'

    def create_sequences(self, X, y, sequence_length):
        X = np.asarray(X)
        y = np.asarray(y)
        X_seq, y_seq = [], []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])
        return np.array(X_seq), np.array(y_seq)

    def train_lstm_model(self, sequence_length, batch_size, hidden_size, dropout, learning_rate,
                         early_stopping_rounds=20, epochs=100):
        X_train_seq, y_train_seq = self.create_sequences(self.X_train, self.y_train, sequence_length)
        X_val_seq, y_val_seq = self.create_sequences(self.X_val, self.y_val, sequence_length)

        X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).to(self.device)
        X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.model = LSTMModel(input_size=len(self.selected_features), hidden_size=hidden_size, dropout=dropout).to(
            self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        self.train_losses = []
        self.val_losses = []

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            train_loss /= len(train_loader.dataset)
            self.train_losses.append(train_loss)

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    output = self.model(X_batch)
                    val_loss += criterion(output.squeeze(), y_batch).item() * X_batch.size(0)
                val_loss /= len(val_loader.dataset)
                self.val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_rounds:
                    break

        self.model.load_state_dict(best_model_state)
        self.model.eval()

        with torch.no_grad():
            y_train_pred = self.model(X_train_tensor).cpu().numpy().squeeze()
        self.mse = mean_squared_error(y_train_seq, y_train_pred)

        with torch.no_grad():
            y_val_pred = self.model(X_val_tensor).cpu().numpy().squeeze()
        val_r2 = r2_score(y_val_seq, y_val_pred)

        return val_r2

    def objective(self, trial, early_stopping_rounds=20):
        params = {
            'sequence_length': trial.suggest_categorical('sequence_length', [10, 20, 30]),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32]),
            'hidden_size': trial.suggest_categorical('hidden_size', [50, 100]),
            'dropout': trial.suggest_categorical('dropout', [0.2, 0.3]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.001, 0.01])
        }
        val_r2 = self.train_lstm_model(**params, early_stopping_rounds=early_stopping_rounds)
        return val_r2

    def tune_lstm_model(self, early_stopping_rounds=20, n_trials=15):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)

        self.best_params = study.best_params
        self.best_r2 = study.best_value
        self.train_lstm_model(**self.best_params, early_stopping_rounds=early_stopping_rounds)

        print(f"Best parameters: {self.best_params}")
        print(f"Validation R²: {self.best_r2:.4f}")
        print(f"Train MSE: {self.mse:.4f}")

    def evaluate_lstm_model(self):
        X_test_seq, y_test_seq = self.create_sequences(self.X_test, self.y_test, self.best_params['sequence_length'])
        X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_test_tensor).cpu().numpy().squeeze()

        metrics = self.compute_metrics(y_test_seq, y_pred)
        self.print_metrics(metrics)
        self.plot_results(y_pred=y_pred, y_true=y_test_seq, model_name=self.model_name)
        self.explore_residuals(y_pred=y_pred, y_true=y_test_seq, model_name=self.model_name)
        self.plot_learning_curves()
        return metrics

    def plot_learning_curves(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Learning Curves for LSTM Model')
        plt.legend()
        plt.grid(True)
        plt.savefig('lstm_learning_curves.png')
        plt.show()


if __name__ == '__main__':
    np.random.seed(42)
    model = LSTM_Builder()
    model.tune_lstm_model(early_stopping_rounds=20, n_trials=15)
    model.evaluate_lstm_model()
