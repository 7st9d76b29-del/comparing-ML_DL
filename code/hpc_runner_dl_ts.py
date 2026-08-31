import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import randint, loguniform, uniform
from sklearn.base import BaseEstimator, ClassifierMixin

from nested_cv_dl_ts import NestedCVEngine

N_CORES = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

# --- PyTorch Network Architectures ---
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, rnn_type='lstm', dropout=0.0):
        super().__init__()
        rnn_drop = dropout if num_layers > 1 else 0.0
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=rnn_drop)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=rnn_drop)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]) # Isolate last time step

class TCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.0):
        super().__init__()
        layers = []
        in_channels = input_dim
        for i in range(num_layers):
            layers.append(nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = hidden_dim
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2) # Switch to (Batch, Features, Sequence Length) for Conv1D
        out = self.network(x)
        out = out.mean(dim=2) # Global Average Pooling over time
        return self.fc(out)

# --- Scikit-Learn Wrapper for Deep Learning Models ---
class DeepTimeSeriesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_type='lstm', hidden_dim=64, num_layers=1,
                 lr=0.001, epochs=50, batch_size=128, dropout=0.2, 
                 max_seq_len=None, step_size=1, random_state=None):
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.max_seq_len = max_seq_len 
        self.step_size = step_size      
        self.random_state = random_state

    def _preprocess_sequences(self, X):
        """ Applies time-step hyperparameter slicing before passing to PyTorch """
        X_proc = X
        if self.step_size is not None and self.step_size > 1:
            X_proc = X_proc[:, ::self.step_size, :]
        if self.max_seq_len is not None:
            limit = min(self.max_seq_len, X_proc.shape[1])
            X_proc = X_proc[:, :limit, :]
        return X_proc

    def fit(self, X, y):
        torch.set_num_threads(1)
        
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        X_processed = self._preprocess_sequences(X)

        self.classes_ = np.unique(y)
        output_dim = len(self.classes_)
        input_dim = X_processed.shape[2] 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_type in ['lstm', 'gru']:
            self.model_ = RNNModel(input_dim, self.hidden_dim, output_dim, self.num_layers, self.model_type, self.dropout)
        elif self.model_type == 'tcn':
            self.model_ = TCNModel(input_dim, self.hidden_dim, output_dim, self.num_layers, self.dropout)

        self.model_.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        dataset = TensorDataset(torch.tensor(X_processed, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model_(batch_X), batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        self.model_.eval()
        X_processed = self._preprocess_sequences(X)
        dataset = TensorDataset(torch.tensor(X_processed, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        probs = []
        with torch.no_grad():
            for batch_X, in loader:
                batch_X = batch_X.to(self.device)
                probs.append(torch.softmax(self.model_(batch_X), dim=1).cpu().numpy())
        return np.vstack(probs)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

# --- Standard Class Definitions for Proper Pickling ---
class lstm(DeepTimeSeriesClassifier):
    pass

class gru(DeepTimeSeriesClassifier):
    pass

class tcn(DeepTimeSeriesClassifier):
    pass

# --- Deep Learning Model Dictionary & Search Grids ---
MODELS = {
    "lstm": (lstm, {
        'model_type': ['lstm'],
        'hidden_dim': randint(50, 201),
        'num_layers': [1, 2, 3],
        'lr': loguniform(1e-4, 1e-1),
        'dropout': uniform(0.1, 0.4),
        # 'max_seq_len': [50, 100, 200, None], 
        # 'step_size': [1, 2, 3]               
    }),
    "gru": (gru, {
        'model_type': ['gru'],
        'hidden_dim': randint(50, 201),
        'num_layers': [1, 2],
        'lr': loguniform(1e-4, 1e-1),
        'dropout': uniform(0.1, 0.4),
        # 'max_seq_len': [50, 100, 200, None], 
        # 'step_size': [1, 2, 3]               
    }),
    "tcn": (tcn, {
        'model_type': ['tcn'],
        'hidden_dim': randint(50, 201),
        'num_layers': randint(1, 4),
        'lr': loguniform(1e-4, 1e-1),
        'dropout': uniform(0.1, 0.4),
        # 'max_seq_len': [50, 100, 200, None], 
        # 'step_size': [1, 2, 3]               
    }),
}

def load_time_series_data(df):
    """ Extracts raw 3D sequences per measurement (Zero-padded to max length) """
    sensor_cols = [col for col in df.columns if col not in ['group', 'label']]
    sequences, labels = [], []
    
    for _, group_data in df.groupby('group', sort=False):
        labels.append(group_data['label'].iloc[0])
        sequences.append(group_data[sensor_cols].values)

    max_seq_len = max(len(seq) for seq in sequences)
    
    X = np.zeros((len(sequences), max_seq_len, len(sensor_cols)))
    for i, seq in enumerate(sequences):
        X[i, :len(seq), :] = seq
        
    return X, pd.Series(labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "gru", "tcn"]) 
    parser.add_argument("--rs", type=int, default=1)
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    dataset_name = os.path.basename(args.data_path).replace(".csv", "")
    df = pd.read_csv(args.data_path)
    
    print(f"Loading raw time-series sequences...")
    X, y = load_time_series_data(df)
    
    print(f"Deep Learning mode: Using 3D shape {X.shape}: {X.shape[0]} sequences, {X.shape[1]} maximum timesteps, {X.shape[2]} sensors.")

    model_class, params = MODELS[args.model]
    
    model_obj = model_class(model_type=args.model, epochs=50, random_state=args.rs)
    current_n_jobs = 1
    n_iter_search = 50  
    

    output_dir = os.path.join("./results_ts", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    engine = NestedCVEngine(model_obj, params, output_dir, dataset_name, n_jobs=current_n_jobs, n_iter=n_iter_search)

    print(f"--- STARTING EXPERIMENT ---")
    print(f"Dataset: {dataset_name} | Model: {args.model} | RS: {args.rs} | Cores: {current_n_jobs} | CV Iterations: {n_iter_search}")
    
    engine.run_experiment(X=X, y=y, rs=args.rs)