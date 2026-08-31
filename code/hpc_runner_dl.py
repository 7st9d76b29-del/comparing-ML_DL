import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
import argparse
import os
import pandas as pd
import numpy as np
import pickle
from scipy.stats import randint, loguniform, uniform
from pytorch_tabnet.tab_model import TabNetClassifier
from nested_cv_dl import NestedCVEngine
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from pytorch_tabular.models import (
    TabTransformerConfig,
    NodeConfig,
)
from sklearn.base import BaseEstimator, ClassifierMixin
from tabpfn import TabPFNClassifier

class TunableTabNet(TabNetClassifier):
    """
    Wrapper to allow RandomizedSearchCV to tune max_epochs and learning rate natively.
    """
    def __init__(self, max_epochs=100, lr=0.02, **kwargs):
        self.max_epochs = max_epochs
        self.lr = lr
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs):
        # Dynamically inject the learning rate into TabNet's required dictionary format
        self.optimizer_params = {'lr': self.lr}
        # Force the chosen max_epochs down into the training loop
        return super().fit(X, y, max_epochs=self.max_epochs, **kwargs)

class TabularRNN(nn.Module):
    """
    Handles both LSTM and GRU for tabular data.
    """
    def __init__(self, input_dim, num_classes, hidden_dim=50, n_layers=1, rnn_type='lstm'):
        super(TabularRNN, self).__init__()
        self.rnn_type = rnn_type.lower()
        
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                               num_layers=n_layers, batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, 
                              num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.float() 
        x = x.unsqueeze(1)
        out, _ = self.rnn(x)
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

class TabularCNN1D(nn.Module):
    """
    1D CNN for Tabular data (e-nose).
    """
    def __init__(self, input_dim, num_classes, dropout=0.1):
        super(TabularCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.float()
        x = x.unsqueeze(2)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = out.squeeze(2)
        out = self.fc(out)
        return out

device_setting = 'cuda' if torch.cuda.is_available() else 'cpu'

lstm_estimator = NeuralNetClassifier(
    module=TabularRNN,
    module__rnn_type='lstm',
    module__input_dim=16,       
    module__num_classes=2,      
    criterion=nn.CrossEntropyLoss,
    optimizer=optim.Adam,
    max_epochs=20,
    batch_size=32,
    train_split=None,
    verbose=0,
    iterator_train__drop_last=True,
    device=device_setting
)

gru_estimator = NeuralNetClassifier(
    module=TabularRNN,
    module__rnn_type='gru',
    module__input_dim=16, 
    module__num_classes=2,
    criterion=nn.CrossEntropyLoss,
    optimizer=optim.Adam,
    max_epochs=20,
    batch_size=32,
    train_split=None,
    iterator_train__drop_last=True,
    verbose=0,
    device=device_setting
)

cnn_estimator = NeuralNetClassifier(
    module=TabularCNN1D,
    module__input_dim=16,
    module__num_classes=2,
    criterion=nn.CrossEntropyLoss,
    optimizer=optim.Adam,
    max_epochs=20,
    batch_size=32,
    train_split=None,
    iterator_train__drop_last=True,
    verbose=0,
    device=device_setting
)


class PyTorchTabularWrapper(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn compatible wrapper for pytorch_tabular models.
    """
    def __init__(self, model_name='tabtransformer', max_epochs=100, lr=0.01, batch_size=128, random_state=42):
        self.model_name = model_name
        self.max_epochs = max_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None

    def fit(self, X, y, **kwargs):
        num_features = X.shape[1]
        feature_cols = [f"feature_{i}" for i in range(num_features)]
        
        df_train = pd.DataFrame(X, columns=feature_cols)
        df_train['target'] = y

        # --- DYNAMIC BATCH SIZE SAFETY CHECK ---
        # Detect if the validation split math creates a remainder of 1 
        # (PyTorch Tabular defaults to a 0.2 val split)
        val_size = int(len(df_train) * 0.2)
        train_size = len(df_train) - val_size
        
        safe_batch_size = self.batch_size
        if val_size % safe_batch_size == 1 or train_size % safe_batch_size == 1:
            safe_batch_size += 2  # Shift batch size to avoid the PyTorch bug

        # --- REMOVED drop_last=True ---
        data_config = DataConfig(
            target=['target'], 
            continuous_cols=feature_cols,
            categorical_cols=[]
        )
        
        trainer_config = TrainerConfig(
            auto_lr_find=False, 
            batch_size=safe_batch_size,
            max_epochs=int(self.max_epochs),
            accelerator='auto',
            seed=self.random_state
        )
        
        optimizer_config = OptimizerConfig()

        if self.model_name == 'tabtransformer':
            model_config = TabTransformerConfig(
                task="classification",
                metrics=["accuracy"],
                learning_rate=float(self.lr)
            )
        elif self.model_name == 'node':
            model_config = NodeConfig(
                task="classification",
                metrics=["accuracy"],
                learning_rate=float(self.lr),
            )
        elif self.model_name == 'saint':
            from pytorch_tabular.models import FTTransformerConfig
            model_config = FTTransformerConfig(
                task="classification",
                metrics=["accuracy"],
                learning_rate=float(self.lr),
            )
        else:
            raise ValueError(f"Unknown model_name for PyTorch Tabular: {self.model_name}")

        self.tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            verbose=False,
            suppress_lightning_logger=True
        )
        
        self.tabular_model.fit(train=df_train)
        self.is_fitted_ = True 
        
        return self

    def predict(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, attributes=["is_fitted_"])

        num_features = X.shape[1]
        feature_cols = [f"feature_{i}" for i in range(num_features)]
        df_test = pd.DataFrame(X, columns=feature_cols)
        
        preds_df = self.tabular_model.predict(df_test) 
        
        if 'prediction' in preds_df.columns:
            return preds_df['prediction'].values
            
        pred_cols = [col for col in preds_df.columns if str(col).endswith('_prediction')]
        if pred_cols:
            return preds_df[pred_cols[0]].values
            
        raise KeyError(f"Could not find prediction column. Available columns: {preds_df.columns.tolist()}")

    def predict_proba(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, attributes=["is_fitted_"])

        num_features = X.shape[1]
        feature_cols = [f"feature_{i}" for i in range(num_features)]
        df_test = pd.DataFrame(X, columns=feature_cols)
        
        preds_df = self.tabular_model.predict(df_test) 
        
        proba_cols = [col for col in preds_df.columns if "probability" in col.lower() or "class_" in col.lower()]
        return preds_df[proba_cols].values
    
# Create un-instantiated baseline objects
tabtransformer_estimator = PyTorchTabularWrapper(model_name='tabtransformer')
node_estimator = PyTorchTabularWrapper(model_name='node')
saint_estimator = PyTorchTabularWrapper(model_name='saint') 

# TabPFN is already an sklearn-compatible class
tabpfn_estimator = TabPFNClassifier(device=device_setting, n_estimators=16)

tabnet_estimator = TunableTabNet(verbose=0, seed=42)

MODELS=({
    "lstm": (lstm_estimator, {
        'module__hidden_dim': randint(50, 201),  
        'module__n_layers': randint(1, 4),       
        'lr': loguniform(1e-2, 0.5)              
    }),
    "gru": (gru_estimator, {
        'module__hidden_dim': randint(20, 51),   
        'module__n_layers': randint(1, 3),       
        'lr': loguniform(1e-4, 1e-2)             
    }),
    "cnn": (cnn_estimator, {
        'module__dropout': uniform(0.01, 0.49),  
        'lr': loguniform(1e-4, 0.5)              
    }),
    "tabnet": (tabnet_estimator, {
        'max_epochs': randint(50, 151), 
        'lr': loguniform(1e-3, 1e-2)    
    }),
    "tabtransformer": (tabtransformer_estimator, {
        'max_epochs': randint(50, 151),       
        'lr': loguniform(1e-4, 1e-2)          
    }),
    "node": (node_estimator, {
        'max_epochs': randint(50, 151),       
        'lr': loguniform(1e-4, 5e-2)          
    }),
    "saint": (saint_estimator, {
        'max_epochs': randint(50, 151),       
        'lr': loguniform(1e-4, 5e-2)          
    }),
    "tabpfn": (tabpfn_estimator, {
        'n_estimators': randint(4, 33) 
    })
})

def load_data(path):
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True) 
    parser.add_argument("--rs", type=int, default=1)
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    dataset_name = os.path.basename(args.data_path).replace(".csv", "")
    
    # 1. Load Data
    df = load_data(args.data_path)
    
    # 2. Extract Groups, X, and y
    if 'group' in df.columns:
        groups = df['group'].values
        y = df['label']
        X = df.drop(columns=['label', 'group'])

    # 3. Define dynamic shape values before model setup
    num_features = X.shape[1]
    num_classes = len(np.unique(y))
    
    model_class, params = MODELS[args.model]
    
    # 4. Updated Model Initialization for CPU/GPU and Modern Libraries
    if args.model in ["lstm", "gru", "cnn"]:
        model_obj = model_class
        model_obj.set_params(
            module__input_dim=num_features,
            module__num_classes=num_classes,
            iterator_train__num_workers=0 
        )
        current_n_jobs = 1 
        
    elif args.model in ["tabnet","tabtransformer", "node", "saint"]:
        model_obj = model_class
        current_n_jobs = 1 
        
    else:
        model_obj = model_class(random_state=args.rs)
        current_n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", -1))

    output_dir = os.path.join("./results", dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    engine = NestedCVEngine(
        model_obj, 
        params, 
        output_dir, 
        dataset_name, 
        n_jobs=current_n_jobs
    )
    engine.model_name = args.model
    print(f"--- STARTING EXPERIMENT ---")
    print(f"Dataset: {dataset_name} | Model: {args.model} | RS: {args.rs} | Parallel Jobs: {current_n_jobs}")
    
    # Pass the data to the engine
    engine.run_experiment(X=X, y=y, groups=groups, rs=args.rs)