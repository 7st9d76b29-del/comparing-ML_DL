import argparse
import os
import pandas as pd
import numpy as np
import pickle
from scipy.stats import randint, loguniform

# Model Imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from nested_cv_dl import NestedCVEngine
N_CORES = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

MODELS = {
    "et": (ExtraTreesClassifier, {'n_estimators': randint(10, 1000), 'max_features': ['sqrt', 'log2', None], 'max_depth': [None] + list(range(5, 100))}),
    "ada": (AdaBoostClassifier, {'n_estimators': randint(10, 1000), 'learning_rate': loguniform(1e-5, 1)}),
    # "lgbm": (LGBMClassifier, {'n_estimators': randint(10, 1000), 'learning_rate': loguniform(1e-5, 1), 'max_depth': randint(3, 15), 'num_leaves': randint(20, 150)}),

    "dt": (DecisionTreeClassifier, {'max_depth': [None] + list(range(5, 100)), 'min_samples_leaf': randint(1, 5)}),
    "rf": (RandomForestClassifier, {'n_estimators': randint(10, 1000), 'max_features': ['sqrt', 'log2', None], 'max_depth': [None] + list(range(5, 100))}),
    "ridge": (RidgeClassifier, {'alpha': loguniform(1e-3, 1e2)}),
    "lr": (LogisticRegression, {'C': loguniform(1e-3, 1e2), 'penalty': ["l1","l2","elasticnet",None], 'solver': ["lbfgs","newton-cg","liblinear","sag", "saga"]}),
    "gnb": (GaussianNB, {'var_smoothing': loguniform(1e-15, 1e-2)}),
    "l_svc": (LinearSVC, {'C': loguniform(1e-3, 1e3), 'penalty': ['l1', 'l2'] }),
    "svc": (SVC, {'C': loguniform(1e-1, 1e3), 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': ['scale', 'auto']}),
    "bnb": (BernoulliNB, {'alpha': loguniform(1e-15, 1)}),
     
     # Fixed loguniform(0,1) to (1e-15,1) to avoid log(0) error
    # "mnb": (MultinomialNB, {'alpha': loguniform(1e-15, 1)}), # Fixed loguniform(0,1)
    "gb": (GradientBoostingClassifier, {'n_estimators': randint(10,1000), 'learning_rate': loguniform(1e-5, 1), 'max_depth': randint(1, 5)}),
    "xgb": (XGBClassifier, {'n_estimators': randint(10, 1000), 'learning_rate': loguniform(1e-5, 1), 'reg_alpha': loguniform(0.5, 1)}),
    "cat": (CatBoostClassifier, {'depth': randint(4, 10), 'learning_rate': loguniform(1e-5, 1), 'iterations': randint(10,1000)}),
    "knn": (KNeighborsClassifier, {'n_neighbors': randint(1, 100), 'leaf_size': randint(1, 100), 'p': [1, 2]}),
    "mlp": (MLPClassifier, {'hidden_layer_sizes': [(50,30,10), (50,30), (50,), (30,), (10,)], 'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': [ 'sgd', 'adam','lbfgs'],
                             'alpha': loguniform(1e-5,1e-1), 'learning_rate': ['constant', 'adaptive', 'invscaling']}),

    "gb_lin": (XGBClassifier, {'booster': ['gblinear'], 'n_estimators': randint(10, 1000), 'learning_rate': loguniform(1e-5, 1), 'reg_alpha': loguniform(1e-3, 1e2), 'reg_lambda': loguniform(1e-3, 1e2)}),
    
    # <-- Added Linear Discriminant Analysis (LDA)
    "lda": (LinearDiscriminantAnalysis, 
        # Configuration A: Test the 'auto' shrinkage method
        {'solver': ['lsqr', 'eigen'], 'shrinkage': ['auto'] + list(np.random.uniform(0, 1, size=300))}
    ),
}

# Get SLURM cores

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
    # IMPORTANT: Extract group IDs before dropping the column
    if 'group' in df.columns:
        groups = df['group'].values
        y = df['label']
        X = df.drop(columns=['label', 'group'])
    else:
        y = df['label']
        X = df.drop(columns=['label'])

    
    model_class, params = MODELS[args.model]
    
    # 3. Updated Model Initialization for CPU and Modern Libraries
    if args.model in ["xgb", "gb_lin"]:
        xgb_args = {
            'eval_metric': 'mlogloss', 
            'random_state': args.rs,
            'n_jobs': N_CORES
        }
        # Only tree-based XGBoost uses hist method; gb_lin ignores this
        if args.model == "xgb":
            xgb_args['tree_method'] = 'hist'
            
        model_obj = model_class(**xgb_args)
        current_n_jobs = 1

    elif args.model == "cat":
        model_obj = model_class(
            task_type='CPU', # Switched to CPU
            thread_count=4, # Tell CatBoost to use SLURM cores
            verbose=0, 
            random_seed=args.rs
        )
        current_n_jobs = 1 # CatBoost handles its own parallelism internally via thread_count
    elif args.model == "lgbm":
        model_obj = model_class(
            random_state=args.rs,
            n_jobs=N_CORES, # LightGBM parallelizes internally
            verbose=-1      # Suppress verbose output logs
        )
        current_n_jobs = 1
    elif args.model in ["rf", "et"]:
        model_obj = model_class(
            random_state=args.rs,
            n_jobs=N_CORES # Parallel trees
        )
        current_n_jobs = 1 # Sequential folds to save memory
    elif args.model in ["gnb","knn", "bnb","lda"]:
        model_obj = model_class()
        current_n_jobs = N_CORES

    elif args.model == "l_svc":
        model_obj = model_class(dual='auto', max_iter=1000, random_state=args.rs)
        current_n_jobs = N_CORES

    elif args.model == "mlp":
        model_obj = model_class(max_iter=200, random_state=args.rs)
        current_n_jobs = N_CORES

    elif args.model == "lr":
        model_obj = model_class(max_iter=100, random_state=args.rs)
        current_n_jobs = N_CORES

    elif args.model == "svc":
        model_obj = model_class(max_iter=10000, random_state=args.rs, cache_size=1000)
        current_n_jobs = N_CORES

    else:
        model_obj = model_class(random_state=args.rs)
        current_n_jobs = N_CORES

    output_dir = os.path.join("./results", dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    engine = NestedCVEngine(model_obj, params, output_dir, dataset_name, n_jobs=current_n_jobs)

    print(f"--- STARTING EXPERIMENT ---")
    print(f"Dataset: {dataset_name} | Model: {args.model} | RS: {args.rs} | Cores: {current_n_jobs}")
    
    # Pass the data to the engine
    if 'group' in df.columns:
        engine.run_experiment(X=X, y=y, groups=groups,  rs=args.rs)
