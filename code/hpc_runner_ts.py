import argparse
import os
import pandas as pd
import numpy as np
import pickle
from scipy.stats import randint, loguniform

# Model Imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 

from nested_cv_ts import NestedCVEngine

N_CORES = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

MODELS = {
    "et": (ExtraTreesClassifier, {'n_estimators': randint(10, 1000), 'max_features': ['sqrt', 'log2', None], 'max_depth': [None] + list(range(5, 100))}),
    "ada": (AdaBoostClassifier, {'n_estimators': randint(10, 1000), 'learning_rate': loguniform(1e-5, 1)}),
    "dt": (DecisionTreeClassifier, {'max_depth': [None] + list(range(5, 100)), 'min_samples_leaf': randint(1, 5)}),
    "rf": (RandomForestClassifier, {'n_estimators': randint(10, 1000), 'max_features': ['sqrt', 'log2', None], 'max_depth': [None] + list(range(5, 100))}),
    "ridge": (RidgeClassifier, {'alpha': loguniform(1e-3, 1e2)}),
    "lr": (LogisticRegression, {'C': loguniform(1e-3, 1e2), 'penalty': ["l1","l2","elasticnet",None], 'solver': ["lbfgs","newton-cg","liblinear","sag", "saga"]}),
    "gnb": (GaussianNB, {'var_smoothing': loguniform(1e-15, 1e-2)}),
    "l_svc": (LinearSVC, {'C': loguniform(1e-3, 1e3), 'penalty': ['l1', 'l2'] }),
    "svc": (SVC, {'C': loguniform(1e-1, 1e3), 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': ['scale', 'auto']}),
    "bnb": (BernoulliNB, {'alpha': loguniform(1e-15, 1)}),
    "gb": (GradientBoostingClassifier, {'n_estimators': randint(10,1000), 'learning_rate': loguniform(1e-5, 1), 'max_depth': randint(1, 5)}),
    "xgb": (XGBClassifier, {'n_estimators': randint(10, 1000), 'learning_rate': loguniform(1e-5, 1), 'reg_alpha': loguniform(0.5, 1)}),
    "cat": (CatBoostClassifier, {'depth': randint(4, 10), 'learning_rate': loguniform(1e-5, 1), 'iterations': randint(10,1000)}),
    "knn": (KNeighborsClassifier, {'n_neighbors': randint(1, 100), 'leaf_size': randint(1, 100), 'p': [1, 2]}),
    "mlp": (MLPClassifier, {'hidden_layer_sizes': [(50,30,10), (50,30), (50,), (30,), (10,)], 'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': [ 'sgd', 'adam','lbfgs'],
                             'alpha': loguniform(1e-5,1e-1), 'learning_rate': ['constant', 'adaptive', 'invscaling']}),
    "lda": (LinearDiscriminantAnalysis, {'solver': ['lsqr', 'eigen'], 'shrinkage': ['auto'] + list(np.random.uniform(0, 1, size=300))}),
}

def extract_time_series_features(df):
    """
    Groups data by 'group' and extracts transient/dynamic features 
    for each sensor column per time series measurement.
    """
    sensor_cols = [col for col in df.columns if col not in ['group', 'label']]
    
    feature_rows = []
    labels = []
    
    for group_id, group_data in df.groupby('group', sort=False):
        group_features = {}
        labels.append(group_data['label'].iloc[0])
        
        for col in sensor_cols:
            sensor_data = group_data[col].values
            
            s0 = sensor_data[0]
            s_max = np.max(sensor_data)
            delta_s = s_max - s0
            ratio_s = s_max / (s0 + 1e-8)
            
            diffs = np.diff(sensor_data) if len(sensor_data) > 1 else np.array([0.0])
            max_slope = np.max(diffs) if len(diffs) > 0 else 0.0
            min_slope = np.min(diffs) if len(diffs) > 0 else 0.0
            
            t_max = np.argmax(sensor_data)
            auc = np.trapezoid(sensor_data) if hasattr(np, 'trapezoid') else np.trapz(sensor_data)
            
            # Store extracted features per sensor
            group_features[f"{col}_S0"] = s0
            group_features[f"{col}_Smax"] = s_max
            group_features[f"{col}_DeltaS"] = delta_s
            group_features[f"{col}_RatioS"] = ratio_s
            group_features[f"{col}_MaxSlope"] = max_slope
            group_features[f"{col}_MinSlope"] = min_slope
            group_features[f"{col}_Tmax"] = t_max
            group_features[f"{col}_AUC"] = auc
            
        feature_rows.append(group_features)
        
    X_df = pd.DataFrame(feature_rows)
    y = pd.Series(labels)
    
    return X_df, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True) 
    parser.add_argument("--rs", type=int, default=1)
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    dataset_name = os.path.basename(args.data_path).replace(".csv", "")
    
    # 1. Load Raw Data
    df = pd.read_csv(args.data_path)
    
    # 2. Extract Time-Series Features per Group
    print(f"Extracting transient time-series features...")
    X, y = extract_time_series_features(df)
    print(f"Extracted {X.shape[1]} features across {X.shape[0]} unique time-series measurements.")

    model_class, params = MODELS[args.model]
    
    # 3. Model Initialization (Code untouched from previous)
    if args.model in ["xgb"]:
        xgb_args = {'eval_metric': 'mlogloss', 'random_state': args.rs, 'n_jobs': N_CORES}
        if args.model == "xgb":
            xgb_args['tree_method'] = 'hist'
        model_obj = model_class(**xgb_args)
        current_n_jobs = 1

    elif args.model == "cat":
        model_obj = model_class(task_type='CPU', thread_count=4, verbose=0, random_seed=args.rs)
        current_n_jobs = 1
    elif args.model == "lgbm":
        model_obj = model_class(random_state=args.rs, n_jobs=N_CORES, verbose=-1)
        current_n_jobs = 1
    elif args.model in ["rf", "et"]:
        model_obj = model_class(random_state=args.rs, n_jobs=N_CORES)
        current_n_jobs = 1
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

    output_dir = os.path.join("./results_ts", dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    engine = NestedCVEngine(model_obj, params, output_dir, dataset_name, n_jobs=current_n_jobs)

    print(f"--- STARTING EXPERIMENT ---")
    print(f"Dataset: {dataset_name} | Model: {args.model} | RS: {args.rs} | Cores: {current_n_jobs}")
    
    # 4. Run Standard Stratified Nested Cross-Validation
    engine.run_experiment(X=X, y=y, rs=args.rs)