import os
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    balanced_accuracy_score, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

class NestedCVEngine:
    def __init__(self, model_estimator, param_dist, output_base, dataset_name, n_jobs=-1):
        self.model_estimator = model_estimator
        self.param_dist = param_dist
        self.output_base = output_base
        self.dataset_name = dataset_name
        self.model_name = model_estimator.__class__.__name__
        self.n_jobs = n_jobs
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()

    def calculate_g_mean(self, y_true, y_pred):
        recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
        return np.exp(np.mean(np.log(recalls + 1e-10)))

    def run_experiment(self, X, y, rs):
        fresh_clf = clone(self.model_estimator)
        
        # 1. Encode target 'y'
        y_encoded = self.label_encoder.fit_transform(y)
        class_labels = self.label_encoder.classes_
        self._save_label_mapping(rs)
        
        if isinstance(X, pd.DataFrame):
            X = X.values

        pipeline = Pipeline([
            ('scaler', self.scaler),
            ('clf', fresh_clf)
        ])
        
        adjusted_params = {f'clf__{k}': v for k, v in self.param_dist.items()}
        unique_class_labels = np.unique(y_encoded)
        is_binary = len(unique_class_labels) <= 2
        
        # Using standard Stratified K-Fold now
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=rs)
        results = []
        
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_encoded)):
            fold_start_time = time.time()
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
            
            search = RandomizedSearchCV(
                pipeline, adjusted_params, n_iter=300, cv=inner_cv,
                n_jobs=self.n_jobs, random_state=rs, scoring='balanced_accuracy',
                return_train_score=True
            )
            search.fit(X_train, y_train)

            y_pred_test = search.predict(X_test)
            y_pred_train = search.predict(X_train)

            cm = confusion_matrix(y_test, y_pred_test, labels=unique_class_labels)
            self._save_confusion_matrix(cm, class_labels, rs, fold + 1)
            model_size_kb = len(pickle.dumps(search.best_estimator_)) / 1024
            
            test_auc = np.nan
            train_auc = np.nan
            try:
                clf_internal = search.best_estimator_.named_steps['clf']
                if hasattr(clf_internal, "predict_proba"):
                    y_prob_test = search.predict_proba(X_test)
                    y_prob_train = search.predict_proba(X_train)
                    if is_binary:
                        test_auc = roc_auc_score(y_test, y_prob_test[:, 1])
                        train_auc = roc_auc_score(y_train, y_prob_train[:, 1])
                    else:
                        test_auc = roc_auc_score(y_test, y_prob_test, multi_class='ovr', average='macro')
                        train_auc = roc_auc_score(y_train, y_prob_train, multi_class='ovr', average='macro')
                elif hasattr(clf_internal, "decision_function"):
                    y_score_test = search.decision_function(X_test)
                    y_score_train = search.decision_function(X_train)
                    if is_binary:
                        test_auc = roc_auc_score(y_test, y_score_test)
                        train_auc = roc_auc_score(y_train, y_score_train)
                    else:
                        test_auc = roc_auc_score(y_test, y_score_test, multi_class='ovr', average='macro')
                        train_auc = roc_auc_score(y_train, y_score_train, multi_class='ovr', average='macro')
            except Exception as e:
                print(f"AUC Calculation skipped for {self.model_name} fold {fold+1}: {e}")
                
            fold_end_time = time.time()
            total_time_sec = fold_end_time - fold_start_time
            
            fold_results = {
                'RandomState': rs,
                'Fold': fold + 1,
                'Model_Size_KB': round(model_size_kb, 2),
                'Time_Seconds': round(total_time_sec, 2),

                'Test_Accuracy': accuracy_score(y_test, y_pred_test),
                'Test_Balanced_Acc': balanced_accuracy_score(y_test, y_pred_test),
                'Test_F1_Macro': f1_score(y_test, y_pred_test, average='macro', zero_division=0),
                'Test_GMean_Macro': self.calculate_g_mean(y_test, y_pred_test),
                'Test_Precision_Macro': precision_score(y_test, y_pred_test, average='macro', zero_division=0),
                'Test_Recall_Macro': recall_score(y_test, y_pred_test, average='macro', zero_division=0),
                'Test_AUC_Macro': test_auc,
                
                'Train_Accuracy': accuracy_score(y_train, y_pred_train),
                'Train_Balanced_Acc': balanced_accuracy_score(y_train, y_pred_train),
                'Train_F1_Macro': f1_score(y_train, y_pred_train, average='macro', zero_division=0),
                'Train_GMean_Macro': self.calculate_g_mean(y_train, y_pred_train),
                'Train_Precision_Macro': precision_score(y_train, y_pred_train, average='macro', zero_division=0),
                'Train_Recall_Macro': recall_score(y_train, y_pred_train, average='macro', zero_division=0),
                'Train_AUC_Macro': train_auc,
                
                'Best_Params': search.best_params_
            }
            results.append(fold_results)
            
        self._save_summary_csv(results, rs)

    def _save_confusion_matrix(self, cm, labels, rs, fold):
        path = os.path.join(self.output_base, self.model_name, f"rs_{rs}", "confusion_matrices")
        os.makedirs(path, exist_ok=True)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.to_csv(os.path.join(path, f"cm_fold_{fold}.csv"))

    def _save_label_mapping(self, rs):
        path = os.path.join(self.output_base, self.model_name, f"rs_{rs}")
        os.makedirs(path, exist_ok=True)
        mapping = pd.DataFrame({
            'Integer': range(len(self.label_encoder.classes_)),
            'Original_Label': self.label_encoder.classes_
        })
        mapping.to_csv(os.path.join(path, "label_mapping.csv"), index=False)

    def _save_summary_csv(self, results, rs):
        path = os.path.join(self.output_base, self.model_name, f"rs_{rs}")
        os.makedirs(path, exist_ok=True)
        pd.DataFrame(results).to_csv(os.path.join(path, "results.csv"), index=False)