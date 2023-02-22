import json
import joblib
from joblib import dump

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import optuna
from optuna.samplers import TPESampler
from sklearn.decomposition import PCA

optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import ShuffleSplit

from sklearn.compose import make_column_selector as selector

from sklearn.linear_model import LogisticRegressionCV
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")
import src.ModPerf as mdp

# from omegaconf import DictConfig, OmegaConf
# from hydra.core.config_store import ConfigStore

# class ColumnSelector(BaseEstimator, TransformerMixin):
#     """Select only specified columns."""
#     def __init__(self, columns):
#         self.columns = columns
        
#     def fit(self, X, y=None):
#         return self
    
#     def transform(self, X):
#         return X[self.columns]

class SklearnModeller:
    def __init__(
        self, 
        processed_data_save_path, 
        sklearn_models_path,
        n_optuna_trials,
        n_cv_splits,
        random_state
        ):
        # `features_from must be hyperparam: 
        # `lasso`, `xgb` or ...

        self.random_state = random_state
        self.n_optuna_trials = n_optuna_trials
        self.n_cv_splits = n_cv_splits
        self.sklearn_models_path = sklearn_models_path
        self.X_train = pd.read_pickle(processed_data_save_path + "/X_train.pkl")
        self.y_train = pd.read_pickle(processed_data_save_path + "/y_train.pkl")
        self.X_test = pd.read_pickle(processed_data_save_path + "/X_test.pkl")
        self.y_test = pd.read_pickle(processed_data_save_path + "/y_test.pkl")


    def __call__(self):

        def callback(study, trial):
            if study.best_trial.number == trial.number:
                study.set_user_attr(key="best_model", value=trial.user_attrs["best_model"])

        CROSS_VAL = ShuffleSplit(n_splits=self.n_cv_splits, test_size = 0.3, 
            train_size = 0.5, random_state = self.random_state)

        # Model bodies
        def lasso_objective(CROSS_VAL=CROSS_VAL):

            def objective(trial, CROSS_VAL=CROSS_VAL):

                Cs = int(trial.suggest_categorical("Cs", [1, 10, 100, 500, 1000, 1500]))

                clf = LogisticRegressionCV(
                    Cs=Cs,
                    penalty = 'l1',
                    solver = 'liblinear',
                    cv = CROSS_VAL, 
                    random_state = self.random_state,
                    n_jobs = -1,
                    scoring = 'roc_auc'
                    )

                clf.fit(self.X_train, self.y_train)
                y_pred = clf.predict_proba(self.X_train)
                trial.set_user_attr(key="best_model", value=clf)
                
                return roc_auc_score(self.y_train, y_pred[:,1])
                
            sampler = TPESampler(seed=self.random_state)
            study = optuna.create_study(sampler=sampler, direction='maximize')
            study.optimize(objective, 
                n_trials=self.n_optuna_trials, 
                callbacks=[callback], 
                n_jobs=-1)
            best_model = study.user_attrs["best_model"]
            trial = study.best_trial

            best_model.fit(self.X_train, self.y_train)
            dump(best_model, self.sklearn_models_path + '/Lasso.joblib')
            
            mdp.output_model_log(
                self.y_test, 
                best_model.predict_proba(self.X_test)[:,1], 
                self.sklearn_models_path + '/Models_info/' + '_Lasso_',
                trial.params)
                        
            print('LASSO ready...')

        def xgb_objective(CROSS_VAL=CROSS_VAL):

            def objective(trial,CROSS_VAL=CROSS_VAL):

                importance_type = trial.suggest_categorical("importance_type", \
                    ['gain', 'weight', 'cover', 'total_gain', 'total_cover'])
                max_depth = int(trial.suggest_categorical("max_depth", [7, 15, 20]))
                 

                clf = XGBClassifier(
                    n_estimators = 1000,
                    random_state = self.random_state,
                    tree_method = 'hist',
                    n_jobs = -1, 
                    enable_categorical = True,
                    importance_type = importance_type,
                    max_depth = max_depth

                    )

                for train_index, _ in CROSS_VAL.split(self.X_train):
                    X_A = self.X_train.iloc[train_index, :]
                    y_A = self.y_train.iloc[train_index]
                    clf.fit(X_A, y_A)

                y_pred = clf.predict_proba(self.X_train)
                trial.set_user_attr(key="best_model", value=clf)
                
                return roc_auc_score(self.y_train, y_pred[:,1])
                
            sampler = TPESampler(seed=self.random_state)
            study = optuna.create_study(sampler=sampler, direction='maximize')
            study.optimize(objective, 
                n_trials=self.n_optuna_trials, 
                callbacks=[callback], 
                n_jobs=-1)
            best_model = study.user_attrs["best_model"]
            trial = study.best_trial

            best_model.fit(self.X_train, self.y_train)
            dump(best_model, self.sklearn_models_path + '/Xgb.joblib')
            
            mdp.output_model_log(
                self.y_test, 
                best_model.predict_proba(self.X_test)[:,1], 
                self.sklearn_models_path + '/Models_info/' + '_XGB_',
                trial.params)
                        
            print('XGB ready...')

        def exrt_objective(CROSS_VAL=CROSS_VAL):

            def objective(trial, CROSS_VAL=CROSS_VAL):
                
                max_depth = int(trial.suggest_categorical("max_depth", [7, 15, 20]))

                clf = ExtraTreesClassifier(
                    n_estimators = 1000,
                    random_state = self.random_state,
                    n_jobs = -1,
                    max_depth = max_depth
                    )

                for train_index, _ in CROSS_VAL.split(self.X_train):
                    X_A = self.X_train.iloc[train_index, :]
                    y_A = self.y_train.iloc[train_index]
                    clf.fit(X_A, y_A)

                y_pred = clf.predict_proba(self.X_train)
                trial.set_user_attr(key="best_model", value=clf)
                
                return roc_auc_score(self.y_train, y_pred[:,1])
                
            sampler = TPESampler(seed=self.random_state)
            study = optuna.create_study(sampler=sampler, direction='maximize')
            study.optimize(objective, 
                n_trials=self.n_optuna_trials, 
                callbacks=[callback], 
                n_jobs=-1)
            best_model = study.user_attrs["best_model"]
            trial = study.best_trial

            best_model.fit(self.X_train, self.y_train)
            dump(best_model, self.sklearn_models_path + '/EXRT.joblib')
            
            mdp.output_model_log(
                self.y_test, 
                best_model.predict_proba(self.X_test)[:,1], 
                self.sklearn_models_path + '/Models_info/' + '_EXRT_',
                trial.params)
                        
            print('EXRT ready...')

        def svc_objective(CROSS_VAL=CROSS_VAL):

            def objective(trial,CROSS_VAL=CROSS_VAL):
                
                C = int(trial.suggest_categorical("C", [1, 5, 10, 100]))
                kernel = trial.suggest_categorical("kernel", ["rbf", "sigmoid"])

                clf = SVC(
                    C = C,
                    kernel = kernel,
                    gamma = "auto",
                    probability = True,
                    random_state = self.random_state,
                    )


                for train_index, _ in CROSS_VAL.split(self.X_train):
                    X_A = self.X_train.iloc[train_index, :]
                    y_A = self.y_train.iloc[train_index]
                    clf.fit(X_A, y_A)

                y_pred = clf.predict_proba(self.X_train)
                trial.set_user_attr(key="best_model", value=clf)
                
                return roc_auc_score(self.y_train, y_pred[:,1])
                
            sampler = TPESampler(seed=self.random_state)
            study = optuna.create_study(sampler=sampler, direction='maximize')
            study.optimize(objective, 
                n_trials=self.n_optuna_trials, 
                callbacks=[callback], 
                n_jobs=-1)
            best_model = study.user_attrs["best_model"]
            trial = study.best_trial

            best_model.fit(self.X_train, self.y_train)
            dump(best_model, self.sklearn_models_path + '/SVC.joblib')
            
            mdp.output_model_log(
                self.y_test, 
                best_model.predict_proba(self.X_test)[:,1], 
                self.sklearn_models_path + '/Models_info/' + '_SVC_',
                trial.params)
                        
            print('SVC ready...')

        lasso_objective()
        xgb_objective()
        exrt_objective()
        svc_objective()