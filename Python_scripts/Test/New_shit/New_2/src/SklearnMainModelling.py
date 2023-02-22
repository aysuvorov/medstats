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

from sklearn.compose import make_column_selector as selector

from sklearn.linear_model import LogisticRegressionCV
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")
import src.ModPerf as mdp

from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore

# class ColumnSelector(BaseEstimator, TransformerMixin):
#     """Select only specified columns."""
#     def __init__(self, columns):
#         self.columns = columns
        
#     def fit(self, X, y=None):
#         return self
    
#     def transform(self, X):
#         return X[self.columns]

class Preprocessor:
    def __init__(self, data_load_path, features_from, features_len, random_state):
        # `features_from must be hyperparam: 
        # `lasso`, `xgb` or ...

        self.random_state = random_state
        self.X_train = pd.read_pickle(data_load_path + "/X_train.pkl")
        self.y_train = pd.read_pickle(data_load_path + "/y_train.pkl")
        self.X_test = pd.read_pickle(data_load_path + "/X_test.pkl")
        self.y_test = pd.read_pickle(data_load_path + "/y_test.pkl")

        def json_list_loader(fp):
            f = open(fp)
            content = json.load(f)
            f.close()
            return content

        if features_from == 'lasso':
            self.features = json_list_loader("./data/03_selected_features/Lasso_features.json")
        elif features_from == 'xgb':
            self.features = json_list_loader("./data/03_selected_features/Xgb_features.json")
        else:
            print('Wrong feature list!')

        self.X_train = self.X_train[self.features[:features_len]]
        self.X_test = self.X_test[self.features[:features_len]]


    def preprocess(self, USE_PCA = False):

        category_cols = [col for col in self.X_train.columns \
            if self.X_train[col].dtype == 'category']
        numeric_cols = [col for col in self.X_train.columns \
            if col not in category_cols]

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
                                    ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(
                    handle_unknown='ignore', 
                    drop = 'if_binary',
                    sparse = False))
                                    ])

        if USE_PCA:
            preprocessor = ColumnTransformer(
                        transformers=[
                            ("num", numeric_transformer, selector(dtype_exclude="category")),
                            ("cat", categorical_transformer, selector(dtype_include="category")),
                            ("pca", PCA(0.98), selector(dtype_exclude="object"))
                            ],
                            n_jobs = -1
                            )
            
        else:
            preprocessor = ColumnTransformer(
                        transformers=[
                            ("num", numeric_transformer, numeric_cols),
                            ("cat", categorical_transformer, category_cols)
                            ],
                            remainder='passthrough',
                            n_jobs = -1
                            )

        self.X_train_prep = pd.DataFrame(preprocessor.fit_transform(self.X_train))
        self.X_test_prep = pd.DataFrame(preprocessor.transform(self.X_test))

    def run(self, OPTUNA_TRIALS = 10):

        def callback(study, trial):
            if study.best_trial.number == trial.number:
                study.set_user_attr(key="best_model", value=trial.user_attrs["best_model"])

        # Output writer
        def output_model_log(real, pred, f_name, best_dict_params):
            text_file = open('./data/04_sklearn_models/Models_info/'+ f_name + ".txt", "w")
            text_file.write(f'\nMODEL PARAMS: {best_dict_params}')
            text_file.write(f'\nTest data ROC AUC score: {roc_auc_score(real, pred)}')
            text_file.write(f'\n')
            text_file.write(f'{mdp.ModPerf_thresholds(real, pred).to_markdown()}')
            text_file.close()

            with open('./data/04_sklearn_models/Models/' + f_name + '.npy', 'wb') as f:
                np.save(f, np.array([real, pred]))
            f.close()

        # Model bodies
        def lasso_objective():

            def objective(trial):
                Cs = int(trial.suggest_categorical("Cs", [1, 10, 100, 500, 1000, 1500]))

                clf = LogisticRegressionCV(
                    Cs=Cs,
                    penalty = 'l1',
                    solver = 'liblinear',
                    cv = 10, 
                    random_state = self.random_state,
                    n_jobs = -1,
                    scoring = 'roc_auc'
                    )

                # calibrator = CalibratedClassifierCV(
                #     clf, method = 'sigmoid', cv=7, n_jobs = -1)

                # pipe = Pipeline([
                #     ('calibrator', calibrator)
                #     ])

                clf.fit(self.X_train_prep, self.y_train)
                y_pred = clf.predict_proba(self.X_train_prep)
                trial.set_user_attr(key="best_model", value=clf)
                
                return roc_auc_score(self.y_train, y_pred[:,1])
                
            sampler = TPESampler(seed=self.random_state)
            study = optuna.create_study(sampler=sampler, direction='maximize')
            study.optimize(objective, 
                n_trials=OPTUNA_TRIALS, 
                callbacks=[callback], 
                n_jobs=-1)
            best_model = study.user_attrs["best_model"]
            trial = study.best_trial

            best_model.fit(self.X_train_prep, self.y_train)
            dump(best_model, './data/04_sklearn_models/Models/Lasso.joblib')
            
            output_model_log(
                self.y_test, 
                best_model.predict_proba(self.X_test_prep)[:,1], 
                '_Lasso_',
                trial.params)
                        
            print('LASSO ready...')

        def xgb_objective():

            # best_model.fit(self.X_train_prep, self.y_train)

            def objective(trial):

                importance_type = trial.suggest_categorical("importance_type", \
                    ['gain', 'weight', 'cover', 'total_gain', 'total_cover'])
                max_depth = int(trial.suggest_categorical("max_depth", [7, 15, 20]))
                 

                clf = XGBClassifier(
                    n_estimators = 500,
                    random_state = self.random_state,
                    tree_method = 'hist',
                    n_jobs = -1, 
                    enable_categorical = True,
                    importance_type = importance_type,
                    max_depth = max_depth

                    )

                clf.fit(self.X_train_prep, self.y_train)
                y_pred = clf.predict_proba(self.X_train_prep)
                trial.set_user_attr(key="best_model", value=clf)
                
                return roc_auc_score(self.y_train, y_pred[:,1])
                
            sampler = TPESampler(seed=self.random_state)
            study = optuna.create_study(sampler=sampler, direction='maximize')
            study.optimize(objective, 
                n_trials=OPTUNA_TRIALS, 
                callbacks=[callback], 
                n_jobs=-1)
            best_model = study.user_attrs["best_model"]
            trial = study.best_trial

            best_model.fit(self.X_train_prep, self.y_train)
            dump(best_model, './data/04_sklearn_models/Models/Xgb.joblib')
            
            output_model_log(
                self.y_test, 
                best_model.predict_proba(self.X_test_prep)[:,1], 
                '_XGB_',
                '_')
                        
            print('XGB ready...')

        def exrt_objective():

            def objective(trial):
                
                max_depth = int(trial.suggest_categorical("max_depth", [7, 15, 20]))

                clf = ExtraTreesClassifier(
                    n_estimators = 1000,
                    random_state = self.random_state,
                    n_jobs = -1,
                    max_depth = max_depth
                    )


                clf.fit(self.X_train_prep, self.y_train)
                y_pred = clf.predict_proba(self.X_train_prep)
                trial.set_user_attr(key="best_model", value=clf)
                
                return roc_auc_score(self.y_train, y_pred[:,1])
                
            sampler = TPESampler(seed=self.random_state)
            study = optuna.create_study(sampler=sampler, direction='maximize')
            study.optimize(objective, 
                n_trials=OPTUNA_TRIALS, 
                callbacks=[callback], 
                n_jobs=-1)
            best_model = study.user_attrs["best_model"]
            trial = study.best_trial

            best_model.fit(self.X_train_prep, self.y_train)
            dump(best_model, './data/04_sklearn_models/Models/EXRT.joblib')
            
            output_model_log(
                self.y_test, 
                best_model.predict_proba(self.X_test_prep)[:,1], 
                '_EXRT_',
                trial.params)
                        
            print('EXRT ready...')

        def svc_objective():

            def objective(trial):
                
                C = int(trial.suggest_categorical("C", [1, 5, 10, 100]))
                kernel = trial.suggest_categorical("kernel", ["rbf", "sigmoid"])

                clf = SVC(
                    C = C,
                    kernel = kernel,
                    gamma = "auto",
                    probability = True,
                    random_state = self.random_state,
                    )


                clf.fit(self.X_train_prep, self.y_train)
                y_pred = clf.predict_proba(self.X_train_prep)
                trial.set_user_attr(key="best_model", value=clf)
                
                return roc_auc_score(self.y_train, y_pred[:,1])
                
            sampler = TPESampler(seed=self.random_state)
            study = optuna.create_study(sampler=sampler, direction='maximize')
            study.optimize(objective, 
                n_trials=OPTUNA_TRIALS, 
                callbacks=[callback], 
                n_jobs=-1)
            best_model = study.user_attrs["best_model"]
            trial = study.best_trial

            best_model.fit(self.X_train_prep, self.y_train)
            dump(best_model, './data/04_sklearn_models/Models/SVC.joblib')
            
            output_model_log(
                self.y_test, 
                best_model.predict_proba(self.X_test_prep)[:,1], 
                '_SVC_',
                trial.params)
                        
            print('SVC ready...')

        lasso_objective()
        xgb_objective()
        exrt_objective()
        svc_objective()