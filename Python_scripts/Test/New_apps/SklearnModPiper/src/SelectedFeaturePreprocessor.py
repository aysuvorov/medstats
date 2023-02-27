import json
from joblib import dump

import pandas as pd

import optuna
from sklearn.decomposition import PCA

optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.compose import make_column_selector as selector


import warnings
warnings.filterwarnings("ignore")

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

class Preprocessor:
    def __init__(
        self, 
        splitted_data_load_path, 
        selected_feature_load_path, 
        processed_data_save_path,
        features_selector_model, 
        features_len, 
        pca_use,
        random_state
        ):
        # `features_from must be hyperparam: 
        # `lasso`, `xgb` or ...

        self.random_state = random_state
        self.pca_use = pca_use
        self.X_train = pd.read_pickle(splitted_data_load_path + "/X_train.pkl")
        self.y_train = pd.read_pickle(splitted_data_load_path + "/y_train.pkl")
        self.X_test = pd.read_pickle(splitted_data_load_path + "/X_test.pkl")
        self.y_test = pd.read_pickle(splitted_data_load_path + "/y_test.pkl")

        self.processed_data_save_path = processed_data_save_path

        def json_list_loader(fp):
            f = open(fp)
            content = json.load(f)
            f.close()
            return content

        if features_selector_model == 'lasso':
            self.features = json_list_loader(selected_feature_load_path + "/Lasso_features.json")
        elif features_selector_model == 'xgb':
            self.features = json_list_loader(selected_feature_load_path + "/Xgb_features.json")
        else:
            print('Wrong feature list!')

        self.X_train = self.X_train[self.features[:features_len]]
        self.X_test = self.X_test[self.features[:features_len]]

    
    def __call__(self):

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

        if self.pca_use:
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

        self.X_train_prep.to_pickle(self.processed_data_save_path + "/X_train.pkl")
        self.y_train.to_pickle(self.processed_data_save_path + "/y_train.pkl")
        
        self.X_test_prep.to_pickle(self.processed_data_save_path + "/X_test.pkl")
        self.y_test.to_pickle(self.processed_data_save_path + "/y_test.pkl")

        print('\nFeatures preprocessed and saved...')
