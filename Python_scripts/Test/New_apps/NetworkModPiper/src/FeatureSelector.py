import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegressionCV

from sklearn.model_selection import ShuffleSplit
from xgboost import XGBClassifier

class LassoSelector:
    def __init__(
        self, 
        splitted_data_path, 
        selected_feature_path,
        random_state
        ):

        self.splitted_data_path = splitted_data_path
        self.selected_feature_path = selected_feature_path
        self.random_state = random_state
        self.X_train = pd.read_pickle(self.splitted_data_path + "/X_train.pkl")
        self.y_train = pd.read_pickle(self.splitted_data_path + "/y_train.pkl")

    def __call__(self):

        category_cols = [col for col in self.X_train.columns \
            if self.X_train[col].dtype == 'category']
        numeric_cols = [col for col in self.X_train.columns \
            if col not in category_cols]

        # Creating pipeline for data preprocessing
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

        preprocessor = ColumnTransformer(
                    transformers=[
                        ("num", numeric_transformer, numeric_cols),
                        ("cat", categorical_transformer, category_cols)
                        ],
                        remainder='passthrough'
                        )

        clf = LogisticRegressionCV(Cs=np.logspace(-15, 15, num=25),
            penalty = 'l1',
            solver = 'saga',
            cv = 10, 
            random_state = self.random_state)

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("clf", clf)
            ])

        pipe.fit(self.X_train, self.y_train)

        importance = np.abs(pipe[-1].coef_)[0]
        features = [f.replace("num__", "").replace("cat__", "").replace("_1", "") for f \
            in pipe[:-1].get_feature_names_out()]

        fig = plt.figure(figsize = (12, 8))
        plt.barh(features, importance)
        plt.title("Feature importances via coefficients")
        fig.savefig(self.selected_feature_path + '/Lasso_feature_importance.png', \
            facecolor='white', transparent=False)

        selected = [x[0] for x in sorted({k:v for k, v in \
            zip(features, importance)}.items(), \
                key=lambda x: x[1], reverse=True)][:25]

        with open(self.selected_feature_path + "/Lasso_features.json", "w") as file_path:
            json.dump(selected, file_path)
        print('LassoSelector finished')
        

class XgbSelector(LassoSelector):

    def __call__(self):

        category_cols = [col for col in self.X_train.columns \
            if self.X_train[col].dtype == 'category']
        numeric_cols = [col for col in self.X_train.columns \
            if col not in category_cols]

        # Creating pipeline for data preprocessing
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

        preprocessor = ColumnTransformer(
                    transformers=[
                        ("num", numeric_transformer, numeric_cols),
                        ("cat", categorical_transformer, category_cols)
                        ],
                        remainder='passthrough'
                        )

        clf = XGBClassifier(
                n_estimators = 500,
                random_state = self.random_state,
                tree_method = 'hist',
                n_jobs = -1, 
                enable_categorical = True
                )

        _X_train = pd.DataFrame(preprocessor.fit_transform(self.X_train))

        _X_train.columns = numeric_cols + category_cols
        _X_train.index = self.X_train.index

        _y_train = pd.Series(self.y_train)
        _y_train.index = self.X_train.index

        importance = []

        rs = ShuffleSplit(n_splits=10, train_size=.5, random_state=self.random_state)

        for train_index, _ in rs.split(_X_train):
            _X_train.iloc[train_index, :]
            clf.fit(
                _X_train.iloc[train_index, :],
                _y_train.iloc[train_index],
            )
            importance = importance + [[clf.feature_importances_]]

        features = _X_train.columns
        importance = np.median(np.array(importance).reshape(-1, len(features)), axis = 0)

        fig = plt.figure(figsize = (12, 8))
        plt.barh(features, importance)
        plt.title("Feature importances via coefficients")
        fig.savefig(self.selected_feature_path + '/Xgb_feature_importance.png', \
            facecolor='white', transparent=False)

        selected = [x[0] for x in sorted({k:v for k, v in \
            zip(features, importance)}.items(), \
                key=lambda x: x[1], reverse=True)][:25]

        with open(self.selected_feature_path + "/Xgb_features.json", "w") as file_path:
            json.dump(selected, file_path)
        print('XgbSelector finished')

