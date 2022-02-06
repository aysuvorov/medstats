import numpy as np
import pandas as pd
import scipy

import glmnet_python
from glmnet import glmnet
from cvglmnet import cvglmnet
from cvglmnetCoef import cvglmnetCoef

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
from scipy.stats import binomtest


############################

def lasso_searcher(
    X_train, y_train,
    sc_lst,
    random_state, 
    c_v_splits,
    alpha = 1,
    scale=True
    ):

    sc = StandardScaler()

    g = np.random.RandomState(random_state)
    foldid = g.choice(a=range(c_v_splits), size=X_train.shape[0], replace=True)

    X_train_sc = X_train.copy()

    if scale:
        for col in sc_lst:
            X_train_sc[[col]] = sc.fit_transform(X_train_sc[[col]])

    X_train_sc = scipy.array(X_train_sc, dtype = scipy.float64)
    Y_train_sc = scipy.array(y_train, dtype = scipy.float64)

    fit = cvglmnet(x=X_train_sc, 
                            y=Y_train_sc, 
                            family = 'binomial', 
                            foldid=foldid,  
                            alpha = alpha, 
                            standardize=False)

    coeffs = cvglmnetCoef(fit, s = 'lambda_min').ravel()

    A = pd.DataFrame(coeffs[1:], index=sc_lst)
    A['abs'] = abs(A.iloc[:,0])
    A.columns = ['Coefs', 'abs']
    return(A.sort_values('abs',ascending=False))


def rf_searcher(
    X_train, y_train,
    sc_lst,
    random_state, 
    c_v_splits,
    n_estimators = 100,
    scale=True
    ):

    sc = StandardScaler()

    X_train_sc = X_train.copy()

    if scale:
        for col in sc_lst:
            X_train_sc[[col]] = sc.fit_transform(X_train_sc[[col]])

    grid_param = {
    'n_estimators': [n_estimators],
    'criterion': ['entropy']
    }

    clf = RandomForestClassifier()
    cv = ShuffleSplit(n_splits=c_v_splits, 
        test_size=0.3, 
        random_state=random_state)

    model = GridSearchCV(estimator=clf,
                     param_grid=grid_param,
                     scoring='roc_auc',
                     cv=cv,
                     n_jobs=-1)

    model.fit(X_train_sc, y_train)
    coeffs = model.best_estimator_.feature_importances_

    A = pd.DataFrame(coeffs, index=X_train.columns)
    A.columns = ['Coefs']
    return(A.sort_values('Coefs',ascending=False))#[:features_number]


def logictic_iter(
        train_X, test_X, 
        train_y, test_y,
        random_state = 0,
        c_v_splits = 5

    ):

    clf = LogisticRegression()

    cv = ShuffleSplit(n_splits=c_v_splits, 
            test_size=0.3, 
            random_state=random_state)

    grid_param = {
        'random_state': [random_state]
        }

    model = GridSearchCV(estimator=clf,
                        param_grid=grid_param,
                        scoring='roc_auc',
                        cv=cv,
                        n_jobs=-1)

    model.fit(train_X, train_y)
    #mod = model.best_estimator_
    score = roc_auc_score(test_y, model.best_estimator_.predict_proba(test_X)[:, 1])
    #score = roc_auc_score(test_y, model.best_estimator_.predict(test_X))
    #return(mod, score)
    return(score)


def rf_iter(
        train_X, test_X, 
        train_y, test_y,
        n_estimators = 1,
        random_state = 0,
        c_v_splits = 5

    ):

    clf = RandomForestClassifier()

    cv = ShuffleSplit(n_splits=c_v_splits, 
            test_size=0.3, 
            random_state=random_state)

    grid_param = {
    'n_estimators': [n_estimators],
    'criterion': ['entropy']
    }

    model = GridSearchCV(estimator=clf,
                        param_grid=grid_param,
                        scoring='roc_auc',
                        cv=cv,
                        n_jobs=-1)

    model.fit(train_X, train_y)
    #mod = model.best_estimator_
    score = roc_auc_score(test_y, model.best_estimator_.predict_proba(test_X)[:, 1])
    #return(mod, score)
    return(score)


class Model(object):

    def __init__(
        self,
        method,
        name,
        auc_mean,
        auc_std
        ):

        self.method = method
        self.name = name
        self.auc_mean = auc_mean
        self.auc_std = auc_std

#################################################################
    
