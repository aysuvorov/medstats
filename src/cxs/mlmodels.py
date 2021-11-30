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
    
def model_quality(real, pred):

    real = real.to_numpy()
    pred = pred.to_numpy()

    # auc and CI
    n_bootstraps = 1000
    rng_seed = 42
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(pred), len(pred))
        if len(np.unique(real[indices])) < 2:
            continue

        score = roc_auc_score(real[indices], pred[indices])
        bootstrapped_scores = bootstrapped_scores + [score]

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    auc_ci_l = sorted_scores[int(0.025 * len(sorted_scores))]
    auc_ci_u = sorted_scores[int(0.975 * len(sorted_scores))]
    auc = roc_auc_score(real, pred)
    
    # brier_score
    brier = brier_score_loss(real, pred)

    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(real, pred).ravel()

    se = tp/(tp + fn)
    se_ci = binomtest(int(se*100), n=100).proportion_ci()
    sp = tn / (fp + tn)
    sp_ci = binomtest(int(sp*100), n=100).proportion_ci()
    #try:
    npv = tn / (fn + tn)
    try:
        npv_ci = binomtest(int(npv*100), n=100).proportion_ci()
    except:
    #    npv = 0.9999999
        npv_ci = [np.nan,np.nan]
    ppv = tp / (tp + fp)
    ppv_ci = binomtest(int(ppv*100), n=100).proportion_ci()
    lr_pos = se/(1-sp)
    lr_neg = (1-se)/sp
    
    ind = ['AUC', 'AUC 95%CI', 'Se', 'Se 95% CI', 'Sp', 'Sp 95% CI',
        'PPV', 'PPV 95% CI', 'NPV', 'NPV 95% CI', 'LR+', 'LR-',
        'Brier']

    return(pd.Series([auc.round(3), 
        str(auc_ci_l.round(3)) + ' - ' + str(auc_ci_u.round(3)),
        se.round(3),
        str(round(se_ci[0], 3)) + ' - ' + str(round(se_ci[1], 3)),
        sp.round(3),
        str(round(sp_ci[0], 3)) + ' - ' + str(round(sp_ci[1], 3)),
        ppv.round(3),
        str(round(ppv_ci[0], 3)) + ' - ' + str(round(ppv_ci[1], 3)),
        #npv.round(3),
        round(npv,3),
        str(round(npv_ci[0], 3)) + ' - ' + str(round(npv_ci[1], 3)),
        lr_pos.round(3),
        lr_neg.round(3),
        brier.round(3)
        ], index=ind))    


def threshold_getter(real, pred):

    g = pd.DataFrame()

    for score in sorted(list(set(pred))):#[1:]:
    
        faf = model_quality(real, (pred >= score).astype(int))    
        g = pd.concat([g, faf], axis=1)

    g.columns = sorted(list(set(pred)))#[1:]

    return(g.T)