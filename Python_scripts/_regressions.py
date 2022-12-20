

"""
Regressions:

logistic, COX, univariate, multivariate, stepwise selection

AUC, sensetivity, specificity
"""

import numpy as np
import pandas as pd
import statsmodels.api as sma
import matplotlib.pyplot as plt
import seaborn as sns
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri as rpyn
rpyn.activate()

from lifelines import CoxPHFitter
from scipy import interpolate
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri, FloatVector, IntVector, FactorVector, Formula
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, confusion_matrix
from scipy.stats import binomtest
from scipy.stats import binomtest

stats = importr('stats')
base = importr('base')

#+-------------------------------------------------------------------
"""
Regressions
"""

# Univariate logistic with/without adjustments

def onedim_logregr(df, group, adj = False, adj_cols_lst = None):

    logregr = pd.DataFrame()

    columns = [x for x in df.columns if x != group]

    if adj:

        for col in columns:

            tb = df[[col, group] + adj_cols_lst].dropna()

            try:
                logit_model=sma.GLM(
                    tb[group].astype(float), 
                    sma.add_constant(tb.drop(group, axis=1).astype(float)), 
                    family = sma.families.Binomial()
                    )
                result=logit_model.fit()
                params = np.exp(result.params)[1]
                conf0 = np.exp(result.conf_int())[0][1]
                conf1 = np.exp(result.conf_int())[1][1]
                p = result.pvalues[1]

            except:

                params = 'NA'
                conf0 = 'NA'
                conf1 = 'NA'
                p = 1

            logregr = logregr.append({'Names': col, 'OR': params, 'lower': conf0, 'upper': conf1,'p_val': round(p, 3)}, ignore_index=True).reindex(columns=['Names', 'OR', 'lower', 'upper', 'p_val'])

    else:

        for col in columns:

            tb = df[[col, group]].dropna()

            try:
                logit_model=sma.GLM(
                    tb[group].astype(float), 
                    sma.add_constant(tb[col].astype(float)), 
                    family = sma.families.Binomial()
                    )
                result=logit_model.fit()
                params = np.exp(result.params)[1]
                conf0 = np.exp(result.conf_int())[0][1]
                conf1 = np.exp(result.conf_int())[1][1]
                p = result.pvalues[1]

            except:

                params = 'NA'
                conf0 = 'NA'
                conf1 = 'NA'
                p = 1

            logregr = logregr.append({'Names': col, 'OR': params, 'lower': conf0, 'upper': conf1,'p_val': round(p, 3)}, ignore_index=True).reindex(columns=['Names', 'OR', 'lower', 'upper', 'p_val'])

    return(logregr)

# Univariate COX with/without adjustments

def onedim_coxregr(df, group, time, adj = False, adj_cols_lst = None):

    cph = CoxPHFitter()

    columns = [x for x in df.columns if (x != group) and (x != time) ]

    coxregr = pd.DataFrame()

    if adj:
        for col in columns:
            try:
                model = cph.fit(df[[col, group, time] + adj_cols_lst].dropna(), duration_col=time, event_col=group)
                HR = round(model.hazard_ratios_[0], 2)
                p = round(model.summary.iloc[:,8][0], 3)
                conf0 = round(model.summary.iloc[:,5][0], 2)
                conf1 = round(model.summary.iloc[:,6][0], 2)

            except:
                HR = 'NA'
                p = 1
                conf0 = 'NA'
                conf1 = 'NA'

            coxregr = coxregr.append({'Фактор': df[col].name, 'HR': HR, 'Нижний 95% ДИ': conf0, 'Верхний 95% ДИ': conf1,'p_val': p}, ignore_index=True)

        coxregr = coxregr.reindex(columns=['Фактор', 'HR', 'Нижний 95% ДИ', 'Верхний 95% ДИ', 'p_val'])
    else:
        for col in columns:
            try:
                model = cph.fit(df[[col, group, time]].dropna(), duration_col=time, event_col=group)
                HR = round(model.hazard_ratios_[0], 2)
                p = round(model.summary.iloc[:,8][0], 3)
                conf0 = round(model.summary.iloc[:,5][0], 2)
                conf1 = round(model.summary.iloc[:,6][0], 2)

            except:
                HR = 'NA'
                p = 1
                conf0 = 'NA'
                conf1 = 'NA'

            coxregr = coxregr.append({'Фактор': df[col].name, 'HR': HR, 'Нижний 95% ДИ': conf0, 'Верхний 95% ДИ': conf1,'p_val': p}, ignore_index=True)

        coxregr = coxregr.reindex(columns=['Фактор', 'HR', 'Нижний 95% ДИ', 'Верхний 95% ДИ', 'p_val'])

    return(coxregr)

## Stepwise - logistic regression from MASS package

def step_mass_R(df, point, k=3.58):

    tb = df.copy()

    for col in [x for x in tb.columns if pd.CategoricalDtype.is_dtype(tb[x]) == True]:
        tb[col] = tb[col].astype(str)
        tb[col] = tb[col].astype('category')

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(tb)

    del tb

    f = stats.glm(point + ' ~ .', family = 'binomial', data = r_df)
    m = stats.step(f, k = k, trace=False) 
    mod_lst = list(base.all_vars(stats.formula(m)))[1:]
    
    return(mod_lst)


def cox_multi(df, mod_lst, duration, point, style='ascii', check_proportional=True):

    cph = CoxPHFitter()

    model = cph.fit(df[mod_lst + [duration, point]] , duration_col=duration, event_col=point)
    b = model.print_summary(style=style, columns=['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p'], decimals=3)

    if check_proportional:
        c = cph.check_assumptions(df[mod_lst + [duration, point]], p_value_threshold=0.05, show_plots=True)
    else:
        c = None
    
    return([b,c])

def odds_multi(df, mod_lst, point):

    model = sma.Logit(df[point], sma.add_constant(df[mod_lst])).fit()

    b = model.summary()

    params = round(np.exp(model.params)[1:],2)
    names = params.index
    conf0 = round(np.exp(model.conf_int())[1:][0],2)
    conf1 = round(np.exp(model.conf_int())[1:][1],2)
    p = round(model.pvalues[1:],3)
    multivar = pd.DataFrame({'Names': names, 'OR': params, 'lower': conf0, 'upper': conf1,'p_val': p})
    c = multivar.reset_index().iloc[:, 1:]
    
    return([b,c])
