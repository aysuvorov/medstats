

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
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri, FloatVector, IntVector, FactorVector, Formula
from sklearn.metrics import roc_auc_score, roc_curve


stats = importr('stats')
base = importr('base')
proc = importr('pROC')


"""
AUC, sensetivity, specificity
"""

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

def auc_plotter(real, pred, save, title='', filename=""):
    auc_r = roc_auc_score(real, pred)

    if auc_r > 0.5:
    
        lr_fpr, lr_tpr, _ = roc_curve(real, pred)

    else:
        lr_fpr, lr_tpr, _ = roc_curve(real.replace([0,1],[1,0]), pred)
        auc_r = roc_auc_score(real.replace([0,1],[1,0]), pred)

    sns.set(style='white')
    fig, ax = plt.subplots(figsize=(6,6))
    #plt.plot(lr_fpr, lr_tpr, marker='.', label = 'AUC ' + str(round(auc_r, 3)))
    plt.plot(lr_fpr, lr_tpr, marker='.')
    plt.plot(lr_fpr, lr_fpr, linestyle='--')
    plt.title(title)
    plt.legend(loc = 4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if save:
        plt.savefig(filename + '.png')

    plt.show()


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