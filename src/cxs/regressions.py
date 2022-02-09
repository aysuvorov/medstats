

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

stats = importr('stats')
base = importr('base')
proc = importr('pROC')


"""
AUC, sensetivity, specificity
"""

def auc_plotter_numeric(real, pred, title=None):

    thresholds = np.sort(pred, axis=None)

    ROC = np.zeros((len(real) + 2, 2))
    ROC = np.append(ROC, [[1,1]],axis = 0)[::-1]

    for i in range(len(real)):
        t = thresholds[i]

        # Classifier / label agree and disagreements for current threshold.
        TP_t = np.logical_and( pred > t, real==1 ).sum()
        TN_t = np.logical_and( pred <=t, real==0 ).sum()
        FP_t = np.logical_and( pred > t, real==0 ).sum()
        FN_t = np.logical_and( pred <=t, real==1 ).sum()

        # Compute false positive rate for current threshold.
        FPR_t = FP_t / float(FP_t + TN_t)
        ROC[i+1,0] = FPR_t

        # Compute true  positive rate for current threshold.
        TPR_t = TP_t / float(TP_t + FN_t)
        ROC[i+1,1] = TPR_t

        # Plot the ROC curve.
    fig = plt.figure(figsize=(6,6))
    plt.plot(ROC[:,0], ROC[:,1], lw=2)
    plt.plot([[0,0], [1,1]], linestyle='--', c = 'gray')
    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,1.1)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.grid()

    AUC = 0.
    for i in range(len(real)):

        AUC += (ROC[i+1,0]-ROC[i,0]) * (ROC[i+1,1]+ROC[i,1]) * -1
    AUC *= 0.5

    if title:
        plt.title(title)
    else:
        plt.title('ROC curve, AUC = %.3f'%AUC)
    plt.show()

#+-------------------------------------------------------------------
class ModPerformance(object):

    def __init__(
        self, 
        real, 
        pred
        )-> None:

        self.real = np.asarray(real)
        self.pred = np.asarray(pred)

        thresholds = np.sort(self.pred, axis=None)

        ROC = np.zeros((len(self.real) + 2, 2))
        ROC = np.append(ROC, [[1,1]],axis = 0)[::-1]

        for i in range(len(self.real)):
            t = thresholds[i]

            # Classifier / label agree and disagreements for current threshold.
            TP_t = np.logical_and( self.pred > t, self.real==1 ).sum()
            TN_t = np.logical_and( self.pred <=t, self.real==0 ).sum()
            FP_t = np.logical_and( self.pred > t, self.real==0 ).sum()
            FN_t = np.logical_and( self.pred <=t, self.real==1 ).sum()

            # Compute false positive rate for current threshold.
            FPR_t = FP_t / float(FP_t + TN_t)
            ROC[i+1,0] = FPR_t

            # Compute true  positive rate for current threshold.
            TPR_t = TP_t / float(TP_t + FN_t)
            ROC[i+1,1] = TPR_t

        AUC = 0.
        for i in range(len(self.real)):

            AUC += (ROC[i+1,0]-ROC[i,0]) * (ROC[i+1,1]+ROC[i,1]) * -1
        AUC *= 0.5

        if AUC < 0.5:
            self.real = (self.real - 1)*-1    
        self.roc = ROC


    def auc(self):  
        result = proc.ci(FloatVector(self.real), FloatVector(self.pred))
        return(result[1], result[0], result[2])


    def thresholds_(real, pred):

        brier = brier_score_loss(real, pred)

        tn, fp, fn, tp = confusion_matrix(real, pred).ravel()

        se = tp/(tp + fn)
        se_ci = binomtest(int(se*100), n=100).proportion_ci()
        sp = tn / (fp + tn)
        sp_ci = binomtest(int(sp*100), n=100).proportion_ci()
        npv = tn / (fn + tn)
        try:
            npv_ci = binomtest(int(npv*100), n=100).proportion_ci()
        except:
            npv_ci = [np.nan,np.nan]
        ppv = tp / (tp + fp)
        ppv_ci = binomtest(int(ppv*100), n=100).proportion_ci()
        lr_pos = se/(1-sp)
        lr_neg = (1-se)/sp
        
        ind = ['Se', 'Se 95% CI', 'Sp', 'Sp 95% CI',
            'PPV', 'PPV 95% CI', 'NPV', 'NPV 95% CI', 'LR+', 'LR-',
            'Brier']

        return(pd.Series([se.round(3),
            str(round(se_ci[0], 3)) + ' - ' + str(round(se_ci[1], 3)),
            sp.round(3),
            str(round(sp_ci[0], 3)) + ' - ' + str(round(sp_ci[1], 3)),
            ppv.round(3),
            str(round(ppv_ci[0], 3)) + ' - ' + str(round(ppv_ci[1], 3)),
            round(npv,3),
            str(round(npv_ci[0], 3)) + ' - ' + str(round(npv_ci[1], 3)),
            lr_pos.round(3),
            lr_neg.round(3),
            brier.round(3)
            ], index=ind)) 


    def threshold_getter(self):

        g = pd.DataFrame()
        for score in sorted(list(set(self.pred))):
            faf = ModPerformance.thresholds_(self.real, np.array(self.pred >= score))    
            g = pd.concat([g, faf], axis=1)
        g.columns = sorted(list(set(self.pred)))
        g = g.T.reset_index()
        g.rename(columns={'index':'Thres'}, inplace=True)
        self.thres = g
        return(self.thres)

    def plot_roc(self, title=None):
        auc_plotter_numeric(self.real, self.pred, title=title)

    def intrplt(self, se=None, sp=None):

        g = ModPerformance.threshold_getter(self)

        if se:
            f = interpolate.interp1d(g['Se'], g['Sp'])
            print('Sp : ')
            return([f(sp), binomtest(int(f(sp)*100), n=100).proportion_ci()])
        elif sp:
            f = interpolate.interp1d(g['Sp'], g['Se'])
            print('Se : ')
            return([f(se), binomtest(int(f(se)*100), n=100).proportion_ci()])


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