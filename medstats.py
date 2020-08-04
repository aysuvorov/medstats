"""
Module by Alexander Suvorov 

First published 31-07-2020

E-mail / Skype: yourmedstat@gmail.com

"""

import numpy as np
import pandas as pd
import scipy.stats as st
import math as m
import statsmodels as sm
import statsmodels.api as sma
import gspread

from sklearn.utils import resample
from sklearn.metrics import roc_curve, auc
from dask import delayed
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, fisher_exact, chi2_contingency, kruskal
from oauth2client.client import GoogleCredentials
from lifelines import CoxPHFitter

"""
Fills NaN using func - mean, median, interpolate

(Simple imputer)
"""
def filler(df, func):
	for col in df:
		df[col].apply(pd.to_numeric, errors='coerce')

	df = df.fillna(df.func())
	return df

"""
Importer from google docs using table key and sheet number (0,1,2...)

"""
def import_gsheet(key, sheet = 0):
	gc = gspread.authorize(GoogleCredentials.get_application_default())

	spreadsheet = gc.open_by_key(key) # (the string in the url following spreadsheet/d/)
	sheet =  spreadsheet.get_worksheet(sheet)  # 0 - the first sheet in the file
	df = pd.DataFrame(sheet.get_all_records())
	return df


"""
Sample size for proportions

k = test / placebo
"""

def prop_size(p1, p2, alpha = 0.05, beta = 0.8, k = 1):
    za = st.norm.ppf(1 - alpha/2)
    zb = st.norm.ppf(beta)
    n = m.ceil((((za + zb)**2) * ( p1 * (1-p1) + (p2 * (1-p2)))) / ((p1 - p2)**2))
    n1 = m.ceil((0.5 * n * (1 + k)))
    n2 = m.ceil((0.5 * n * (1 + (1/k))))
    group = n1 + n2
    return print("В настоящем исследовании в группе `Препарат` требуется "  + str(n1) + " пациентов, группе `Плацебо` - "  + str(n2) + ", во всей группе - " + str(group))

"""
Sample size for means
"""

def mean_size(m1, m2, sd1, sd2, alpha = 0.05, beta = 0.8, k = 1):
    za = st.norm.ppf(1 - alpha/2)
    zb = st.norm.ppf(beta)
    pooled = np.sqrt((sd1**2 + sd2**2)/2)
    n = m.ceil((((za + zb)**2) * (2 * (pooled**2))) / ((m1 - m2)**2))
    n1 = m.ceil((0.5 * n * (1 + k)))
    n2 = m.ceil((0.5 * n * (1 + (1/k))))
    group = n1 + n2
    return print("В настоящем исследовании в группе `Препарат` требуется "  + str(n1) + " пациентов, группе `Плацебо` - "  + str(n2) + ", во всей группе - " + str(group))


"""
CDF of 1-d array
"""
def cdf(array):
    n = len(array)
    x = np.sort(array)
    y = np.arange(1, n+1) / n
    return x, y

"""
Bootstrap comparison

Compare arrays processed by some stats functions (means, medians, centiles, R2, rho, slopes?)

Returns median of p-value with 2.5, 97.5% centiles
"""

def bs_multi(a,b, func, R = 100):
    t = func(a) - func(b)
    B = np.empty(shape = R)
    p = np.empty(shape = R)
    ab_concat = np.concatenate((a, b))
    mean_ab_concat = func(ab_concat)

    a_shifted = a - func(a) + mean_ab_concat
    b_shifted = b - func(b) + mean_ab_concat

    for val in range(R):
        for i in range(R):
            a_bs = np.random.choice(a_shifted, len(a))
            b_bs = np.random.choice(b_shifted, len(b))
            permuted_data = np.random.permutation(np.concatenate((a_bs, b_bs))) 
            perm_sample_1 = permuted_data[:len(a)]
            perm_sample_2 = permuted_data[len(a):]
            B[i] = func(perm_sample_1) - func(perm_sample_2)
        p[val] = (np.sum(abs(B) >= abs(t)) / (R+1))
    return(np.percentile(p, [2.5, 50, 97.5]))

# SAMPLE:
# funcs = [np.mean, np.median, np.std, st.iqr]
#
# for f in funcs:
#    print(bs_multi(a=a, b=b, func = f))
    
"""
Bootstrap comparison

Percentile comparison. I.e. compare arrays a,b by 25%. 

Returns median of p-value with 2.5, 97.5% centiles
"""

def bs_perc(a,b, perc, R = 100):
    t = np.percentile(a, [perc]) - np.percentile(b, [perc])
    B = np.empty(shape = R)
    p = np.empty(shape = R)
    ab_concat = np.concatenate((a, b))
    mean_ab_concat = np.percentile(ab_concat, [perc])

    a_shifted = a - np.percentile(a, [perc]) + mean_ab_concat
    b_shifted = b - np.percentile(b, [perc]) + mean_ab_concat

    for val in range(R):
        for i in range(R):
            a_bs = np.random.choice(a_shifted, len(a))
            b_bs = np.random.choice(b_shifted, len(b))
            permuted_data = np.random.permutation(np.concatenate((a_bs, b_bs))) 
            perm_sample_1 = permuted_data[:len(a)]
            perm_sample_2 = permuted_data[len(a):]
            B[i] = np.percentile(perm_sample_1, [perc]) - np.percentile(perm_sample_2, [perc])
        p[val] = (np.sum(abs(B) >= abs(t)) / (R+1))

    return(np.percentile(p, [25, 50, 75]))
    
"""
Bootstrap comparison:

Proportions comparison. 

Returns median of p-value with 2.5, 97.5% centiles

inv - share in treatment group, aka 0.76
inv_n - number of patients in treatment group

plac - share in control group, aka 0.55
plac_n - number of patients in control group
"""    

def bs_props(inv, inv_n, plac, plac_n, R=100):
    diff = inv - plac
    ni = (np.array(inv*inv_n)).astype(int)
    t0 = np.zeros(inv_n - ni)
    t1 = np.ones(ni)

    npc = (np.array(plac*plac_n)).astype(int)
    p0 = np.zeros(plac_n - npc)
    p1 = np.ones(npc)

    tt = np.concatenate((t0, t1))
    pp = np.concatenate((p0, p1))

    p = np.empty(shape = R)
    B = np.empty(shape = R)

    for val in range(R):
        for iter in range(R):
            t_bs = resample(tt, n_samples=inv_n, replace=True)
            p_bs = resample(pp, n_samples=plac_n, replace=True)
            permuted_data = np.random.permutation(np.concatenate((t_bs, p_bs))) 
            perm_sample_1 = permuted_data[:inv_n]
            perm_sample_2 = permuted_data[inv_n:]
            B[iter] = np.sum(perm_sample_1)/len(perm_sample_1) - np.sum(perm_sample_2) / len(perm_sample_2)
        p[val] = np.sum(abs(B) >= abs(diff)) / (R+1)

    return(np.percentile(p, [2.5, 50, 97.5]))    

    
"""
Creating dummy variables

cat_vars - list of variables in original df to proceed
"""

def dummification(df, cat_vars):
    for var in cat_vars:
        cat_list='var'+'_'+var
        cat_list = pd.get_dummies(df[var], prefix=var)
        data1=df.join(cat_list)
        df=data1

    data_vars=df.columns.values.tolist()
    to_keep=[i for i in data_vars if i not in cat_vars]

    df=df[to_keep]
    return df

"""
Describe variables function - in russian

Categorial variables must be dummified!!!

Returns variable type, N of patients, median, centiles or share%

@delayed requires .compute() method

"""

def summary(df):
    summarize = pd.DataFrame()
    for col in df:
        J = len(list(df.columns))
        for j in range(J):
            if len(np.unique(df[col])) < 3:
                v = df[col].name
                var = 'Категориальная'
                n = np.sum(df[col])
                percents = round(n / len(df[col])*100, 1)
                med = centiles0 = centiles1 = avg = sd = minn = maxx = sh = '-'
            else:
                v = df[col].name
                var = 'Числовая'
                n = len(df[col])
                percents = '-'
                flist = (np.mean, np.std, np.min, np.max, np.median)
                avg, sd, minn, maxx, med = [f(df[col]) for f in flist]
                centiles0 = round(np.percentile(df[col], 25), 2)
                centiles1 = round(np.percentile(df[col], 75), 2)
                sh = round(shapiro(df[col])[1], 3)
        summarize = summarize.append({'Фактор': v, 'Тип': var, 'Количество': n, 'Доля, %': percents,'Медиана': med, '25%': centiles0, '75%': centiles1, 'Среднее': avg , 'Ст.отклон':  sd, \
                                      'Мин': minn, 'Макс': maxx, 'Критерий Шапиро-Уилка, р': sh}, ignore_index=True)
        summarize = summarize.reindex(columns=['Фактор', 'Тип', 'Количество', 'Доля, %', 'Мин','25%', 'Медиана', '75%', 'Макс', 'Среднее','Ст.отклон', 'Критерий Шапиро-Уилка, р'])
    return summarize

"""
Compare variables by 2 groups

Vars must be dummified!!!

By default numerics compared using Mann-Whitney, shares - with Fisher exact test

save_tab enables xlsx export

@delayed requires .compute() method
"""

def compare(df, group, gr_id_1 = 0, gr_id_2 = 1, name_1 = 'Группа 0', name_2 = 'Группа 1', save_tab = False):

    x = df.loc[df[group] == gr_id_1]
    y = df.loc[df[group] == gr_id_2]

    comparison = pd.DataFrame()

    for col in df:
        J = len(list(df.columns))
        for j in range(J):
            if len(np.unique(df[col])) < 3:
                v = df[col].name 
                gr1obs = np.sum(x[col])
                gr2obs = np.sum(y[col])
                gr1exp = len(x[col]) - gr1obs
                gr2exp = len(y[col]) - gr2obs
                obs = np.array([[gr1obs, gr2obs], [gr1exp, gr2exp]])
                # в таблицу идут:
                p_val = round(fisher_exact(obs)[1], 3)
                p1 = str(round(gr1obs/len(x[col])*100,1)) + " %" + " (" + str(gr1obs) +")"
                p2 = str(round(gr2obs/len(y[col])*100,1)) + " %" + " (" + str(gr2obs) +")"
            else:
                v = df[col].name
                gr1obs = round(np.median(x[col]),2)
                gr2obs = round(np.median(y[col]),2)
                cent1 = np.percentile(x[col], [25,75])
                cent2 = np.percentile(y[col], [25,75])
                # в таблицу идут:
                p_val = round(mannwhitneyu(x[col], y[col])[1], 3)
                p1 = str(gr1obs) + " (" + str(round(cent1[0],2)) + "; " + str(round(cent1[1],2)) + ")"
                p2 = str(gr2obs) + " (" + str(round(cent2[0],2)) + "; " + str(round(cent2[1],2)) + ")"
        comparison = comparison.append({'Фактор': v, name_1: p1, name_2: p2,'p_val': p_val}, ignore_index=True)
        comparison = comparison.reindex(columns=['Фактор', name_1, name_2, 'p_val'])

    if save_tab == True:
        return pd.DataFrame.to_excel(comparison, 'Сравнение по группам.xlsx')
    else:
        return comparison
        
"""
One-dimensional regression analysis

Vars must be dummified!!!

pmin - significance level

"""

def regr_onedim(df, group, signif_only = False, pmin = 0.05, save_tab = False):

    reg_data = df.drop(columns=group)
    y = df[[group]]

    logregr = pd.DataFrame()

    g = reg_data.columns[1:-1]

    for col in g:
        J = len(g)
        for j in range(J):
            v = reg_data[col].name
            X= sma.add_constant(reg_data[col])
            logit_model=sma.GLM(y,X, family = sma.families.Binomial())
            result=logit_model.fit()
            params = round(np.exp(result.params)[1], 1)
            conf0 = round(np.exp(result.conf_int())[0][1],2)
            conf1 = np.exp(result.conf_int())[1][1]
            p = round(result.pvalues[1], 3)
        logregr = logregr.append({'Names': v, 'OR': params, 'lower': conf0, 'upper': conf1,'p_val': p}, ignore_index=True)
        logregr = logregr.reindex(columns=['Names', 'OR', 'lower', 'upper', 'p_val'])    

    if signif_only == True:
        logregr = logregr[logregr['p_val'] < pmin]
    else:
        pass

    if save_tab == True:
        return pd.DataFrame.to_excel(logregr, 'Одномерный регрессионный анализ.xlsx')
    else:
        return logregr

"""
One-dimensional regression analysis with sex and age adjustment

You shoild provide sex and age columns

UNDER DEVELOPMENT!

"""
def regr_onedim_adj(df, group, age_col, sex_col, signif_only = False, pmin = 0.05, save_tab = False):

    reg_data = df.drop(columns=group)
    y = df[[group]]

    logregr = pd.DataFrame()

    g = reg_data.columns #[1:-1]

    for col in g:
        J = len(g)
        for j in range(J):
            v = reg_data[col].name
            logit_model=sma.GLM(y,sma.add_constant(reg_data[[col, age_col, sex_col]]), family = sma.families.Binomial())
            result=logit_model.fit()
            params = round(np.exp(result.params)[1], 1)
            conf0 = round(np.exp(result.conf_int())[0][1],2)
            conf1 = np.exp(result.conf_int())[1][1]
            p = round(result.pvalues[1], 3)
        logregr = logregr.append({'Names': v, 'OR': '{0:.2f}'.format(params), 'lower': '{0:.2f}'.format(conf0), 'upper': '{0:.2f}'.format(conf1),'p_val': p}, ignore_index=True)
        logregr = logregr.reindex(columns=['Names', 'OR', 'lower', 'upper', 'p_val'])    

    if signif_only == True:
        logregr = logregr[logregr['p_val'] < pmin]
    else:
        pass

    if save_tab == True:
        return pd.DataFrame.to_excel(logregr, 'Одномерный регрессионный анализ.xlsx')
    else:
        return logregr

"""
Multivariate regression for logistic regression

"""
def regr_multi(df, group, lst, save_tab = False):
    logit_model=sma.GLM(df[group],sma.add_constant(df[lst]), family = sma.families.Binomial())
    result=logit_model.fit()
    params = round(np.exp(result.params)[1:],2)
    names = params.index
    conf0 = round(np.exp(result.conf_int())[1:][0],2)
    conf1 = round(np.exp(result.conf_int())[1:][1],2)
    p = round(result.pvalues[1:],3)
    multivar = pd.DataFrame({'Names': names, 'OR': params, 'lower': conf0, 'upper': conf1,'p_val': p})
    multivar.reset_index().iloc[:, 1:]

    if save_tab == True:
        return pd.DataFrame.to_excel(multivar, 'Многомерный регрессионный анализ.xlsx')
    else:
        return multivar

"""
ROC threshold cut offs calculations

df - core data frame
vars - list of vars of interest 
group - target var, as 'GROUP'
time - time var as 'TIME'
group - in format 'GROUP', must be dummified
family - 'logistic' or 'cox'


EXAMPLE:
roc_cut(df, ['var1', 'var2', 'var3'], ['group_var'])

"""
@delayed
def roc_job(pred, predictor, pos):
    d = pd.DataFrame()
    d['prob'] = pred
    d['cut'] = predictor
    return d[d['prob'] == pos].iloc[0,1]

def roc_cut(df, vars, group, time = 0, family = 'logistic', save_tab = False):
    roc_cut = pd.DataFrame()

    if family == 'logistic':
        table = df[vars]
        y = df[group]
        for col in table:
            J = len(list(table.columns))
            for j in range(J):
                v = table[col].name
                pred = sma.GLM(y, sma.add_constant(table[col]), family = sma.families.Binomial()).fit().predict(sma.add_constant(table[col]))
                fpr, tpr, thresholds =roc_curve(y, pred)
                r = round(auc(fpr, tpr)*100, 1) #AUC
                i = np.arange(len(tpr))

                roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
                roc = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
                thres = roc.iloc[0,4]
                sens = round(roc.iloc[0,1]*100, 1) #sensetivity
                spec = round(roc.iloc[0,2]*100, 1) #specifisity
                cut = roc_job(pred, predictor = table[col], pos = thres).compute()
            roc_cut = roc_cut.append({'Фактор': v, 'AUC, %': r, 'Порог': cut,'Чувствительность, %': sens, 'Специфичность, %':spec}, ignore_index=True)

    elif family == 'cox':
        cph = CoxPHFitter()
        table = df[vars]  
        table[time] = df[time]
        table[group] = df[group]
        for col in table.columns[:-2]:
            v = table[col].name
            cph.fit(table[[col, group, time]], duration_col=time, event_col=group)
            pred = cph.predict_survival_function(table[[col]], np.percentile(df[[time]],0.99)).T

            fpr, tpr, thresholds =roc_curve(table[group], pred)
            r = round(auc(fpr, tpr)*100, 1) #AUC
            i = np.arange(len(tpr))

            roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
            roc = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
            thres = roc.iloc[0,4]
            sens = round(roc.iloc[0,1]*100, 1) #sensetivity
            spec = round(roc.iloc[0,2]*100, 1) #specifisity
            cut = roc_job(pred[0], predictor = table[col], pos = thres).compute()
            roc_cut = roc_cut.append({'Фактор': v, 'AUC, %': r, 'Порог': cut,'Чувствительность, %': sens, 'Специфичность, %':spec}, ignore_index=True)     

    else:
        print('Error')

    roc_cut = roc_cut.reindex(columns=['Фактор', 'AUC, %', 'Порог','Чувствительность, %', 'Специфичность, %'])

    if save_tab == True:
        return pd.DataFrame.to_excel(roc_cut, 'Пороги по ROC-анализу.xlsx')
    else:
        return roc_cut

    return roc_cut

"""
One dimensional Cox regression analysis

EXAPLE:
cox_onedim(df, 'DEATH', 'TIME')

"""

def cox_onedim(df, group, time, save_tab = False):

    reg_data = df.drop(columns=[group, time])
    cph = CoxPHFitter()

    coxregr = pd.DataFrame()

    for col in reg_data:
        v = reg_data[col].name
        model = cph.fit(df[[col, group, time]], duration_col=time, event_col=group)
        HR = round(model.hazard_ratios_[0], 2)
        p = round(model.summary.iloc[:,8][0], 3)
        conf0 = round(model.summary.iloc[:,5][0], 2)
        conf1 = round(model.summary.iloc[:,6][0], 2)
        coxregr = coxregr.append({'Фактор': v, 'HR': HR, 'Нижний 95% ДИ': conf0, 'Верхний 95% ДИ': conf1,'p_val': p}, ignore_index=True)
    
    coxregr = coxregr.reindex(columns=['Фактор', 'HR', 'Нижний 95% ДИ', 'Верхний 95% ДИ', 'p_val'])
	
    if save_tab == True:
        return pd.DataFrame.to_excel(coxregr, 'Регрессия Кокса, одномерный анализ.xlsx')
    else:
        return coxregr

    return coxregr

"""
Backwise selection model

lst - list of significant vars. 
pmin - significance level
steps - minimal steps for selection
group - in format 'GROUP', must be dummified
family - 'logistic' or 'cox'

"""

def backwise(df, lst, group, time = 0, family = 'logistic', steps = 100, pmin = 0.05):
    if family == 'logistic':
        lst = lst
        X = df[lst]
        y = df[group]
        n = len(lst)
        pmax = 1
        steps = steps

        for i in range(steps):
            if pmax > pmin:
                X = df[lst]
                model = sma.GLM(y, sma.add_constant(X), family = sma.families.Binomial()).fit()
                pvalues = model.pvalues.iloc[1:].sort_values()
                pmax = pvalues[-1]
                out = pvalues.index[-1]
                lst.remove(out)
            else:
                break

        S = sma.add_constant(df[list(pvalues.index)])

        result=sma.GLM(y, S, family = sma.families.Binomial()).fit()
        v = S.columns
        OR = round(np.exp(result.params),1)
        upper = round(np.exp(result.conf_int()[0]),1)
        lower = round(np.exp(result.conf_int()[1]),1)
        p_val = round(result.pvalues,3)

        stepwise = (pd.DataFrame({'Фактор': v, 'ОШ': OR, 'Верхний 95% ДИ' : upper, 'Нижний 95% ДИ':lower, 'Значимость, p':p_val})).reindex(columns=['Фактор', 'ОШ', 'Верхний 95% ДИ', 'Нижний 95% ДИ', 'Значимость, p'])
        return stepwise

    elif family =='cox':
        X = df[lst]
        X[[time]] = df[[time]]
        X[[group]] = df[[group]]
        pmax = 1
        steps = steps
        cph = CoxPHFitter()

        for i in range(steps):
            if pmax > pmin:
                X = df[lst]
                X[[group]] = df[[group]]
                X[[time]] = df[[time]]
                model = cph.fit(X, duration_col=time, event_col=group)
                pvalues = model.summary.iloc[:,8].sort_values()
                pmax = pvalues[-1]
                out = pvalues.index[-1]
                lst.remove(out)
            else:
                break

        result=cph.fit(X, duration_col=time, event_col=group)
        HR = round(result.hazard_ratios_, 2).to_frame()
        p = round(result.summary.iloc[:,8], 3).to_frame()
        conf0 = round(result.summary.iloc[:,5], 2).to_frame()
        conf1 = round(result.summary.iloc[:,6], 2).to_frame()
        back = pd.concat([HR, conf0, conf1, p], axis = 1).reset_index()
        back.columns = ['Фактор','HR', 'Нижний 95% ДИ', 'Верхний 95% ДИ', 'Значимость, р']
        return back

    else:
        print('ERROR: family must be logistic or cox')
