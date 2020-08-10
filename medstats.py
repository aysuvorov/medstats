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


from sklearn.utils import resample
from sklearn.metrics import roc_curve, auc
from dask import delayed
from numba import jit
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, fisher_exact, chi2_contingency, kruskal
from lifelines import CoxPHFitter

"""
filler

"""
def filler(df, func):
	for col in df:
		df[col].apply(pd.to_numeric, errors='coerce')

	df = df.fillna(func())
	return df

"""
prop_size

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
mean_size

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
bs_multi

"""

@jit
def bs_multi_job_rand(x, y):
    b = np.random.choice(x, len(y))
    return b
@jit
def bs_multi_job_perm(x,y):
    b = np.random.permutation(np.concatenate((x, y)))
    return b


def bs_multi_job(a_shifted, a, b_shifted, b, func):
    a_bs = bs_multi_job_rand(a_shifted, a)
    b_bs = bs_multi_job_rand(b_shifted, b)
    permuted_data = bs_multi_job_perm(a_bs, b_bs)
    perm_sample_1 = permuted_data[:len(a)]
    perm_sample_2 = permuted_data[len(a):]
    g = func(perm_sample_1) - func(perm_sample_2)
    return g

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
            B[i] = bs_multi_job(a_shifted, a, b_shifted, b, func)
        p[val] = (np.sum(abs(B) >= abs(t)) / (R+1))
    return(np.percentile(p, [2.5, 50, 97.5]))

# SAMPLE:
# funcs = [np.mean, np.median, np.std, st.iqr]
#
# for f in funcs:
#    print(bs_multi(a=a, b=b, func = f))
    
"""
bs_perc

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

    return(np.percentile(p, [2.5, 50, 97.5]))
    
"""
bs_props

"""    

def bs_props(inv, inv_n, plac, plac_n, R=100):
    diff = inv - plac
    ni = (np.array(inv*inv_n/100)).astype(int)
    t0 = np.zeros(inv_n - ni)
    t1 = np.ones(ni)

    npc = (np.array(plac*plac_n/100)).astype(int)
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
dummification

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
summary

"""

def to_array(df, col):
    return df[[col]].to_numpy()[:,0]

@jit(nopython = True)
def cat_perc(var):
    n = round(np.sum(var),0)
    percents = round(n / len(var)*100, 1)
    return n, percents

@jit(nopython = True)
def summ_numer(var):
    avg = np.mean(var) 
    sd = np.std(var) 
    mn = np.min(var) 
    mx = round(np.max(var),1)
    md = round(np.median(var),1)
    c25 = np.percentile(var, 25)
    c75 = np.percentile(var, 75)
    return avg, sd, mn, mx, md, c25, c75    

def summary(df):
    summarize = pd.DataFrame()
    for col in df:
        J = len(list(df.columns))
        var = to_array(df, col)
        for j in range(J):
            if len(np.unique(var)) == 1:
                v = df[col].name
                vartype = 'Уникальная'
                n = round(len(var),0)
                med = avg = minn = maxx = sh = np.unique(var)
            elif len(np.unique(var)) < 3:
                v = df[col].name
                vartype = 'Категориальная'
                n, percents = cat_perc(var)
                percents = str(percents).join(' %')
                med = avg = minn = maxx = sh = '-'
            else:
                v = df[col].name
                vartype = 'Числовая'
                n = round(len(var),0)
                percents = '-'
                avg, sd, minn, maxx, med, c25, c75 = summ_numer(var)
                avg = ''.join([str(round(avg, 1)), ' ± ', str(round(sd, 1))])
                med = ''.join([str(round(med,1)), ' (',str(round(c25,1)),'; ',str(round(c75, 1)),')'])
                sh = round(shapiro(var)[1], 3)
        summarize = summarize.append({'Фактор': v, 'Тип': vartype, 'Количество': '% 6.0f' % n, 'Доля, %': percents,'Медиана и 25/75 перцентили': med, 'Среднее и ст. отклонение': avg , \
                                      'Мин': minn, 'Макс': maxx, 'Критерий Шапиро-Уилка, р': sh}, ignore_index=True)
        summarize = summarize.reindex(columns=['Фактор', 'Тип', 'Количество', 'Доля, %', 'Мин','Медиана и 25/75 перцентили', 'Макс', 'Среднее и ст. отклонение', 'Критерий Шапиро-Уилка, р'])

    if save_tab == True:
        return pd.DataFrame.to_excel(summarize, 'Описательные статистики.xlsx')
    else:
        return summarize

"""
compare

"""

@jit(nopython = True)
def compare_cat(x_var, y_var):
    gr1obs = np.sum(x_var)
    gr2obs = np.sum(y_var)
    gr1exp = len(x_var) - gr1obs
    gr2exp = len(y_var) - gr2obs
    return gr1obs, gr2obs, gr1exp, gr2exp

@jit(nopython = True)
def compare_numer_unnorm(x_var, y_var):
    gr1obs = round(np.median(x_var),2)
    gr2obs = round(np.median(y_var),2)
    cent1 = np.percentile(x_var, [25,75])
    cent2 = np.percentile(y_var, [25,75])
    return gr1obs, gr2obs, cent1, cent2

@jit(nopython = True)
def compare_numer_norm(x_var, y_var):
    gr1obs = round(np.median(x_var),2)
    gr2obs = round(np.median(y_var),2)
    sd1 = np.std(x_var)
    sd2 = np.std(y_var)
    return gr1obs, gr2obs, sd1, sd2


def compare(df, group, gr_id_1 = 0, gr_id_2 = 1, name_1 = 'Группа 0', name_2 = 'Группа 1', test = 'mw', save_tab = False):

    x = df.loc[df[group] == gr_id_1].drop(columns = group)
    y = df.loc[df[group] == gr_id_2].drop(columns = group)

    comparison = pd.DataFrame()

    for col in x:
        J = len(list(x.columns))
        x_var = to_array(x, col)
        y_var = to_array(y, col)
        for j in range(J):
            if len(np.unique(df[col])) < 3:
                v = df[col].name 
                gr1obs, gr2obs, gr1exp, gr2exp = compare_cat(x_var, y_var)
                # в таблицу идут:
                p_val = round(fisher_exact(np.array([[gr1obs, gr2obs], [gr1exp, gr2exp]]))[1], 3)
                p1 = ''.join([str(round(gr1obs/len(x_var)*100,1)), " %", " (", str(gr1obs),")"])
                p2 = ''.join([str(round(gr2obs/len(y_var)*100,1)), " %", " (", str(gr2obs), ")"])
            else:
                v = df[col].name
                if test == 'mw':
                    gr1obs, gr2obs, cent1, cent2 = compare_numer_unnorm(x_var, y_var)
                    # в таблицу идут:
                    p_val = round(mannwhitneyu(x[col], y[col])[1], 3)
                    p1 = ''.join([str(gr1obs), " (", str(round(cent1[0],2)), "; ", str(round(cent1[1],2)), ")"])
                    p2 = ''.join([str(gr2obs), " (", str(round(cent2[0],2)), "; ", str(round(cent2[1],2)), ")"])
                elif test == 'tt':
                    gr1obs, gr2obs, sd1, sd2 = compare_numer_norm(x_var, y_var)
                    p_val = round(ttest_ind(x_var, y_var, equal_var = False)[1], 3)
                    p1 = ''.join([str(gr1obs), " ± ", str(round(sd1, 1))])
                    p2 = ''.join([str(gr2obs), " ± ", str(round(sd2, 1))])
                else:
                    print('ERROR: test is mw or tt (Welch)')

        comparison = comparison.append({'Фактор': v, name_1: p1, name_2: p2,'p_val': p_val}, ignore_index=True)
        comparison = comparison.reindex(columns=['Фактор', name_1, name_2, 'p_val'])

    if save_tab == True:
        return pd.DataFrame.to_excel(comparison, 'Сравнение по группам.xlsx')
    else:
        return comparison
        
"""
regr_onedim

"""

def regr_onedim(df, group, adjusted = False, signif_only = False, age_col = 1, sex_col = 1, save_tab = False):

    reg_data = df.drop(columns=group)
    y = df[[group]]
    logregr = pd.DataFrame()

    if adjusted == False:
        for col in reg_data.columns:
            J = len(reg_data.columns)
            for j in range(J):
                v = reg_data[col].name
                logit_model=sma.GLM(y,sma.add_constant(reg_data[[col]]), family = sma.families.Binomial())
                result=logit_model.fit()
                params = round(np.exp(result.params)[1], 2)
                conf0 = round(np.exp(result.conf_int())[0][1],2)
                conf1 = round(np.exp(result.conf_int())[1][1],2)
                p = round(result.pvalues[1], 3)
            logregr = logregr.append({'Names': v, 'OR': '{0:.2f}'.format(params), 'lower': '{0:.2f}'.format(conf0), 'upper': '{0:.2f}'.format(conf1),'p_val': p}, ignore_index=True)
            logregr = logregr.reindex(columns=['Names', 'OR', 'lower', 'upper', 'p_val']) 

    else:
        for col in reg_data.columns:
            J = len(reg_data.columns)
            for j in range(J):
                v = reg_data[col].name
                logit_model=sma.GLM(y,sma.add_constant(reg_data[[col, age_col, sex_col]]), family = sma.families.Binomial())
                result=logit_model.fit()
                params = round(np.exp(result.params)[1], 2)
                conf0 = round(np.exp(result.conf_int())[0][1],2)
                conf1 = round(np.exp(result.conf_int())[1][1],2)
                p = round(result.pvalues[1], 3)
            logregr = logregr.append({'Names': v, 'OR': '{0:.2f}'.format(params), 'lower': '{0:.2f}'.format(conf0), 'upper': '{0:.2f}'.format(conf1),'p_val': p}, ignore_index=True)
            logregr = logregr.reindex(columns=['Names', 'OR', 'lower', 'upper', 'p_val'])[(logregr['Names'] != age_col) & (logregr['Names'] != sex_col)]


    if signif_only == True:
        logregr = logregr[logregr['p_val'] < pmin]
    else:
        pass

    if save_tab == True:
        return pd.DataFrame.to_excel(logregr, 'Одномерный регрессионный анализ.xlsx')
    else:
        return logregr

"""
regr_multi

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
roc_cut


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
cox_onedim

EXAPLE:
cox_onedim(df, 'DEATH', 'TIME')

"""

def cox_onedim(df, group, time, adjusted = False, signif_only = False, age_col = 1, sex_col = 1, save_tab = False):

    reg_data = df.drop(columns=[group, time])
    cph = CoxPHFitter()

    coxregr = pd.DataFrame()

    if adjusted == False:
        for col in reg_data:
            v = reg_data[col].name
            model = cph.fit(df[[col, group, time]], duration_col=time, event_col=group)
            HR = round(model.hazard_ratios_[0], 2)
            p = round(model.summary.iloc[:,8][0], 3)
            conf0 = round(model.summary.iloc[:,5][0], 2)
            conf1 = round(model.summary.iloc[:,6][0], 2)
            coxregr = coxregr.append({'Фактор': v, 'HR': HR, 'Нижний 95% ДИ': conf0, 'Верхний 95% ДИ': conf1,'p_val': p}, ignore_index=True)
    
        coxregr = coxregr.reindex(columns=['Фактор', 'HR', 'Нижний 95% ДИ', 'Верхний 95% ДИ', 'p_val'])

    else:
        for col in reg_data:
            v = reg_data[col].name
            model = cph.fit(df[[col, age_col, sex_col, group, time]], duration_col=time, event_col=group)
            HR = round(model.hazard_ratios_[0], 2)
            p = round(model.summary.iloc[:,8][0], 3)
            conf0 = round(model.summary.iloc[:,5][0], 2)
            conf1 = round(model.summary.iloc[:,6][0], 2)
            coxregr = coxregr.append({'Фактор': v, 'HR': HR, 'Нижний 95% ДИ': conf0, 'Верхний 95% ДИ': conf1,'p_val': p}, ignore_index=True)
    
        coxregr = coxregr.reindex(columns=['Фактор', 'HR', 'Нижний 95% ДИ', 'Верхний 95% ДИ', 'p_val'])[(coxregr['Фактор'] != age_col) & (coxregr['Фактор'] != sex_col)]        

    if signif_only == True:
        coxregr = coxregr[coxregr['p_val'] < pmin]
    else:
        pass
	
    if save_tab == True:
        return pd.DataFrame.to_excel(coxregr, 'Регрессия Кокса, одномерный анализ.xlsx')
    else:
        return coxregr

    return coxregr

"""
backwise

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
