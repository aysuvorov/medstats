
import numpy as np
import pandas as pd
import scipy.stats as st
from seaborn import palettes
import statsmodels as sm
import statsmodels.api as sma
import matplotlib.pyplot as plt
import seaborn as sns
import rpy2
import statsmodels.stats.api as sms

from unicodedata import normalize
from scipy.stats.stats import ttest_ind
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import shapiro, kstest, ttest_ind, ttest_rel, mannwhitneyu, fisher_exact, chi2_contingency, kruskal, wilcoxon, f_oneway
from pandas.api.types import CategoricalDtype
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from statsmodels.formula.api import ols, logit, mixedlm, gee, poisson
from numpy.linalg import LinAlgError
from lifelines import KaplanMeierFitter

## Rpy2 env
import rpy2.robjects as ro

from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
env = ro.r.globalenv()

from rpy2.robjects import pandas2ri, FloatVector, IntVector, FactorVector, Formula 

import rpy2.robjects.numpy2ri as rpyn
rpyn.activate()

stats = importr('stats')
base = importr('base')
emmeans = importr('emmeans')
afex = importr('afex')

###################################################################
###################################################################

"""
Simple filler for subgroups in data frame.

Main function is `group_simple_imputer`

group_simple_imputer()
====================

Arguments:
---------
df - dataframe
columns_lst - list of columns to replace NaNs with group columns
group - grouping column
func - if None (default) fills with zero, else - fills with median

missing_columns_names()
=====================
Get the list of names of all columns with missings

Arguments:
---------
df- dataframe

Example:
=======

df = group_simple_imputer(df, 
    ['Группа терапии'] + missing_columns_names(df), 
    'Группа терапии', 
    func=9)    

"""

# subgroup_simple_filler
def subgroup_simple_filler(df, group, category, func=None):

    A = df[df[group] == category].isnull().sum()
    impute_names = A[A>0].index
    del A

    if func:
        for var in impute_names:
            df.loc[df[df[group] == category].index, [var]] = df.loc[df[df[group] == category].index, [var]].fillna((df.loc[df[df[group] == category].index, [var]].astype(float).median()))
            if df.loc[df[df[group] == category].index, [var]].isnull().sum()[0] > 0:
                print('Error chars in: ' + str(var))

    else:    
        df.loc[df[df[group] == category].index, impute_names] = df.loc[df[df[group] == category].index, impute_names].fillna(0)
    return(df)

# missing_columns_names
def missing_columns_names(df):
    return(list(df.isnull().sum()[df.isnull().sum() > 0].index))

# group_simple_imputer
def group_simple_imputer(df, column_lst, group, func=None):

    B = df[column_lst]

    for i in list(set(B[group])):
        subgroup_simple_filler(B, group, i, func=func)

    for col in B.columns:
        df[col] = B[col]

    return(df)


def locf_filler(df, group, word_pattern, func=None, inplace=False):

    data = df[['Группа терапии'] + [x for x in df.columns if word_pattern in x]]

    data = group_simple_imputer(data, data.columns[:2], 'Группа терапии', func=func)

    data = data.T.fillna(method='ffill').T

    if inplace:
        for col in data.columns:
            df[col] = data[col]
        return(df)
    else:
        return(data)


"""
Draw dynamics for longitudinal data
"""

## Binary

def draw_cat_repeated(
    df, 
    group, 
    word, 
    ylabel, 
    pict_sav=False, 
    time_vals_lst=None, 
    figsize=(8, 5),
    xlabel='Визит'
    ):

    col_lst = [group] + [x for x in df.columns if word in x]
    data = df[col_lst].copy()

    if time_vals_lst:
        data.columns = [group] + time_vals_lst
    
    else:
        data.columns = [group] + list(range(0, len([x for x in df.columns if word in x])))

    data = pd.melt(data, id_vars=group)
    data.columns = ['group','time', 'val']
    data['time'] = data['time'].astype(int)

    B = data.copy()
    B['val'] = B['val'].astype(int)
    B = B.groupby(['group', 'time']).mean() *100 
    B = B.reset_index()

    sns.set(style='whitegrid')
    f, ax = plt.subplots(figsize = figsize)
    g = sns.barplot(x="time", 
        y="val", 
        hue = 'group', 
        data=B,
        ci=None)
    plt.ylabel(ylabel + ', %')
    plt.xlabel(xlabel)
    g.legend().set_title(None)
    plt.show()

    if pict_sav:
        g.figure.savefig(ylabel  + ' - percent plot.png')


# Numeric

def draw_num_repeated(
    df, 
    group, 
    word, 
    ylabel, 
    pict_sav=False, 
    time_vals_lst=None, 
    figsize=(8, 5),
    xlabel='Визит'
    ):
    

    col_lst = [group] + [x for x in df.columns if word in x]

    data = df[col_lst].copy()

    if time_vals_lst:
        data.columns = [group] + time_vals_lst
    
    else:
        data.columns = [group] + list(range(0, len([x for x in df.columns if word in x])))

    data = pd.melt(data, id_vars=group)
    data.columns = ['group','time', 'val']

    sns.set(style='whitegrid')
    f, ax = plt.subplots(figsize = figsize)
    g = sns.pointplot(x="time", 
        y="val", 
        hue = 'group', 
        data=data,
        #ci=0.025,
        dodge=0.25)
    plt.ylabel(ylabel)
    plt.xlabel('Визит')
    g.legend().set_title(None)
    plt.show()

    if pict_sav:
        g.figure.savefig(ylabel  + ' - anova plot.png', bbox_inches='tight')




"""
Regression models
"""

## Logit model with group:baseline and group^time interaction

def logit_repeated_category(df, group, word, name, pict=True, pict_sav=False, time=None, figsize=(8, 5)):

    col_lst = [group] + [x for x in df.columns if word in x]

    data = df[col_lst].copy()
    data.columns = [group] + list(range(0, len([x for x in df.columns if word in x])))

    data = pd.melt(data, id_vars=group)
    data.columns = ['group','time', 'val']
    data['time'] = data['time'].astype(int)
    if time:
        data['time'] = data['time'].replace(sorted(list(set(data['time'].dropna()))), time)

    else:
        pass

    B = data.copy()
    B['val'] = B['val'].astype(int)
    B = B.groupby(['group', 'time']).mean() *100 
    B = B.reset_index()

    if pict:
        sns.set(style='whitegrid')
        f, ax = plt.subplots(figsize = figsize)
        g = sns.barplot(x="time", 
            y="val", 
            hue = 'group', 
            data=B,
            ci=None)
        plt.ylabel(name + ', %')
        plt.xlabel('Визит')
        g.legend().set_title(None)
        plt.show()

        if pict_sav:
            g.figure.savefig(name  + ' - percent plot.png')

    # Regression
    col_lst = [group] + [x for x in df.columns if word in x]

    tab = df[col_lst].copy()
    tab.columns = [group] + list(range(0, len([x for x in tab.columns if word in x])))

    tab = pd.melt(tab, id_vars=tab.columns[0:2])
    tab.columns = ['group','baseline','time', 'val']
    tab['time'] = tab['time'].astype(int) 
    tab['baseline'] = tab['baseline'].astype(int) 
    tab['val'] = tab['val'].astype(int) 

    try:
        mod = logit("val ~ C(group) + baseline + time + baseline*C(group) + C(group)*time", tab).fit()
        return(mod.summary())

    except LinAlgError:
        print(name, ' Regression error singular matrix!\n')


## GEE poisson for categoricals

def mixed_repeated_category(df, group, word, name, pict=True, pict_sav=False):

    col_lst = [group] + [x for x in df.columns if word in x]

    data = df[col_lst].copy()
    data.columns = [group] + list(range(0, len([x for x in df.columns if word in x])))

    data = pd.melt(data, id_vars=group)
    data.columns = ['group','time', 'val']
    data['time'] = data['time'].astype(int) 

    B = data.copy()
    B['val'] = B['val'].astype(int)
    B = B.groupby(['group', 'time']).mean() *100 
    B = B.reset_index()

    if pict:
        sns.set(style='whitegrid')
        g = sns.barplot(x="time", 
            y="val", 
            hue = 'group', 
            data=B,
            ci=None)
        plt.ylabel(name + ', %')
        plt.xlabel('Визит')
        g.legend().set_title(None)
        plt.show()

        if pict_sav:
            g.figure.savefig(name  + ' - percent plot.png', bbox_inches='tight')

    
    data = data.groupby(['group', 'time']).agg({'val':['sum', 'count']}).reset_index().droplevel([0], axis=1)
    data['val'] = data['sum'] / data['count'] * 100
    data = data.drop(['sum', 'count'], axis=1)
    data.columns = ['group', 'time', 'val']
    data['uin'] = data.index
    
    fam = sma.families.Poisson()
    ind = sma.cov_struct.Autoregressive()

    try:
        mod = gee("val ~ C(group) + time + C(group)*time", "uin", data, cov_struct=ind, family=fam).fit()

    except:
        data['val_2'] = data['val'].astype(int) + 0.001
        mod = gee("val_2 ~ C(group) + time + C(group)*time", "uin", data, cov_struct=ind, family=fam).fit()  
    
    return(mod.summary())

## ANOVA RM type III for numerics through R magic

def mixed_repeated_numeric(df, uin, group, word, name, transform='log', pict=True, pict_sav=False, print_contrasts=True):
    """
    Создание смешанных линейных моделей для оценки взаимодействия времени и группы
    
    - Преобразование осуществляется через ранжирование значений в длинном формате
    - В таблице не додлжно быть пропусков

    df - таблица с данными
    uin - ключ индивидуального пациента
    group - переменная группы
    word - фрагмент переменной, определяющей визит и признак (например "кашель" для всех переменных с кашлем)
    name - название переменной, как оно будет на графике
    transform - преобразование данных. Ранжирование ('rank'), логарифмирование ('log'), box-cox ('bc'), 
        'none' - без трансформирования
    pict - показывать ли график
    model - тип модели - 'mixed' или 'gee'
    pict_sav - сохранять ли график
    """

    col_lst = [group, uin] + [x for x in df.columns if word in x]

    data = df[col_lst].copy()
    data.columns = [group, uin] + list(range(len([x for x in df.columns if word in x])))
    data = pd.melt(data, id_vars=[group, uin])
    data.columns = ['group','id','time', 'val']

    if pict:
        sns.set(style='whitegrid')
        g = sns.pointplot(x="time", 
            y="val", 
            hue = 'group', 
            data=data,
            ci=None,
            dodge=True)
        plt.ylabel(name)
        plt.xlabel('Визит')
        g.legend().set_title(None)
        plt.show()

        if pict_sav:
            g.figure.savefig(name  + ' - anova plot.png', bbox_inches='tight')

    if transform == 'log':
        try:
            data['val'] = data['val'] + 0.001
            data['val'] = np.log(data['val'])
        except:
            print('Log error!')

    elif transform == 'bc':
        try:
            data['val'] = data['val'] + 0.001
            data['val'], _ = boxcox(data['val'])
        except:
            print('Box-Cox error!')

    elif transform == 'rank':
        try:
            data['val'] = data['val'].rank()
        except:
            print('Rank error!')
    else:
        data['val'] = data['val'].astype(float)

    data['time'] = data['time'].astype('category')

    for col in [x for x in data.columns if pd.CategoricalDtype.is_dtype(data[x]) == True]:
        data[col] = data[col].astype(str)
        data[col] = data[col].astype('category')

    with localconverter(ro.default_converter + pandas2ri.converter):
        env['data'] = ro.conversion.py2rpy(data)


    A = ro.r('''

    library('emmeans')
    library('afex')

    data$UIN = factor(data$id)
    data$group = factor(data$group)
    data$time = factor(data$time)

    fit_afex = aov_ez("id", "val", data=data, between = "group", within = "time", check_contrasts=TRUE)
    print(summary(fit_afex))


        ''')

    if print_contrasts:
        #print(A)
        

    #else:
        B = ro.r('''

        #print('EMMEANS')

        #r1 <- emmeans(fit_afex, ~time*group)
        #print(r1)
        #print(emmeans(r1, 'group', contr = "pairwise"))
        #print(emmeans(r1, 'time', contr = "pairwise"))
        #B = emmeans(r1, c('group', 'time'), contr = "pairwise")
        #print(B)
        #print(confint(B))
            ''')

        
        #print(A, '\n')
        print(B)



def mixed_repeated_numeric_2(df, group, word, name, 
    transform='log', model='mixed', pict=True, pict_sav=False,
    time=None, figsize=(8, 5)):
    """
    Создание смешанных линейных моделей для оценки взаимодействия времени и группы
    
    - Преобразование осуществляется через ранжирование значений в длинном формате
    - В таблице не додлжно быть пропусков

    df - таблица с данными
    group - переменная группы
    word - фрагмент переменной, определяющей визит и признак (например "кашель" для всех переменных с кашлем)
    name - название переменной, как оно будет на графике
    transform - преобразование данных. Ранжирование ('rank'), логарифмирование ('log'), box-cox ('bc'), 
        'none' - без трансформирования
    pict - показывать ли график
    model - тип модели - 'mixed' или 'gee'
    pict_sav - сохранять ли график
    """

    col_lst = [group] + [x for x in df.columns if word in x]

    data = df[col_lst].copy()
    
    data.columns = [group] + list(range(len([x for x in df.columns if word in x])))

    data = pd.melt(data, id_vars=[group])

    data.columns = ['group','time', 'val']

    if time:
        data['time'] = data['time'].replace(sorted(list(set(data['time'].dropna()))), time)

    else:
        pass

    if pict:
        sns.set(style='whitegrid')
        f, ax = plt.subplots(figsize = figsize)
        g = sns.pointplot(x="time", 
            y="val", 
            hue = 'group', 
            data=data,
            ci=None,
            dodge=True)
        plt.ylabel(name)
        plt.xlabel('Визит')
        g.legend().set_title(None)
        plt.tight_layout()

        plt.show()

        if pict_sav:
            g.figure.savefig(name  + ' - anova plot.png', bbox_inches='tight')

    if transform == 'log':
        try:
            data['val'] = data['val'] + 0.001
            data['val'] = np.log(data['val'])
        except:
            print('Log error!')

    elif transform == 'bc':
        try:
            data['val'] = data['val'] + 0.001
            data['val'], _ = boxcox(data['val'])
        except:
            print('Box-Cox error!')

    elif transform == 'rank':
        try:
            data['val'] = data['val'].rank()
        except:
            print('Rank error!')
    else:
        pass

    data['time'] = data['time'].astype(int)

    data = data.groupby(['group','time']).mean('val').reset_index()

    data['uin'] = data.index

    fam = sma.families.Gaussian()

    ind = sma.cov_struct.Autoregressive()

    if model == 'mixed':
        mod = mixedlm("val ~ C(group) + time + C(group)*time", data, groups=data["uin"], re_formula="~C(group)").fit()
        #mod = mixedlm("val ~ C(group) + time + C(group)*time", data, groups=data["uin"]).fit()

    else:
        mod = gee("val ~ C(group) + time + C(group)*time", "uin", data, cov_struct=ind, family=fam).fit()

    return(mod.summary()) 

"""
Feron times of categoricals
"""

def Feron_times_categoricals(df, group, word_lst, names_lst):
    """
    Считает сроки категориальных переменных у тех, у кого есть симптом

    df - таблица, 
    group - группирующая переменная, 
    word_lst - список с уникальным куском слова по каждому симптому, 
    names_lst - список с приличными названиями, соответствующими word_lst
    """

    A = pd.Series()

    for w, j in zip(word_lst, names_lst):

        B = pd.Series()
        col_lst = [group] + [x for x in df.columns if w in x]
        data = df[col_lst]
        B = data.iloc[:, 1:].sum(axis=1).replace(0, np.nan)
        A = pd.concat([A, B], axis=1)
    A.columns = [group] + names_lst
    A[group] = df[group]

    return(A)


def shuvalov_plot(
        A, 
        group, 
        group_category_lst=['Виферон', 'Другая терапия'], 
        annot_lst=None, 
        mean_vals=False,
        days_lim=10):
    """
    Строит barplot по срокам симптомов, как это хочет Шувалов, с использованием
    значимости

    A - таблица из предыдущей фугкции, 
    group - группирующая переменная, 
    group_category_lst=['Виферон', 'Другая терапия'] - лист с категориями группирующей переменной
    annot_lst - список с введением значимости вручную
    mean_vals - указывать ли средние значения (если False - указываются медианы)

    """

    B = pd.melt(A, id_vars=group).dropna()

    annot = []
    coords1 = []
    coords2 = []
    coords = []
    means1 = []
    means2 = []

    for var in list(B['variable'].drop_duplicates()):
        g1 = B[(B[group] == group_category_lst[0]) & (B['variable'] == var)]['value']
        g2 = B[(B[group] == group_category_lst[1]) & (B['variable'] == var)]['value']

        try:
            g_max1 = np.mean(B[(B[group] == group_category_lst[0]) & (B['variable'] == var)]['value']) + \
                np.std(B[(B[group] == group_category_lst[0]) & (B['variable'] == var)]['value'])
            g_max1 = g_max1 + 0.05 * g_max1
        except: 
            g_max1 = 0
        try:
            g_max2 = np.mean(B[(B[group] == group_category_lst[1]) & (B['variable'] == var)]['value']) + \
                np.std(B[(B[group] == group_category_lst[1]) & (B['variable'] == var)]['value'])
            g_max2 = g_max2 + 0.05 * g_max2
        except: 
            g_max2 = 0

#        g_max = np.nanmax([g_max1, g_max2]) + np.nanmax(np.max(A)/A.shape[1]) #1 #g_max + 1#0.2 * g_max 
        g_max = np.nanmax([g_max1, g_max2]) + \
            np.nanmax([g_max1, g_max2]) * 0.05


        if mean_vals:
            try:
                m1 = float(round(np.mean(g1),1))
            except:
                m1 = ' '
            try:
                m2 = float(round(np.mean(g2),1))
            except:
                m2 = ' '

            if ttest_ind(g1, g2)[1] >= 0.05:
                annot = annot + [' ']
            else:
                annot = annot + ['*']

        else:

            try:
                m1 = int(round(np.median(g1),0))
            except:
                m1 = ' '
            try:
                m2 = int(round(np.median(g2),0))
            except:
                m2 = ' '

            try:
                if mannwhitneyu(g1, g2)[1] >= 0.05:
                    annot = annot + [' ']
                else:
                    annot = annot + ['*']
            except:
                annot = annot + ['*']
        
        coords = coords + [g_max]
        coords1 = coords1 + [g_max1]
        coords2 = coords2 + [g_max2]
        means1 = means1 + [m1]
        means2 = means2 + [m2]    

        ## Draw

    if annot_lst:
        annot = annot_lst
        # annot=[' ', ' ', ' ', '*', '*', ' ', '*', ' ', ' ', ' ', ' ', ' ', '*', ' ', ' ']

    sns.set(style='whitegrid')

    plt.figure(figsize=(15,8))
    #g = sns.boxplot(x="variable", 
    #        y="value", 
    #        hue = group, 
    #        data=B,
    #        linewidth=0.7,
    #        width=0.7)

    g = sns.barplot(x="variable", 
        y="value", 
        hue = group, 
        data=B,
        ci='sd',
        palette=['darkorange', 'steelblue'])
    plt.ylim(0, days_lim)

    names = list(B['variable'].drop_duplicates())

    for i,j,txt in zip(list(range(len(names))), coords, annot):
        plt.text(i, j, txt)

    for i,j1, j2, r,c in zip(list(range(len(names))), coords1, coords2, means1, means2):
        plt.text(i-0.3, j1, r)
        plt.text(i+0.1, j2, c)

    plt.xlabel('')
    plt.ylabel('Длительность, дни')
    plt.ylim(0,None)
    plt.xticks(rotation=90)
    g.legend().set_title(None)
    plt.show()
    return(list(annot))


"""
Функция отрисовки КМ для длительности категориальных симптомов для Мили

"""

def k_m_feron_plotter(
    df, 
    group_var, 
    time_var, 
    fillNA = True, 
    save=False, 
    plot_name=None,
    xlabel = 'Длительность терапии, дни'
    ):

    tab = df[[time_var, group_var]]
    tab['outcome'] = 1

    if fillNA:
        for col in tab.columns:
            try:
                tab[col] = tab[col].fillna(0)
            except:
                pass
    else:
        tab = tab.dropna()

    plt.style.use('seaborn-whitegrid')
    kmf = KaplanMeierFitter()

    T = time_var
    E = 'outcome'

    fig, ax = plt.subplots(figsize=(8,4))


    for group in tab[group_var].unique():
        kmf.fit(tab[tab[group_var] == group][T], tab[tab[group_var] == group][E], label=group)
        ax = kmf.plot(ax=ax, ci_show=False, cmap='coolwarm')

    g = df[T].dropna().astype(float).max()
    try:
        g = round(g)
    except:
        g = 0
    plt.xticks(np.array(range(0, g+1)))
    plt.title(T)
    plt.xlabel(xlabel)
    plt.ylabel('Кумулятивная доля имеющих симптом, %')
    plt.xlim([0.5, None])
    
    
    if save:
        plt.savefig(plot_name + ".png")   