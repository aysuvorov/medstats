
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

# import sys
# sys.path.append('/home/guest/Документы/medstats/src/cxs')


# Functions

"""
Data cleaning
"""

def columnn_normalizer(df, col_lst):
    """
    Удаление бешеных разделителей в колонках
    """
    for col in col_lst:
        for i in range(len(df[col])):
            try:
                df[col][i] = normalize('NFKC', df[col][i])
                df[col][i] = df[col][i].replace(',','.')
                try:
                    df[col][i] = float(df[col][i])
                except ValueError:
                    pass
            except TypeError:
                pass
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass
    
    return(df)


def column_categorical(df, col_lst):
    for col in df.columns:
        if col in col_lst:
            df[col] = df[col].astype('category')
        else:
            pass
    
    return(df)


def miss_counter(data):

    missing_df = pd.DataFrame(data.isnull().sum())
    missing_df.columns = ['Miss_abs_counts']
    missing_df['Valid_abs_counts'] = data.shape[0] - missing_df['Miss_abs_counts']
    missing_df['Miss_Rates,%'] = missing_df['Miss_abs_counts']/data.shape[0]
    missing_df['Valid_Rates,%'] = missing_df['Valid_abs_counts']/data.shape[0]
    return(missing_df[['Valid_abs_counts', 'Valid_Rates,%', 'Miss_abs_counts', 'Miss_Rates,%']])


def p_adjust(vector, n, method = 'BH'):

    vector = FloatVector(np.asarray(vector))
    new_vec = []
    for i in vector:
        new_vec = new_vec + [float(stats.p_adjust(i, n=n, metho=method))]

    return new_vec


def dplyr_filter(df, filter_var, value_lst):
    """
    Filters `filter_var` variable of df data frame and returns data frame
    with only `value_lst` values in `filter_var`.
    Works just as `dplyr filter()` function

    Parameters
    ----------
    df: data frame
    filter_var: variable to filter
    value_lst: values to dtore in filter_var variable

    Returns
    -------
    filtered pandas.DataFrame

    """

    index_lst = [i for part in value_lst for i in df[df[filter_var] == part].index ]
    return df.loc[index_lst, :]

"""
Descriptive statistics
"""
## Simple descriptives

def series_num_summary(d, digits):
    s = d.dropna()
    name = s.name
    valid = s.count()
    M = np.mean(s)
    SD = np.std(s)
    miin = np.min(s)
    try:
        p2_5 = np.percentile(s, 2.5)
        p25 = np.percentile(s, 25)
        Med = np.median(s)
        p75 = np.percentile(s, 75)
        p97_5 = np.percentile(s, 97.5)
        maax = np.max(s)
    except IndexError:
        p2_5 = 0
        p25 = 0
        Med = 0
        p75 = 0
        p97_5 = 0
        maax = 0

    if len(pd.unique(s)) < 2:
        sh = 'Уникальная'

    elif len(pd.unique(s)) < 3:
        sh = 'Не применим'

    else:
        sh = round(shapiro(s)[1],3)

    a = pd.DataFrame({'Фактор': name,
                    'Категории': '-',
                    'Валидные': valid,
                    'Количество': '-',
                    'Доля': '-',
                    'Среднее': round(M,digits),
                    'Ст.откл.': round(SD,digits),
                    'Мин':round(miin, digits),
                    '2.5%': round(p2_5,digits),
                    '25%':round(p25, digits),
                    'Медиана': round(Med,digits),
                    '75%':round(p75,digits),
                    '97.5%': round(p97_5,digits),
                    'Макс':round(maax,digits),
                    'Значимость Ш-У': sh}, index=[0])

    return(a)


def series_cat_summary(d):
    s = d.dropna()
    name = s.name

    if len(pd.unique(s)) < 6:

        valid = len(s.dropna())
        abs = s.dropna().value_counts()
        perc = round(s.dropna().value_counts() / (s.dropna().shape[0])*100,1)
        ind = perc.reset_index()['index']
        ind.index = perc.index=abs.index
        M = SD = miin = p25 = Med = p75 = p2_5 = p97_5 = maax = '-'
    
        if len(pd.unique(s)) < 2:
            sh = 'Уникальная'
        else:
            sh = '-'

        a = pd.DataFrame({'Фактор': name,
                    'Категории': ind,
                    'Валидные': valid,
                    'Количество': abs,
                    'Доля': perc,
                    'Среднее': M,
                    'Ст.откл.': SD,
                    'Мин':miin,
                    '2.5%': p2_5,
                    '25%':p25,
                    'Медиана': Med,
                    '75%':p75,
                    '97.5%': p97_5,
                    'Макс':maax,
                    'Значимость Ш-У': sh})

    else:
        abs = perc = ind = 'Более 5'
        M = SD = miin = p25 = Med = p75 = maax = valid = p2_5 = p97_5 = sh = '-'

        a = pd.DataFrame({'Фактор': name,
                    'Категории': ind,
                    'Валидные': valid,
                    'Количество': abs,
                    'Доля': perc,
                    'Среднее': M,
                    'Ст.откл.': SD,
                    'Мин':miin,
                    '2.5%': p2_5,
                    '25%':p25,
                    'Медиана': Med,
                    '75%':p75,
                    '97.5%': p97_5,
                    'Макс':maax,
                    'Значимость Ш-У': sh}, index=[0])
         
    return(a)           


def summary_all(df, merge_stats=True, digits=1):

    data = df
    tb = pd.DataFrame()

    for col in data.columns:
        if data[col].dtype == object:
            data[col] = data[col].astype('category')
        else:
            pass

    for col in data.columns:
        if pd.CategoricalDtype.is_dtype(data[col]) == True:
            tb = tb.append(series_cat_summary(data[col]), ignore_index=True)

        else:
            tb = tb.append(series_num_summary(data[col], digits=digits), ignore_index=True)

    tb.index = range(tb.shape[0])

    if merge_stats == True:
        tb['Среднее и ст. откл.'] = tb['Среднее'].astype(str) + ' ± '+ tb['Ст.откл.'].astype(str)
        tb['Медиана и 25/75%'] = tb['Медиана'].astype(str) + ' ['+ tb['25%'].astype(str) + '; '+ tb['75%'].astype(str) + ']'
        tb = tb.drop(columns=['Среднее', 'Ст.откл.', 'Медиана', '25%', '75%'])
        tb['Доля, %'] = tb['Количество'].astype(str) + ' ('+ tb['Доля'].astype(str) + ' %)'


        tb = tb[['Фактор', 'Категории', 'Валидные', 'Доля, %', 'Мин', 'Среднее и ст. откл.','2.5%','Медиана и 25/75%', '97.5%', 'Макс', 'Значимость Ш-У']]

        tb['Среднее и ст. откл.'] = tb['Среднее и ст. откл.'].replace('- ± -', '-')
        tb['Медиана и 25/75%'] = tb['Медиана и 25/75%'].replace('- [-; -]', '-')
        tb['Доля, %'] = tb['Доля, %'].replace(['Более 5 (Более 5 %)', '- (- %)'],['Более 5','-'])

    else:
        pass

    return(tb)

## Compare 2 groups

def compare_category(df, group, var, auto_c='auto'):

    tb = pd.DataFrame()

    a = df[[var, group]].dropna()
    name = var
    mtx = pd.crosstab(a[var], a[group]).to_numpy()


    if auto_c == 'F':
        if mtx.shape[1] > 5:
            p = '-'
            test_name = 'Более 5 категорий'
        
        else:
            rw,col = mtx.shape
            p = round(np.array((stats.fisher_test(
                                                    base.matrix(mtx, nrow=rw, ncol=col)
                                                    ,simulate_p_value = True, B = 100)[0]))[0],3)
            test_name = 'Fisher'

    elif auto_c == 'Chi':
        p = round(chi2_contingency(mtx, correction=True)[1], 3)
        test_name = 'Chi'

    else:
        if np.any(mtx < 5):
            try:
                rw,col = mtx.shape
                p = np.array((stats.fisher_test(base.matrix(mtx, nrow=rw, ncol=col),simulate_p_value = True, B = 100)[0]))[0]
                test_name = 'Fisher'
            except rpy2.rinterface_lib.embedded.RRuntimeError:
                p = 1
                test_name = 'не применим'

        else:
            p = chi2_contingency(mtx, correction=True)[1]
            test_name = 'Chi'

    tb = pd.DataFrame({'Фактор': name,
                    'p, значимость': '{0:.3f}'.format(p),
                    'Критерий': test_name}, index=[0])
    return(tb)


def compare_numerical_2g(df, group, var, auto_n='mw'):

    a = pd.DataFrame()
    tb = pd.DataFrame()

    name = var
    a[group] = df[group].dropna()

    cat = pd.unique(a[group])

    a[var] = df[var].dropna()
    try:
        x = a[a[group] == cat[0]][var].dropna()
        y = a[a[group] == cat[1]][var].dropna()
    except IndexError:
        x = np.zeros(3)
        y = np.zeros(3)

    if auto_n=='mw':
        try:
            p = round(mannwhitneyu(x,y)[1],3)
            test_name = 'U-M-W'    
        except ValueError:
            p = 1
            test_name = 'U-M-W'
    
    elif len(pd.unique(x)) < 4:
        try:
            p = round(mannwhitneyu(x,y)[1],3)
            test_name = 'U-M-W'    
        except ValueError:
            p = 1
            test_name = 'U-M-W'

    elif len(pd.unique(y)) < 4:
        try:
            p = round(mannwhitneyu(x,y)[1],3)
            test_name = 'U-M-W'    
        except ValueError:
            p = 1
            test_name = 'U-M-W'
    
    else:
        if shapiro(x)[1] < 0.05:
            try:
                p = round(mannwhitneyu(x,y)[1],3)
                test_name = 'U-M-W'    
            except ValueError:
                p = 1
                test_name = 'U-M-W'

        elif shapiro(y)[1] < 0.05:
            try:
                p = round(mannwhitneyu(x,y)[1],3)
                test_name = 'U-M-W'    
            except ValueError:
                p = 1
                test_name = 'U-M-W'

        else:
            p = round(ttest_ind(x,y, equal_var = False)[1],3)
            test_name = 't-test'

    tb = pd.DataFrame({'Фактор': name,
                    'p, значимость': '{0:.3f}'.format(p),
                    'Критерий': test_name}, index=[0])
    return(tb)



def compare_df(df, group, auto_c='auto', auto_n='mw'):

    data = df

    tb = pd.DataFrame()

    for col in data.columns:
        if data[col].dtype == object:
            data[col] = data[col].astype('category')
        else:
            pass

    colls = [x for x in list(data.columns) if x != group]

    for col in colls:
        if pd.CategoricalDtype.is_dtype(data[col]) == True:
            tb = tb.append(compare_category(data, group=group, var=col, auto_c=auto_c))

        else:
            tb = tb.append(compare_numerical_2g(data, group=group, var=col, auto_n=auto_n))

    tb.index = range(tb.shape[0])

    return(tb)   


## Compare 3/4 groups

def compare_numerical_multig(df, group, var, n_groups=3, auto_n='krus'):

    a = pd.DataFrame()
    tb = pd.DataFrame()

    name = var

    a[group] = df[group]
    a[var] = df[var]

    a = a.dropna()

    cat = pd.unique(a[group])

    

    if len(cat) == 3:
    
        x = a[a[group] == cat[0]][var].to_numpy().ravel()
        y = a[a[group] == cat[1]][var].to_numpy().ravel()
        z = a[a[group] == cat[2]][var].to_numpy().ravel()

        if auto_n=='krus':
            try:
                p = round(kruskal(x,y,z, nan_policy='omit')[1],3)
                test_name = 'Kr-W'    
            except:
                print(var)
    
        else:

            p = round(f_oneway(x,y,z)[1],3)
            test_name = 'ANOVA'

        tb = pd.DataFrame({'Фактор': name,
                    'p, значимость': '{0:.3f}'.format(p),
                    'Критерий': test_name}, index=[0])
    
    elif len(cat) == 4:
    
        x = a[a[group] == cat[0]][var].to_numpy().ravel()
        y = a[a[group] == cat[1]][var].to_numpy().ravel()
        z = a[a[group] == cat[2]][var].to_numpy().ravel()
        q = a[a[group] == cat[3]][var].to_numpy().ravel()

        if auto_n=='krus':

            p = round(kruskal(x,y,z,q, nan_policy='omit')[1],3)
            test_name = 'Kr-W'    
    
        else:

            p = round(f_oneway(x,y,z,q)[1],3)
            test_name = 'ANOVA'

        tb = pd.DataFrame({'Фактор': name,
                    'p, значимость': '{0:.3f}'.format(p),
                    'Критерий': test_name}, index=[0])

    else:
        pass  

    return(tb)


def compare_multigroup(df, group, n_groups=3, auto_c='auto', auto_n='krus'):

    data = df

    tb = pd.DataFrame()

    for col in data.columns:
        if data[col].dtype == object:
            data[col] = data[col].astype('category')
        else:
            pass

    colls = [x for x in list(data.columns) if x != group]

    for col in colls:
        if pd.CategoricalDtype.is_dtype(data[col]) == True:
            tb = tb.append(compare_category(data, group=group, var=col, auto_c=auto_c))

        else:
            tb = tb.append(compare_numerical_multig(data, group=group, var=col, n_groups=n_groups, auto_n=auto_n))

    tb.index = range(tb.shape[0])

    return(tb)  


## 2 groups descriptives and compare combine

def funky_compare_2g(df, col_lst, group, 
    cat_1, cat_2, 
    name_1, name_2, 
    filename = None, 
    total_group = False, 
    store_only_comparison=False, 
    digits=2, 
    save_tab=True):

    """
    funky_compare_2g - сохранят описательные и сравнительные статистики по 2-м группам

    df - общая таблица,
    col_lst - список отобранных колонок с переменными, по которым будет проведено сравнение, вместе с группирующей переменной
    group - группирующая переменная
    cat_1, cat_2 - названия категорий группирующей переменной
    name_1, name_2 - названия подгрупп групирующей переменной, как они будут в таблицах и на листах
    filename - имя файла при сохранении
    total_group - использовать ли описательные статистики для всей группы (сливая подгруппы в одну)
    store_only_comparison - сохранять/выводить ли только таблицу сравнения
    digits - кол-во знаков после запятой для вещественных признаков
    save_tab - сохранять ли таблицу? Если не сохранять, будет выведена только сравнительная таблица в формате pandas Dataframe

    """

    s = summary_all(df[df[group]==cat_1][col_lst], merge_stats=True, digits=digits)
    d = summary_all(df[df[group]==cat_2][col_lst], merge_stats=True, digits=digits)

    compare_table = compare_df(df[col_lst], group=group, auto_c='auto', auto_n='auto')

    tab = s[['Фактор', 'Категории','Доля, %', 'Валидные', 'Медиана и 25/75%','Среднее и ст. откл.']].merge(d[['Фактор', 'Категории','Доля, %', 'Валидные', 'Медиана и 25/75%', 'Среднее и ст. откл.']], \
        on=['Фактор', 'Категории'], suffixes=('_' + name_1, '_' + name_2))

    tab = tab.merge(compare_table, on='Фактор')

    tab = tab[['Фактор', 'Категории', 'Валидные_' + name_1, 'Валидные_' + name_2, 'Доля, %_' + name_1, 'Доля, %_' + name_2, 'Медиана и 25/75%_' + name_1, 'Медиана и 25/75%_' + name_2, 'Среднее и ст. откл._' + name_1, 'Среднее и ст. откл._' + name_2,'p, значимость', 'Критерий']]

    if save_tab:

        writer = pd.ExcelWriter(filename, engine='xlsxwriter')

        if store_only_comparison:
            tab.to_excel(writer, sheet_name='Сравнение')

        else:

            if total_group:
                summary_all(df[col_lst]).to_excel(writer, sheet_name='Вся группа')
            else: pass
            
            s.to_excel(writer, sheet_name=name_1)
            d.to_excel(writer, sheet_name=name_2)

            tab.to_excel(writer, sheet_name='Сравнение')

        writer.save()

        return(print('Saved'))

    else:

        return(tab)

## 3 groups descriptives and compare combine with multiple comparison adjustment

def funky_compare_3g(
    df, 
    col_lst, 
    group, 
    group_set_lst,
    filename = None, 
    correct = True, 
    save_tab = True,
    digits=2
    ):

    data = df[col_lst].copy()
    
    data[group] = data[group].replace(group_set_lst, [1,2,3])
    print(str(group_set_lst) + ' = ' + str([1,2,3]))

    p_common_df = compare_multigroup(data, group) 

    tab12 = funky_compare_2g(data[data[group] != 3], 
        data.columns, 
        group, 
        1, 
        2, 
        '_1', 
        '_2', 
        filename = None, 
        total_group = False, 
        store_only_comparison=True, 
        digits=digits, 
        save_tab=False).rename(columns={'p, значимость': 'p_12'})

    tab23 = funky_compare_2g(data[data[group] != 1], 
        data.columns, 
        group, 
        2, 
        3, 
        '_2', 
        '_3', 
        filename = None, 
        total_group = False, 
        store_only_comparison=True, 
        digits=digits, 
        save_tab=False).rename(columns={'p, значимость': 'p_23'})

    tab13 = funky_compare_2g(data[data[group] != 2], 
        data.columns, 
        group, 
        1, 
        3, 
        '_1', 
        '_3', 
        filename = None, 
        total_group = False, 
        store_only_comparison=True, 
        digits=digits, 
        save_tab=False).rename(columns={'p, значимость': 'p_13'})

    tab = tab12.merge(tab23, on=['Фактор', 'Категории']).merge(tab13, on=['Фактор', 'Категории']).merge(p_common_df, on='Фактор')
    tab = tab[['Фактор', 'Категории', 'Валидные__1_x', 'Валидные__2_x', 'Валидные__3_x', 'Доля, %__1_x',
        'Доля, %__2_x','Доля, %__3_x', 'Медиана и 25/75%__1_x', 'Медиана и 25/75%__2_x', 'Медиана и 25/75%__3_x',
        'Среднее и ст. откл.__1_x', 'Среднее и ст. откл.__2_x', 'Среднее и ст. откл.__3_x', 'p, значимость', 'p_12','p_23', 'p_13','Критерий_y']]
    tab.columns = ['Фактор', 'Категории', 'Валидные__1', 'Валидные__2', 'Валидные__3', 'Доля, %__1',
        'Доля, %__2','Доля, %__3', 'Медиана и 25/75%__1', 'Медиана и 25/75%__2', 'Медиана и 25/75%__3',
        'Среднее и ст. откл.__1', 'Среднее и ст. откл.__2', 'Среднее и ст. откл.__3', 'p, значимость', 'p_12','p_23', 'p_13','post-hoc','Критерий']    

    if correct:
        for i in tab.index:
            row = tab.loc[i, ['p_12','p_23', 'p_13']]
            tab.loc[i, ['p_12','p_23', 'p_13']] = p_adjust(row, n = len(row))


    if save_tab:
        tab.to_excel(filename)
        print('Saved')
        
    else:
        return(tab)


"""
Dummification with NaN preserved
"""

def dummy_serie(df, col):
    tab = pd.get_dummies(df[col], prefix = col)
    tab.loc[df[col].isnull(), tab.columns.str.startswith(str(col))] = np.nan
    return(tab)

def dummification(df, cat_vars):
    data = df[cat_vars]
    tab = pd.DataFrame()
    for col in data:
        tab = pd.concat([dummy_serie(df, col), tab], axis = 1)
    
    tab = tab[tab.columns[::-1]]
    df =df.drop(columns = cat_vars)
    df = pd.concat([df, tab], axis = 1)
        
    return(df)


"""
95% CI for means, medians, proportions
"""

## Numerics

def numerics_95CI(df, num_vars):
    data = pd.DataFrame()
    for col in df[num_vars].columns:
        name = df[col].name
        A = to_array(df, col)
        test = shapiro(A)[1]
        if test < 0.05:
            B = np.zeros(10000)
        
            for i in range(0,10000):
                B[i] = np.median(np.random.choice(A, len(A)))
        
            way = 'BS'
            stat = 'Median'
            low = np.percentile(B, 2.5)
            high = np.percentile(B, 97.5)
    
        else:
            way = 'Conf.Int'
            stat = 'Mean'
            low = sms.DescrStatsW(A).tconfint_mean()[0]
            high = sms.DescrStatsW(A).tconfint_mean()[1]

        data = data.append({'Фактор': name, 'Способ': way, 'Статистика': stat,'2.5% CI': round(low,2), '97.5% CI': round(high, 2)}, ignore_index=True) #
        
    return(data.reindex(columns=['Фактор', 'Способ','Статистика','2.5% CI', '97.5% CI']))

## Proportions

def binary_95CI(df, cat_vars):
    data = pd.DataFrame()
    for col in df[cat_vars].columns:
        name = df[col].name
        A = to_array(df, col)
        CI = proportion_confint(np.sum(A), len(A))
        low = CI[0]
        high = CI[1]

        data = data.append({'Фактор': name, '2.5% CI': round(low*100, 1), '97.5% CI': round(high*100, 1)}, ignore_index=True) #
        
    return(data.reindex(columns=['Фактор', '2.5% CI', '97.5% CI']))

"""
Graphics
"""

## Simple KDE+boxplots for numerics

def dist_box(df, var, label = None, label_X = None, label_Y = 'Количество наблюдений'):
    sns.set(style = 'whitegrid')
    labels = label
    #fig, ax = plt.subplots(figsize = (8, 8))
    fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, figsize = (7, 7), gridspec_kw={"height_ratios": (.15, .85)})
    sns.boxplot(df[var], ax=ax_box, color = 'lightblue')
    sns.histplot(df[[var]], ax = ax_hist, color = 'blue', bins = 10, kde = True, label='_nolegend_')
    ax_hist.set(xlabel=label_X)
    ax_hist.get_legend().remove()
    plt.ylabel(label_Y)
    ax_box.set(xlabel='')
    plt.show()

## draw every variable without grouping (boxplots / barplots)

def draw_data_frame(df, col_lst):

    for col in col_lst:
        if pd.CategoricalDtype.is_dtype(df[col]) == True:
            if len(set(df[col].dropna())) < 6:

                B = round(df[col].dropna().value_counts() / df[col].dropna().shape[0]*100, 1).reset_index()
                B.columns=[col, 'Доля, %']

                sns.set(style='whitegrid')
                fig, ax = plt.subplots(figsize=(7,7))
                g = sns.barplot(data=B, x=col, y='Доля, %', ax=ax)
                for p in g.patches:
                    g.annotate(format(p.get_height(), '.1f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')
                plt.show()

            else: pass
        
        else:
            sns.set(style = 'whitegrid')
            fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, figsize = (7, 7), gridspec_kw={"height_ratios": (.15, .85)})
            sns.boxplot(df[col], ax=ax_box, color = 'lightblue')
            sns.histplot(df[[col]], ax = ax_hist, color = 'blue', bins = 10, kde = True, label='_nolegend_')
            ax_hist.set(xlabel=col)
            ax_hist.get_legend().remove()
            plt.ylabel('Количество наблюдений')
            ax_box.set(xlabel='')
            plt.show()

## draw every variable with grouping (boxplots / barplots)

def draw_data_frame_group(df, col_lst, group, pict_sav=True):

    for col in col_lst:
        if pd.CategoricalDtype.is_dtype(df[col]) == True:
            if len(set(df[col].dropna())) < 6:

                b = pd.crosstab(df[col], df[group], normalize='columns')

                b = b.rename(columns=str).reset_index().head()
                b = pd.melt(b, id_vars=col)
                b['value'] = b['value']*100

                sns.set(style='whitegrid')
                fig, ax = plt.subplots(figsize=(8,8))
                g = sns.barplot(data=b, x=group, y='value', hue=col, ax=ax)
                #ax.legend([],[], frameon=False)
                g.legend_.set_title(None)
                for p in g.patches:
                    g.annotate(format(p.get_height(), '.1f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')
                plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)## легенда снаружи
                plt.xlabel('')
                plt.ylabel(col)
                plt.show()

                if pict_sav:
                    g.figure.savefig(col  + '.png')

            else: pass
        
        else:
            sns.set(style='whitegrid')
            fig, ax = plt.subplots(figsize=(5,6))
            g=sns.boxplot(group, col, hue=group, data=df, ax=ax, dodge=False)
            ax.legend([],[], frameon=False)
            plt.ylabel(col)
            plt.xlabel('')
            plt.show()

            if pict_sav:
                g.figure.savefig(col  + '.png')


"""
Regression models
"""

## Logit model with group:baseline and group^time interaction

def logit_repeated_category(df, group, word, name, pict=True, pict_sav=False):

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



def mixed_repeated_numeric_2(df, group, word, name, transform='log', model='mixed', pict=True, pict_sav=False):
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
    
    if save:
        plt.savefig(plot_name + ".png")   



## Polar Plot Daniel Munblit

def polar_plot_munblit(
    df,
    cols,
    id_var,
    figsize=(8, 15),
    save=False,
    figname=None,
    title=''
    ):

    #show if any NaNs...
    df_na = df.copy()
    df_na = df_na.replace(1,0)
    df_na = df_na.fillna(1)

    # create filler...
    filler_df = df.copy()
    filler_df = filler_df.replace(0,1).fillna(1)

    df = df.fillna(0)
    # set figure size
    plt.figure(figsize=figsize)

    # plot polar axis
    ax = plt.subplot(111, polar=True)
    plt.axis('off')

    # Set the coordinates limits
    #upperLimit = 4
    lowerLimit = 2

    max = 1
    slope = (max - lowerLimit) / max
    coeff = (slope * 1)/25
    nstart = 0.2 
    cols = cols
    id = df[id_var]

    a = nstart
    n = []
    for i in range(len(cols)):
        a = a + abs(coeff)
        n = n + [a]

    ##########################################
    ##### Filler

    for col, bot, fill_color in zip(cols, n, ['No symptoms'] + [None]*(len(cols) - 1)):

        heights = (slope * filler_df[col])/25

        # Compute the width of each bar. In total we have 2*Pi = 360°
        width = 2*np.pi / len(filler_df.index)

        # Compute the angle each bar is centered on:
        indexes = list(range(1, len(filler_df.index)+1))
        angles = [element * width for element in indexes]

        # Draw bars
        bars = ax.bar(
            x=angles, 
            height=heights, 
            width=width, 
            bottom=bot,
            linewidth=2, 
            color = '#dedede',
            edgecolor="white", 
            alpha=0.5, label=fill_color)
        

    #### Plot ###############################

    for col, bot, color in zip(cols, n, ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', 'lightgreen',
              '#bcbd22', '#17becf']):

        heights = (slope * df[col])/25

        # Compute the width of each bar. In total we have 2*Pi = 360°
        width = 2*np.pi / len(df.index)

        # Compute the angle each bar is centered on:
        indexes = list(range(1, len(df.index)+1))
        angles = [element * width for element in indexes]

        # Draw bars
        bars = ax.bar(
            x=angles, 
            height=heights, 
            width=width, 
            bottom=bot,
            linewidth=2, 
            color = color,
            edgecolor="white",
            label=col)
        #ax.legend()

    ##### NANS #######################################

    for col, bot, labelz in zip(cols, n, ['Missing data'] + [None]*(len(cols) - 1)):

        heights = (slope * df_na[col])/25

        # Compute the width of each bar. In total we have 2*Pi = 360°
        width = 2*np.pi / len(df_na.index)

        # Compute the angle each bar is centered on:
        indexes = list(range(1, len(df.index)+1))
        angles = [element * width for element in indexes]

        # Draw bars
        bars = ax.bar(
            x=angles, 
            height=heights, 
            width=width, 
            bottom=bot,
            linewidth=2, 
            color = 'darkgray',
            edgecolor="white",
            label=labelz
            )
        ax.legend(bbox_to_anchor=(1.7, .5), loc='center right')


    ###
        # Add labels
    for bar, angle, height, label in zip(bars,angles, heights, id):

        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle)

        # Flip some labels upside down
        alignment = ""
        if angle >= np.pi/2 and angle < 3*np.pi/2:
            alignment = "right"
            rotation = rotation + 180
        else: 
            alignment = "left"

        # Finally add the labels
        ax.text(
            x=angle, 
            y=n[-1] + abs(coeff), 
            s=label, 
            ha=alignment, 
            va='center', 
            rotation=rotation, 
            rotation_mode="anchor")
    
    ax = plt.gca()
    ax.set_facecolor('xkcd:white')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.title(title, y = 1.15, fontweight='bold')
    plt.tight_layout()


    if save:
        plt.savefig(figname + ".png")






