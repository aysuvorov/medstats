
# Core Libraries
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge

# Scientific Computing and Statistics
from scipy import stats
from scipy.stats import (
    norm, t as t_dist, nct, ttest_ind, shapiro, kstest, mannwhitneyu,
    fisher_exact, chi2_contingency, kruskal, wilcoxon, f_oneway
)
from pandas.api.types import CategoricalDtype
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import proportion_confint
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Machine Learning Preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch

# Optimization (Root Finding)
from scipy.optimize import brentq

# Plotting Enhancements
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

# Unicode Handling
from unicodedata import normalize

# Missing Values Visualization
import missingno as msno

# Itertools Combinatorial Generators
from itertools import product, combinations

# R Integration via rpy2
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, FloatVector, IntVector, FactorVector, Formula
# rpyn.activate()
# stats_r = importr('stats')
# base = importr('base')

# Warning Management
import warnings

# Miscellaneous Mathematical Operations
from math import ceil


# +-----------------------------------------------------------------------------
# Constants

RS = 1000

# +-----------------------------------------------------------------------------
# +-----------------------------------------------------------------------------

################################################################################
# Data cleaners and mess organizers
################################################################################

# def columnn_normalizer(df, col_lst):
#     """
#     Removing crazy separators in columns
#     """
#     for col in col_lst:
#         for i in range(len(df[col])):
#             try:
#                 df[col][i] = normalize('NFKC', df[col][i])
#                 df[col][i] = df[col][i].replace(',','.')
#                 try:
#                     df[col][i] = float(df[col][i])
#                 except ValueError:
#                     pass
#             except TypeError:
#                 pass
#         try:
#             df[col] = df[col].astype(float)
#         except:
#             pass

#     return(df)

def _norm_string_array(arr):
    """
    Vectorised unicode NFKC normalisation + comma→dot replacement.
    `arr` is a NumPy array of dtype object that contains strings.
    """
    # unicode normalisation (vectorised via np.frompyfunc)
    ufunc_norm = np.frompyfunc(lambda x: normalize('NFKC', x), 1, 1)
    arr = ufunc_norm(arr)

    # replace ',' by '.'   (also vectorised)
    ufunc_rep  = np.frompyfunc(lambda x: x.replace(',', '.'), 1, 1)
    arr = ufunc_rep(arr)

    return arr


def column_normalizer(df, col_lst):
    """
    Normalise the text in `col_lst` and cast to float where possible.
    Modifies the DataFrame in place and also returns it.
    """
    for col in col_lst:
        # --- a) ensure we’re working with strings (object dtype) ----------
        s = df[col].astype(str)

        # --- b) vectorised unicode normalisation + replacement -----------
        s = pd.Series(_norm_string_array(s.values), index=s.index)

        # --- c) convert to numeric if possible ---------------------------
        df[col] = pd.to_numeric(s, errors='ignore')

    return df

################################################################################
# Data frame glimpse and missings analysis
################################################################################

def glimpse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes a DataFrame and creates a summary report for each column.

    Arguments:
        df (pandas.DataFrame): The input DataFrame to analyze.

    Returns:
        pandas.DataFrame: A DataFrame containing analysis results with the following columns:
            - Column: Name of the column.
            - Type: Data type of the column.
            - Valid: Number of non-missing values.
            - Missings: Number and percentage of missing values.
            - Missing category: Category of missingness level (Full, Low, Moderate, High).
            - Uniques: Number of unique values in the column.
    """
    # Create an empty DataFrame to store the results
    result_df = pd.DataFrame(columns=['Column', 'Type', 'Valid', 'Missings', 'Missing category', 'Uniques'])

    for col in df.columns:
        series = df[col]

        # Determine the data type
        dtype = str(series.dtype)

        # Count the number of missing values
        na_count = series.isna().sum()
        valid_count = len(series) - na_count
        total_rows = len(series)
        na_percentage = round(na_count / total_rows * 100, 2)

        # Count the number of unique values
        unique_values = series.nunique()

        # Determine the missingness category
        missing_category = ''
        if na_percentage == 0:
            missing_category = 'Full'
        elif na_percentage <= 10:
            missing_category = 'Low'
        elif na_percentage <= 15:
            missing_category = 'Moderate'
        else:
            missing_category = 'High'

        # Add a row to the resulting DataFrame
        new_row = pd.Series({
            'Column': col,
            'Type': dtype,
            'Valid': valid_count,
            'Missings': f"{na_count} ({na_percentage}%)",  # Format the string with percentages
            'Missing category': missing_category,
            'Uniques': unique_values
        })
        result_df = pd.concat([result_df, new_row.to_frame().T], ignore_index=True)

    return result_df


def plot_missing_by_combos(df: pd.DataFrame,
                           cat_cols: list[str] | None = None,
                           *,
                           valid_color: str = "#BDBDBD",
                           na_color: str = "#d62728",
                           figsize=(9, 4),
                           title_prefix="Missing map – ",
                           max_rows_show: int | None = 200,
                           seed: int | None = 0):
    """
    Рисует отдельный heat-map пропусков для каждой уникальной
    комбинации значений в `cat_cols`
    (если `cat_cols=None` → один график для всего DF).
    """
    cat_cols = [c for c in (cat_cols or []) if c in df.columns]
    cmap = ListedColormap([valid_color, na_color])

    # ── строим словарь {комбинация: подтаблица} ---------------------------
    if cat_cols:
        # Альтернативный подход к группировке
        combo_series = df[cat_cols].astype(str).agg('/'.join, axis=1)
        groups = {}
        for combo in combo_series.unique():
            mask = combo_series == combo
            groups[combo] = df.loc[mask].copy()
    else:
        groups = {'ALL': df.copy()}

    for combo_id, block in groups.items():
        # ▸ случайное подвыборка, если нужно
        if max_rows_show and len(block) > max_rows_show:
            block = block.sample(max_rows_show, random_state=seed, replace=False)

        # Убедимся, что у нас есть данные для отображения
        if len(block) == 0:
            print(f"Пустая группа: {combo_id}")
            continue

        # Создаем матрицу пропусков (1 = пропуск, 0 = значение есть)
        miss = block.isna().astype(int)

        # Проверяем, есть ли пропуски в этой группе
        total_missing = miss.sum().sum()
        if total_missing == 0:
            print(f"Пропусков нет в группе: {combo_id} (n={len(block)})")
            continue

        # ▸ подпись комбинации
        if combo_id == 'ALL':
            combo_name = "ALL"
        else:
            combo_name = combo_id.replace('/', ', ')

        # Создаем фигуру
        fig, ax = plt.subplots(figsize=figsize)

        # Рисуем heatmap с явными параметрами
        sns.heatmap(
            miss,
            cmap=cmap,
            vmin=0,
            vmax=1,
            linewidths=0.2,
            linecolor="lightgray",
            cbar=False,
            yticklabels=False,
            xticklabels=True,
            ax=ax,
            square=False  # Важно: не делать квадратные ячейки
        )

        # Настройки осей
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows")
        ax.set_title(f"{title_prefix}{combo_name} (n={len(block)}, пропусков: {total_missing})")

        # Добавляем сетку для лучшей видимости
        ax.grid(False)

        plt.tight_layout()
        plt.show()

        # Отладочная информация
        print(f"Группа: {combo_name}")
        print(f"Размер: {len(block)} строк")
        print(f"Всего пропусков: {total_missing}")
        print("Пропуски по столбцам:")
        missing_by_col = block.isna().sum()
        for col, count in missing_by_col.items():
            if count > 0:
                print(f"  {col}: {count} пропусков")
        print("-" * 50)


# Альтернативная упрощенная версия для отладки
def plot_missing_by_combos_simple(df: pd.DataFrame,
                                  cat_cols: list[str] | None = None,
                                  figsize=(9, 4)):
    """
    Упрощенная версия для отладки
    """
    cat_cols = [c for c in (cat_cols or []) if c in df.columns]

    if cat_cols:
        # Группируем по категориальным переменным
        for combo, group in df.groupby(cat_cols, dropna=False):
            print(f"\nГруппа: {combo}")
            print(f"Размер: {len(group)} строк")

            # Создаем матрицу пропусков
            miss = group.isna().astype(int)
            total_missing = miss.sum().sum()

            if total_missing > 0:
                fig, ax = plt.subplots(figsize=figsize)
                sns.heatmap(miss, cmap=['#BDBDBD', '#d62728'],
                           cbar=False, yticklabels=False)
                ax.set_title(f"Group: {combo} (missing: {total_missing})")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

                print("Пропуски по столбцам:")
                print(group.isna().sum())
            else:
                print("Пропусков нет")
    else:
        # Без группировки
        miss = df.isna().astype(int)
        total_missing = miss.sum().sum()

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(miss, cmap=['#BDBDBD', '#d62728'],
                   cbar=False, yticklabels=False)
        ax.set_title(f"All data (missing: {total_missing})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print("Пропуски по столбцам:")
        print(df.isna().sum())


# +-----------------------------------------------------------------------------
# Imputation

def smart_imputer(df: pd.DataFrame,
                  *,
                  thresh: float = .10,
                  random_state: int | None = 0) -> pd.DataFrame:
    """
    Импутирует пропуски только там, где их ≤ `thresh`
      • числовые колонки  – IterativeImputer (BayesianRidge)
      • категориальные    – SimpleImputer(most_frequent)

    Parameters
    ----------
    df            : исходный DataFrame
    thresh        : порог доли NaN (0.10 = 10 %)
    random_state  : фиксирует случайность в IterativeImputer

    Returns
    -------
    новый DataFrame (оригинал не меняется)
    """
    out = df.copy()
    miss_share = out.isna().mean()

    # -------- числовые --------------------------------------------------
    num_cols = out.select_dtypes(include="number").columns
    num_targets = [c for c in num_cols if 0 < miss_share[c] <= thresh]

    if num_targets:
        # IterativeImputer требует ≥2 признаков; если 1 → MeanImputer
        if len(num_targets) == 1:
            imp_num = SimpleImputer(strategy="mean")
        else:
            imp_num = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=10,
                random_state=random_state
            )
        out[num_targets] = imp_num.fit_transform(out[num_targets])

    # -------- категориальные -------------------------------------------
    cat_cols = out.select_dtypes(exclude="number").columns
    cat_targets = [c for c in cat_cols if 0 < miss_share[c] <= thresh]

    if cat_targets:
        imp_cat = SimpleImputer(strategy="most_frequent")
        out[cat_targets] = imp_cat.fit_transform(out[cat_targets])
    return out


# +-----------------------------------------------------------------------------
# Data transformers

def factor_transformer(data_frame: pd.DataFrame,
                       min_factor_levels: int = 7) -> pd.DataFrame:
    """
    Convert every column that has ≤ `min_factor_levels` distinct non-NA values
    to pandas 'category' type (using pd.Categorical) and return the new frame.

    Parameters
    ----------
    data_frame : pd.DataFrame
        Source data.
    min_factor_levels : int, default 7
        Upper threshold for the number of distinct levels
        at or below which a column becomes categorical.

    Returns
    -------
    pd.DataFrame
        A copy of `data_frame` with selected columns converted
        to categorical dtype.
    """
    # 1. Count distinct non-missing values per column
    unique_counts = data_frame.nunique(dropna=True)

    # 2. Columns that meet the threshold
    cols_to_convert = unique_counts[unique_counts <= min_factor_levels].index

    # 3. Build and return a copy with those columns cast to category
    return data_frame.assign(
        **{col: pd.Categorical(data_frame[col]) for col in cols_to_convert}
    )


# def miss_counter(data):

#     missing_df = pd.DataFrame(data.isnull().sum())
#     missing_df.columns = ['Miss_abs_counts']
#     missing_df['Valid_abs_counts'] = data.shape[0] - \
#         missing_df['Miss_abs_counts']
#     missing_df['Miss_Rates,%'] = missing_df['Miss_abs_counts']/data.shape[0]
#     missing_df['Valid_Rates,%'] = missing_df['Valid_abs_counts']/data.shape[0]
#     return(missing_df[['Valid_abs_counts', 'Valid_Rates,%', 'Miss_abs_counts',
#         'Miss_Rates,%']])


# def p_adjust(vector, n, method = 'BH'):

#     vector = FloatVector(np.asarray(vector))
#     new_vec = []
#     for i in vector:
#         new_vec = new_vec + [float(stats.p_adjust(i, n=n, method=method))]

#     return new_vec

# +-----------------------------------------------------------------------------
# Dummification

def dummification(df, cat_vars):

    def dummy_serie(df, col):
        tab = pd.get_dummies(df[col], prefix = col)
        tab.loc[df[col].isnull(), tab.columns.str.startswith(str(col))] = np.nan
        for col in tab:
            tab[col] = tab[col].astype('category')
        return(tab)

    data = df[cat_vars]
    tab = pd.DataFrame()
    for col in data:
        tab = pd.concat([dummy_serie(df, col), tab], axis = 1)

    tab = tab[tab.columns[::-1]]
    df =df.drop(columns = cat_vars)
    df = pd.concat([df, tab], axis = 1)

    return(df)


################################################################################
# Descriptive statistics
################################################################################

## Simple descriptives

def _table_shapiro(df, digits):
    """Return p-values of Shapiro–Wilk for every numeric column."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    pvals = {
        col: (shapiro(df[col].dropna())[1]          # p-value
              if 3 <= df[col].notna().sum() <= 5000 else np.nan)
        for col in num_cols
    }
    return (
        pd.Series(pvals, name="Тест Ш-У, значимость")
        .round(digits)
        .reset_index()
        .rename(columns={"index": "Фактор"})
    )


def _fmt_mean_sd(series, d):      # mean ± sd
    return f"{series.mean():.{d}f} ± {series.std(ddof=1):.{d}f}" if series.size else "NA"


def _fmt_median_iqr(series, d):   # median [p25; p75]
    if series.size == 0:
        return "NA"
    q25, q75 = np.percentile(series, [25, 75])
    return f"{series.median():.{d}f} [{q25:.{d}f}; {q75:.{d}f}]"


def _fmt_min(series, d):          # min with NA fallback
    return f"{series.min():.{d}f}" if series.size else "NA"


def _fmt_max(series, d):          # max with NA fallback
    return f"{series.max():.{d}f}" if series.size else "NA"


def _fmt_n_pct(n, total, d):
    pct = n / total * 100 if total else 0
    return f"{n} ({pct:.{d}f}%)"


def summary_all(data: pd.DataFrame, digits: int = 1) -> pd.DataFrame:
    """Mimic the R `summary_all()` with per-level rows for categoricals."""
    # 1. Normality test on numeric columns
    shapiro_df = _table_shapiro(data, 3)

    rows = []

    for col in data.columns:
        s = data[col].dropna()
        total_n = s.size
        is_num = pd.api.types.is_numeric_dtype(s)

        # -------- NUMERIC --------------------------------------------------
        if is_num:
            rows.append(
                {
                    "Фактор": col,
                    "Валидные,N": total_n,
                    "Абс,доля,%": "-",
                    "Среднее, ст.откл": _fmt_mean_sd(s, digits),
                    "Медиана и размахи": _fmt_median_iqr(s, digits),
                    "Мин": _fmt_min(s, digits),
                    "Макс": _fmt_max(s, digits),
                }
            )
        # -------- CATEGORICAL ----------------------------------------------
        else:
            # header-like line for the variable itself
            rows.append(
                {
                    "Фактор": col,
                    "Валидные,N": "NA",
                    "Абс,доля,%": "NA",
                    "Среднее, ст.откл": "NA",
                    "Медиана и размахи": "NA",
                    "Мин": "NA",
                    "Макс": "NA",
                }
            )

            # one row per category
            counts = s.value_counts(dropna=False).sort_index()
            for level, n in counts.items():
                level_name = str(level)
                rows.append(
                    {
                        "Фактор": level_name,
                        "Валидные,N": total_n,
                        "Абс,доля,%": _fmt_n_pct(n, total_n, digits),
                        "Среднее, ст.откл": "-",
                        "Медиана и размахи": "-",
                        "Мин": "-",
                        "Макс": "-",
                    }
                )

    # 2. Build DataFrame
    df = pd.DataFrame(rows)

    # 3. Merge with Shapiro p-values → only numeric rows keep the number
    df = df.merge(shapiro_df, on="Фактор", how="left")

    # 4. Add running index & reorder columns
    df.insert(0, "Индекс", range(1, len(df) + 1))
    final_cols = [
        "Индекс",
        "Фактор",
        "Валидные,N",
        "Абс,доля,%",
        "Среднее, ст.откл",
        "Медиана и размахи",
        "Мин",
        "Макс",
        "Тест Ш-У, значимость",
    ]
    return df[final_cols]


def _r_fisher_test(cont: pd.DataFrame, simulate: bool, seed: int):
    """Run Fisher's exact test in R and return p-value."""
    from rpy2.robjects.packages import importr
    from rpy2.robjects import default_converter

    converter = default_converter + pandas2ri.converter

    with converter.context():
        stats_r = importr('stats')
        r_matrix = pandas2ri.py2rpy(cont)

        if simulate:
            ro.r(f'set.seed({seed})')
            result = stats_r.fisher_test(
                r_matrix,
                simulate_p_value=True,
                B=10000
            )
        else:
            result = stats_r.fisher_test(r_matrix)

        # result is an OrderedDict with key 'p.value'
        return float(result['p.value'][0])


def fisher_exact_r(cont: pd.DataFrame, seed=1000) -> tuple[str, float | None]:
    """
    Run Fisher's exact test via R for ANY sized contingency table with sparse cells.
    """
    np.random.seed(seed)
    try:
        if cont.shape[0] == 2:
            p = _r_fisher_test(cont, simulate=False, seed=seed)
            return "Точный тест Фишера (R)", p
        p = _r_fisher_test(cont, simulate=True, seed=seed)
        return "Точный тест Фишера (R, симуляция)", p
    except Exception:
        # Fallback to chi-square
        try:
            chi2, p_val, *_ = sp_stats.chi2_contingency(cont, correction=True)
            return "Χ²-тест", float(p_val)
        except:
            return "Ошибка", None


def _shapiro_ok(x: np.ndarray) -> bool:
    """TRUE ⇢ p > .05 (looks Normal) – reproduces R part."""
    x = x[~np.isnan(x)]
    if x.size < 3 or x.size > 5000 or np.unique(x).size < 3:
        return False
    try:
        return stats.shapiro(x).pvalue > 0.05
    except Exception:
        return False


def compare_all(
    df: pd.DataFrame,
    group_var: str,
    digits: int = 1,
    add_minmax: bool = False,
) -> pd.DataFrame:
    """
    Python re-implementation of the R function `compare_all`.

    Parameters
    ----------
    df         : pd.DataFrame
    group_var  : str              – name of the grouping column
    digits     : int, default 1   – decimals in formatted output
    add_minmax : bool, default False
                                   – include 'min-max' row for numerics

    Returns
    -------
    pd.DataFrame – summary table (ready for copy/paste to docs)
    """
    # ------------------------------------------------------------ #
    # 0. Preparation                                               #
    # ------------------------------------------------------------ #
    wk = df.copy(deep=True)   # never touch the caller's frame!

    # keep the original order of groups (factor levels first, else order of appearance)
    if pd.api.types.is_categorical_dtype(wk[group_var]):
        groups = list(wk[group_var].cat.categories)
    else:
        groups = list(pd.unique(wk[group_var]))
    n_groups = len(groups)

    # temporary fix: columns starting with a digit -> prepend 'firstnum_'
    newcols = {
        col: f"firstnum_{col}" if re.match(r"^\d", str(col)) else col
        for col in wk.columns
    }
    wk.rename(columns=newcols, inplace=True)

    # mapping to restore names only for the 'Фактор' column later
    reverse_fix = {v: k for k, v in newcols.items()}

    # helper format strings
    fmt_mean_sd = lambda m, s: f"{m:.{digits}f} ± {s:.{digits}f}"
    fmt_med_iqr = lambda d, q1, q3: f"{d:.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"
    fmt_minmax = lambda mn, mx: f"({mn:.{digits}f}; {mx:.{digits}f})"
    fmt_pval = lambda p: f"{p:.3f}" if p is not None and not np.isnan(p) else ""

    rows = []
    row_id = 1

    # ------------------------------------------------------------ #
    # 1. Iterate over every variable except the group var          #
    # ------------------------------------------------------------ #
    for var in [c for c in wk.columns if c != group_var]:

        col_ser = wk[var]
        # ---------- NUMERIC ------------------------------------------------
        if pd.api.types.is_numeric_dtype(col_ser):

            desc = (
                wk.groupby(group_var)[var]
                .agg(
                    mean="mean",
                    sd="std",
                    median="median",
                    q1=lambda x: x.quantile(0.25),
                    q3=lambda x: x.quantile(0.75),
                    mn="min",
                    mx="max",
                    N=lambda x: x.notna().sum(),
                )
                .reindex(groups)
            )
            desc["N"] = desc["N"].fillna(0).astype(int)

            # global normality decision
            is_normal = all(_shapiro_ok(wk.loc[wk[group_var] == g, var].to_numpy())
                            for g in groups)

            # choose the right inferential test
            p_val = None
            test_name = ""
            # all groups must contain at least 2 distinct values
            valid_groups = [
                wk.loc[wk[group_var] == g, var].dropna().nunique() > 1 for g in groups
            ]

            if not all(valid_groups):
                test_name = "Недостаточно данных (нулевая дисперсия в группе)"
            else:
                try:
                    if n_groups == 2:
                        g1, g2 = [wk.loc[wk[group_var] == g, var].dropna() for g in groups]
                        if is_normal:
                            p_val = stats.ttest_ind(g1, g2, equal_var=False).pvalue
                            test_name = "t-тест Стьюдента"
                        else:
                            p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided").pvalue
                            test_name = "Тест Вилкоксона"
                    else:  # > 2 groups
                        arrays = [wk.loc[wk[group_var] == g, var].dropna() for g in groups]
                        if any(len(a) <= 1 for a in arrays):
                            test_name = "Невозможно выполнить тест: недостаточно наблюдений"
                        elif is_normal:
                            p_val = stats.f_oneway(*arrays).pvalue
                            test_name = "ANOVA"
                        else:
                            p_val = stats.kruskal(*arrays).pvalue
                            test_name = "Тест Краскела-Уоллиса"
                except Exception:
                    test_name = "Ошибка при выполнении теста"
                    p_val = None

            # ---------- assemble rows ----------------------------------
            stat_rows = [
                ("среднее ± СО",
                 [fmt_mean_sd(desc.loc[g, 'mean'], desc.loc[g, 'sd']) for g in groups]),
                ("медиана [25%; 75%]",
                 [fmt_med_iqr(desc.loc[g, 'median'], desc.loc[g, 'q1'], desc.loc[g, 'q3'])
                  for g in groups]),
            ]
            if add_minmax:
                stat_rows.append(
                    ("Мин - Макс",
                     [fmt_minmax(desc.loc[g, 'mn'], desc.loc[g, 'mx']) for g in groups])
                )

            for ix, (stat_name, stat_values) in enumerate(stat_rows):
                row = {
                    "id": row_id,
                    "Фактор": var if ix == 0 else "",
                    "Статистика": stat_name,
                    **{g: stat_values[i] for i, g in enumerate(groups)},
                    **{f"n{i+1}": desc.loc[groups[i], "N"] for i in range(n_groups)},
                    "test_used": test_name if ix == 0 else "",
                    "p_value": fmt_pval(p_val) if ix == 0 else "",
                }
                rows.append(row)
                row_id += 1

        # ---------- CATEGORICAL (factor) -----------------------------------
        else:
            levels = pd.unique(col_ser.dropna())
            cont = pd.crosstab(wk[var], wk[group_var], dropna=True) \
                    .reindex(index=levels, columns=groups, fill_value=0)

            # ------- choose the inferential test -------------------------
            if cont.shape[0] < 2 or cont.shape[1] < 2:
                test_name, p_val = "Недостаточно данных для теста", None

            # Any table with sparse cells - use Fisher's test via R
            elif (cont.values < 5).any():
                test_name, p_val = fisher_exact_r(cont)

            else:
                # Well-populated table - use regular chi-square
                chi2, p_val, *_ = stats.chi2_contingency(cont)
                test_name = "Χ²-тест"

            # group-wise denominators
            valid_n = (wk.groupby(group_var)[var]
                        .count()
                        .reindex(groups)
                        .fillna(0)         # NEW ▶ пропуски → 0
                        .astype(int))      # затем int

            # -------- one output row per category level ------------------
            for j, level in enumerate(levels):
                counts = cont.loc[level]
                row = {
                    "id": row_id,
                    "Фактор": var if j == 0 else "",
                    "Статистика": level,
                    **{
                        g: f"{counts[g]} "
                           f"({(counts[g]/valid_n[g]*100 if valid_n[g] else 0):.{digits}f}%)"
                        for g in groups
                    },
                    **{f"n{i+1}": valid_n[groups[i]] for i in range(n_groups)},
                    "test_used": test_name if j == 0 else "",
                    "p_value": (f"{p_val:.3f}" if j == 0 and p_val is not None else "")
                }
                rows.append(row)
                row_id += 1

    # ------------------------------------------------------------ #
    # 2. Wrap-up                                                  #
    # ------------------------------------------------------------ #
    out = pd.DataFrame(rows)

    # Final column order
    out = out[
        ["id", "Фактор", "Статистика", *groups,
         *[f"n{i+1}" for i in range(n_groups)], "test_used", "p_value"]
    ]

    # polish column headers
    out.rename(columns={
        "test_used": "Статистический тест",
        "p_value": "Значимость, р"
    }, inplace=True)

    # restore original variable names (remove the temporary prefix)
    out["Фактор"] = out["Фактор"].replace(reverse_fix)

    return out



def _pair_names(groups: List[str]) -> List[str]:
    """Return ``['g1 - g2', 'g1 - g3', …]`` in the order of combinations."""
    return [f"{g1} - {g2}" for g1, g2 in combinations(groups, 2)]


def pairwise_comparisons(
        data: pd.DataFrame,
        group_var: str,
        p_adjust_method: str = "none"
) -> pd.DataFrame:
    """
    Perform pairwise comparisons (numeric & categorical) for every variable
    against a multi-level grouping factor.

    Parameters
    ----------
    data : pd.DataFrame
    group_var : str
        Column name that defines the groups (must contain >2 levels).
    p_adjust_method : str
        'none' (default) or any method accepted by
        statsmodels.stats.multitest.multipletests
        ('bonferroni', 'holm', 'fdr_bh', …).

    Returns
    -------
    pd.DataFrame
        One row per analysed variable, columns = every pair of groups.
        Cells hold raw or adjusted p-values (NaN if test not applicable).
    """
    # 0. sanity ----------------------------------------------------------------

    if group_var not in data.columns:
        raise KeyError(f"'{group_var}' not found in DataFrame")

    groups = data[group_var].dropna().unique()
    if len(groups) <= 2:
        raise ValueError("Grouping variable must have more than 2 levels")

    group_pairs = list(combinations(groups, 2))
    pair_cols = _pair_names(groups)

    # 1. iterate over variables -----------------------------------------------
    out_rows = []
    idx = 1

    for var in [c for c in data.columns if c != group_var]:

        col = data[var]
        if col.isna().all():
            continue                                                   # skip N/A only

        p_raw = [np.nan] * len(pair_cols)                              # placeholder

        # -------- numeric -----------------------------------------------------
        if pd.api.types.is_numeric_dtype(col):
            # single Shapiro on all non-NA values (same as original R helper)
            try:
                vals = col.dropna()
                normal = len(vals) >= 3 and len(vals) <= 5000 \
                         and stats.shapiro(vals).pvalue > .05
            except Exception:
                normal = False

            for ix, (g1, g2) in enumerate(group_pairs):
                x, y = (data.loc[data[group_var] == g, var].dropna() for g in (g1, g2))
                if len(x) < 2 or len(y) < 2:
                    continue

                try:
                    if normal:
                        p = stats.ttest_ind(x, y, equal_var=False).pvalue
                    else:
                        p = stats.mannwhitneyu(x, y, alternative="two-sided",
                              method="asymptotic").pvalue
                    p_raw[ix] = float(p)
                except Exception as e:
                    warnings.warn(f"{var}: test failed for {g1}-{g2} → {e}")

        # -------- categorical -------------------------------------------------
        else:
            # ensure categorical dtype (remove future-warning)
            if not isinstance(col.dtype, CategoricalDtype):
                col = col.astype("category")

            for ix, (g1, g2) in enumerate(group_pairs):
                sub = data[data[group_var].isin([g1, g2])]
                table = pd.crosstab(sub[var], sub[group_var])

                if table.shape[0] < 2 or table.shape[1] < 2:
                    continue

                try:
                    if (table.values < 5).any():
                        _, p = fisher_exact_r(table)                   # tuple(name, p)
                    else:
                        p = stats.chi2_contingency(table)[1]
                    p_raw[ix] = float(p)
                except Exception as e:
                    warnings.warn(f"{var}: categorical test failed for {g1}-{g2} → {e}")

        # -------- assemble row -----------------------------------------------
        if not np.isnan(p_raw).all():
            row = {"id": idx, "Фактор": var, **dict(zip(pair_cols, p_raw))}

            # adjust if requested
            if p_adjust_method.lower() != "none":
                valid_idx = [i for i, p in enumerate(p_raw) if not np.isnan(p)]
                if valid_idx:
                    adj = multipletests(
                        [p_raw[i] for i in valid_idx],
                        method=p_adjust_method.lower()
                    )[1]
                    for i, adj_p in zip(valid_idx, adj):
                        row[pair_cols[i]] = adj_p

            out_rows.append(row)
            idx += 1


    return pd.DataFrame(out_rows).round(3)


# ------------------------------------------------------------------------------
# 95% CI for means, medians, proportions

## Numerics

def numerics_95CI(df, num_vars, statistic='automatic'):
    """Calculates the 95% confidence interval for numerical variables in a DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing numerical variables.
        num_vars (list): List of names of numerical variables for which to calculate confidence intervals.
        statistic (str, optional): Method for calculating statistics. Can be either 'automatic' or 'mean'.
            Defaults to 'automatic'. If 'automatic', normal distribution uses mean, otherwise median.

    Returns:
        pandas.DataFrame: A DataFrame with the results including factor, calculation method, statistic,
            point estimate, lower bound of the confidence interval (2.5%), and upper bound of the confidence interval (97.5%).

    Notes:
        - If the parameter statistic is set to 'automatic', the Shapiro-Wilk test is performed to determine normality.
          If the distribution is normal, the mean is used; otherwise, the median is used.
        - If the parameter statistic is set to 'mean', the mean is always used.
        - If the parameter statistic is neither 'automatic' nor 'mean', an error message is displayed.
    """
    data = pd.DataFrame()
    for col in df[num_vars].columns:
        name = df[col].name
        A = np.asarray(df[col].dropna())

        if statistic == 'automatic':
            test = shapiro(A)[1]
            if test < 0.05:
                B = np.zeros(1000)

                for i in range(0, 1000):
                    B[i] = np.median(np.random.choice(A, len(A)))

                way = 'BS'
                stat = 'Median'
                point = np.median(A)
                low = np.percentile(B, 2.5)
                high = np.percentile(B, 97.5)

            else:
                way = 'Conf.Int'
                stat = 'Mean'
                point = np.mean(A)
                low = sms.DescrStatsW(A).tconfint_mean()[0]
                high = sms.DescrStatsW(A).tconfint_mean()[1]

        elif statistic == 'mean':
            way = 'Conf.Int'
            stat = 'Mean'
            point = np.mean(A)
            low = sms.DescrStatsW(A).tconfint_mean()[0]
            high = sms.DescrStatsW(A).tconfint_mean()[1]

        else:
            print("Statistic must be 'automatic' or 'mean'")

        data = data.append(
            {
                'Factor': name,
                'Method': way,
                'Statistic': stat,
                'Point Estimate': point,
                '2.5% CI': round(low, 2),
                '97.5% CI': round(high, 2)
            },
            ignore_index=True
        )

    return data.reindex(columns=['Factor', 'Method', 'Statistic', 'Point Estimate', '2.5% CI', '97.5% CI'])

## Proportions

def binary_95CI(df, cat_vars):
    """Calculates the 95% confidence interval for binary variables in a DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing binary variables.
        cat_vars (list): List of names of binary variables for which to calculate confidence intervals.

    Returns:
        pandas.DataFrame: A DataFrame with the results including factor, point estimate,
            lower bound of the confidence interval (2.5%), and upper bound of the confidence interval (97.5%).

    Notes:
        - Point estimates and confidence intervals are calculated using the proportion_confint function from statsmodels.
        - All values are multiplied by 100 to represent them as percentages.
    """
    data = pd.DataFrame()
    for col in df[cat_vars].columns:
        name = df[col].name
        A = np.asarray(df[col].dropna())
        point = np.sum(A) / len(A)
        CI = proportion_confint(np.sum(A), len(A))
        low = CI[0]
        high = CI[1]

        data = data.append({'Factor': name, 'Point': round(point * 100, 1), '2.5% CI': round(low * 100, 1), '97.5% CI': round(high * 100, 1)}, ignore_index=True)

    return data.reindex(columns=['Factor', 'Point', '2.5% CI', '97.5% CI'])

################################################################################
# Regressions
################################################################################


def calc_vif(df: pd.DataFrame,
             target: str,
             thresh: float = 5.0,          # «красная зона»
             drop_const: bool = True) -> Bunch:
    """
    Возвращает DataFrame со значениями VIF и список «проблемных» признаков
    (VIF ≥ thresh).

    Схема:
        •  численные признаки  – без изменений
        •  бинарные            – без изменений (0/1)
        •  номинальные         – OneHot, drop_first
    """
    X = df.drop(columns=[target]).copy()

    # --- делим признаки по типу ----------------------------------------
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    # удаляем константы, если нужно
    if drop_const:
        const_cols = [c for c in num_cols if X[c].nunique(dropna=False) <= 1]
        X.drop(columns=const_cols, inplace=True)
        num_cols = [c for c in num_cols if c not in const_cols]

    # --- трансформации --------------------------------------------------
    ct = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(drop="first"), cat_cols)
        ],
        remainder="drop"
    )

    X_encoded = ct.fit_transform(X)
    feature_names = (num_cols +
                     list(ct.named_transformers_["cat"].get_feature_names_out(cat_cols)))

    # --- вычисляем VIF --------------------------------------------------
    vif_vals = [variance_inflation_factor(X_encoded, i)
                for i in range(X_encoded.shape[1])]

    vif_tbl = (pd.DataFrame({"Feature": feature_names,
                             "VIF": np.array(vif_vals)})
                 .sort_values("VIF", ascending=False)
                 .reset_index(drop=True))

    return Bunch(vif_table=vif_tbl,
                 low_vif=vif_tbl.loc[vif_tbl["VIF"] < thresh, "Feature"].tolist())



def univariate_lineareg(df: pd.DataFrame,
                        target: str,
                        predictors: list[str],
                        *,
                        normalize_numeric: bool = True,
                        digits: int = 3) -> pd.DataFrame:
    """
    Однофакторный линейный регрессионный анализ.

    • Для числовых признаков (n unique > 2) – один коэффициент (β);
      при `normalize_numeric=True` предварительно стандартизируется.
    • Для категориальных автоматически создаются dummy-переменные
      (statsmodels: `C(var)`), на каждый уровень → отдельная строка.

    Возвращает таблицу:

    Переменная | Категории | Коэффициент | 95% ДИ | R² | p-значение
    """
    rows = []

    for var in predictors:
        sub = df[[target, var]].dropna()
        if sub.empty:
            continue

        # ── числовой / категориальный? ──────────────────────────────
        is_num = is_numeric_dtype(sub[var]) and sub[var].nunique() > 2
        formula = f"{target} ~ {var}" if is_num else f"{target} ~ C({var})"

        # ── опциональная стандартизация числовых ───────────────────
        if is_num and normalize_numeric:
            mu, sd = sub[var].mean(), sub[var].std(ddof=0)
            if sd > 0:
                sub[var] = (sub[var] - mu) / sd

        # ── модель --------------------------------------------------
        model = smf.ols(formula, data=sub).fit()
        r2   = round(model.rsquared, digits)
        pmod = round(model.f_pvalue, digits)

        # коэффициенты без Intercept
        params = model.params.drop('Intercept', errors='ignore')
        ci     = model.conf_int().drop('Intercept', errors='ignore')

        for coef_name, beta in params.items():
            # ---- расшифровываем название коэффициента -------------
            if is_num:
                cat_label = None
                beta_name = var                       # для CI
            else:
                # пример имени: C(Therapy)[T.Drug A]
                cat_label = coef_name.split('T.', 1)[-1].rstrip(']')
                beta_name = coef_name

            ci_lo, ci_hi = ci.loc[beta_name]

            rows.append({
                "Переменная": var,
                "Категории":  cat_label,
                "Коэффициент": round(beta, digits),
                "95% ДИ": f"[{ci_lo:.{digits}f}; {ci_hi:.{digits}f}]",
                "R²": r2,
                "p-значение": pmod
            })

    return (pd.DataFrame(rows)
              .sort_values("p-значение")
              .reset_index(drop=True))


def univariate_logreg(df: pd.DataFrame,
                      target: str,
                      predictors: list[str],
                      *,
                      normalize_numeric: bool = True,
                      digits: int = 3) -> pd.DataFrame:
    """
    Однофакторный ЛОГИСТИЧЕСКИЙ регрессионный анализ
    (binary outcome 0/1).

    Таблица-вывод
    -------------
    Переменная | Категории | Отношение шансов | 95% ДИ | R² | p-значение

    • Для числовых признаков (n>2 уникальных) можно стандартизировать.
    • Для категориальных автоматически строятся dummy-переменные
      (statsmodels:  C(var) ).
    • OR и доверительный интервал получаются из exp(coef ± CI).
    • Pseudo-R² = McFadden (model.prsquared).
    • p-value – LRT p-value (model.llr_pvalue).
    """
    rows = []

    for var in predictors:
        sub = df[[target, var]].dropna()
        if sub.empty:
            continue

        # —- определяем тип признака
        is_num = is_numeric_dtype(sub[var]) and sub[var].nunique() > 2
        formula = f"{target} ~ {var}" if is_num else f"{target} ~ C({var})"

        # —- стандартизация числовых
        if is_num and normalize_numeric:
            mu, sd = sub[var].mean(), sub[var].std(ddof=0)
            if sd > 0:
                sub[var] = (sub[var] - mu) / sd

        # —- логистическая модель
        try:
            model = smf.logit(formula, data=sub).fit(disp=False)
        except Exception as e:            # например perfect separation
            print(f"{var}: не удалось подогнать модель ({e})")
            continue

        pseudo_r2 = round(model.prsquared, digits)
        pmod      = round(model.llr_pvalue, digits)

        # коэффициенты без интерсепта
        params = model.params.drop("Intercept", errors="ignore")
        ci     = model.conf_int().drop("Intercept", errors="ignore")

        for coef_name, beta in params.items():
            if is_num:
                cat_label = None
                idx_name  = var            # ключ для CI
            else:
                # название вида C(var)[T.level]
                cat_label = coef_name.split("T.", 1)[-1].rstrip("]")
                idx_name  = coef_name

            ci_lo, ci_hi = ci.loc[idx_name]

            rows.append({
                "Переменная": var,
                "Категории":  cat_label,
                "Отношение шансов": round(np.exp(beta), digits),
                "95% ДИ": f"[{np.exp(ci_lo):.{digits}f}; {np.exp(ci_hi):.{digits}f}]",
                "R²": pseudo_r2,
                "p-значение": pmod
            })

    return (pd.DataFrame(rows)
              .sort_values("p-значение")
              .reset_index(drop=True))


def onedim_coxregr(df, group, time, adj = False, adj_cols_lst = None):
    """AI is creating summary for onedim_coxregr

    Args:
        df: original dataframe
        group: death or target column
        time: time column
        adj (bool, optional): do we need to adjust for covariates? Defaults to False.
        adj_cols_lst (List): if adj == True, provide list of covariates for adjustments. Defaults to None.
    """
    columns = [x for x in df.columns if (x != group) and (x != time) ]

    coxregr = pd.DataFrame()

    if adj:
        for col in columns:
            try:
                model = cph.fit(df[[col, group, time] + adj_cols_lst].dropna(), duration_col=time, event_col=group)
                HR = round(model.hazard_ratios_[0], 2)
                p = round(model.summary['p'][0], 3)
                conf0 = round(model.summary['exp(coef) lower 95%'][0], 2)
                conf1 = round(model.summary['exp(coef) upper 95%'][0], 2)

            except:
                HR = 'NA'
                p = 1
                conf0 = 'NA'
                conf1 = 'NA'

            coxregr = pd.concat(
                [coxregr,
                pd.DataFrame({'Фактор': df[col].name, 'HR': HR, 'Нижний 95% ДИ': conf0, 'Верхний 95% ДИ': conf1,'p_val': p}, index = [1])],
                    ignore_index=True)

        coxregr = coxregr.reindex(columns=['Фактор', 'HR', 'Нижний 95% ДИ', 'Верхний 95% ДИ', 'p_val'])
    else:
        for col in columns:
            try:
                model = cph.fit(df[[col, group, time]].dropna(), duration_col=time, event_col=group)
                HR = round(model.hazard_ratios_[0], 2)
                p = round(model.summary['p'][0], 3)
                conf0 = round(model.summary['exp(coef) lower 95%'][0], 2)
                conf1 = round(model.summary['exp(coef) upper 95%'][0], 2)

            except:
                HR = 'NA'
                p = 1
                conf0 = 'NA'
                conf1 = 'NA'

            coxregr = pd.concat(
                [coxregr,
                pd.DataFrame({'Фактор': df[col].name, 'HR': HR, 'Нижний 95% ДИ': conf0, 'Верхний 95% ДИ': conf1,'p_val': p}, index = [1])],
                    ignore_index=True)

        coxregr = coxregr.reindex(columns=['Фактор', 'HR', 'Нижний 95% ДИ', 'Верхний 95% ДИ', 'p_val'])

    return(coxregr)


def step_cox(df, group, time, vars, iterations = 1000, penalty = .001):
    """AI is creating summary for step_cox

    Args:
        df: original dataframe
        group: death or target column
        time: time column
        vars: vars to select from for multivariate model
        iterations: number of steps. Defaults to 1000.
        penalty: penalizer argument from CoxPHFitter class. Defaults to .001.

    Returns:
        model_tab: multivariate COX model
    """
    var_lst = vars.copy()

    pen = .001

    cph_selector = CoxPHFitter(penalizer = penalty)

    index_p_max = []

    for number in range(iterations):

        if index_p_max in var_lst:
            var_lst.remove(index_p_max)

        model = cph_selector.fit(df[[group, time] + var_lst].dropna(), duration_col=time, event_col=group)
        p_max = model.summary['p'].max()

        if p_max < 0.05:
            break
        else:
            index_p_max = model.summary['p'].idxmax()

    model2 = cph.fit(df[[group, time] + var_lst].dropna(), duration_col=time, event_col=group)

    model_tab = model2.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].reset_index()
    model_tab.columns = ['Factor', 'HR', 'lower CI',	'upper CI',	'p-val']
    model_tab[['HR', 'lower CI',	'upper CI']] = model_tab[['HR', 'lower CI',	'upper CI']].round(2)
    model_tab[['p-val']] = model_tab[['p-val']].round(3)
    return model_tab


# def onedim_logistic(df, target, adj=False, adj_cols_lst=None):
#     """AI is creating summary for onedim_logistic_regression

#     Args:
#         df: original dataframe
#         target: binary target column (0/1)
#         adj (bool, optional): do we need to adjust for covariates? Defaults to False.
#         adj_cols_lst (List): if adj == True, provide list of covariates for adjustments. Defaults to None.
#     """
#     columns = [x for x in df.columns if x != target]

#     logreg_results = pd.DataFrame()

#     # If adjustment is needed, we fit the model including adjusted columns but do not include them in results
#     if adj_cols_lst is not None:
#         for col in columns:
#             try:
#                 # Prepare the data
#                 X = df[[col] + adj_cols_lst].dropna()
#                 y = df[target].loc[X.index]

#                 # Fit the logistic regression model
#                 model = sm.Logit(y, sm.add_constant(X)).fit(disp=0)

#                 # Extract coefficients and statistics
#                 OR = round(np.exp(model.params[col]), 2)  # Odds Ratio
#                 p = round(model.pvalues[col], 3)          # p-value
#                 conf_int = model.conf_int().loc[col]
#                 conf0 = round(np.exp(conf_int[0]), 2)     # Lower CI
#                 conf1 = round(np.exp(conf_int[1]), 2)     # Upper CI

#             except Exception as e:
#                 OR = 'NA'
#                 p = 1
#                 conf0 = 'NA'
#                 conf1 = 'NA'

#             logreg_results = pd.concat(
#                 [logreg_results,
#                  pd.DataFrame({'Фактор': col, 'OR': OR, 'Нижний 95% ДИ': conf0, 'Верхний 95% ДИ': conf1, 'p_val': p}, index=[0])],
#                 ignore_index=True)
#             logreg_results = logreg_results[~logreg_results['Фактор'].isin(adj_cols_lst)]

#     else:
#         # Fit models without adjustment
#         for col in columns:
#             try:
#                 # Prepare the data
#                 X = df[[col]].dropna()
#                 y = df[target].loc[X.index]

#                 # Fit the logistic regression model
#                 model = sm.Logit(y, sm.add_constant(X)).fit(disp=0)

#                 # Extract coefficients and statistics
#                 OR = round(np.exp(model.params[col]), 2)  # Odds Ratio
#                 p = round(model.pvalues[col], 3)          # p-value
#                 conf_int = model.conf_int().loc[col]
#                 conf0 = round(np.exp(conf_int[0]), 2)     # Lower CI
#                 conf1 = round(np.exp(conf_int[1]), 2)     # Upper CI

#             except Exception as e:
#                 OR = 'NA'
#                 p = 1
#                 conf0 = 'NA'
#                 conf1 = 'NA'

#             logreg_results = pd.concat(
#                 [logreg_results,
#                  pd.DataFrame({'Фактор': col, 'OR': OR, 'Нижний 95% ДИ': conf0, 'Верхний 95% ДИ': conf1, 'p_val': p}, index=[0])],
#                 ignore_index=True)

#     logreg_results = logreg_results.reindex(columns=['Фактор', 'OR', 'Нижний 95% ДИ', 'Верхний 95% ДИ', 'p_val'])

#     return logreg_results


def step_logistic(df, target, vars, iterations=1000, threshold=0.05):
    """AI is creating summary for step_logistic

    Args:
        df: original dataframe
        target: binary target column (0/1)
        vars: variables to select from for multivariate model
        iterations: number of steps. Defaults to 1000.
        threshold: p-value threshold for variable selection. Defaults to 0.05.

    Returns:
        model_tab: multivariate logistic regression model summary
    """
    var_lst = vars.copy()
    model_results = pd.DataFrame()

    for number in range(iterations):
        # Fit the logistic regression model with current variables
        X = df[var_lst].dropna()
        y = df[target].loc[X.index]

        if len(y) == 0:
            break  # Exit if there are no observations

        model = sm.Logit(y, sm.add_constant(X)).fit(disp=0)

        # Get p-values and find the maximum p-value
        p_values = model.pvalues[1:]  # Exclude the intercept
        p_max = p_values.max()

        # Check if the maximum p-value is below the threshold
        if p_max < threshold:
            break
        else:
            # Remove the variable with the highest p-value
            index_p_max = p_values.idxmax()
            list(var_lst).remove(index_p_max)

    # Final model fitting with selected variables
    final_model = sm.Logit(y, sm.add_constant(df[var_lst].dropna())).fit(disp=0)

    # Prepare summary table
    model_tab = final_model.summary2().tables[1].reset_index()
    model_tab.columns = ['Factor', 'Coef', 'Std Err', 'z', 'P>|z|', '[0.025', '0.975]']

    # Calculate Odds Ratios and Confidence Intervals
    model_tab['OR'] = np.exp(model_tab['Coef'])
    model_tab['lower CI'] = np.exp(model_tab['[0.025'])
    model_tab['upper CI'] = np.exp(model_tab['0.975]'])

    # Select relevant columns and round values
    model_tab = model_tab[['Factor', 'OR', 'lower CI', 'upper CI', 'P>|z|']]
    model_tab[['OR', 'lower CI', 'upper CI']] = model_tab[['OR', 'lower CI', 'upper CI']].round(2)
    model_tab[['P>|z|']] = model_tab[['P>|z|']].round(3)

    return model_tab


################################################################################
# Graphics
################################################################################

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

def draw_data_frame(df, col_lst, pict_sav=True):

    for col in col_lst:
        if pd.CategoricalDtype.is_dtype(df[col]) == True:
            if len(set(df[col].dropna())) < 6:

                B = round(df[col].dropna().value_counts() / df[col].dropna().shape[0]*100, 1).reset_index()
                B.columns=[col, 'Доля, %']

                sns.set(style='whitegrid')
                fig, ax = plt.subplots(figsize=(7,7))
                g = sns.barplot(data=B, x=col, y='Доля, %', ax=ax)
                for p in g.patches:
                    g.annotate(
                        str(format(p.get_height(), '.1f')) + ' %',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'center',
                    xytext = (0, 9),
                    textcoords = 'offset points')
                plt.show()
                if pict_sav:
                    g.figure.savefig(col  + '.png')

            else: pass

        else:
            sns.set(style = 'whitegrid')
            fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, figsize = (7, 7), gridspec_kw={"height_ratios": (.15, .85)})
            sns.boxplot(df[col], ax=ax_box, color = 'lightblue')
            g = sns.histplot(df[[col]], ax = ax_hist, color = 'blue', bins = 10, kde = True, label='_nolegend_')
            ax_hist.set(xlabel=col)
            ax_hist.get_legend().remove()
            plt.ylabel('Количество наблюдений')
            ax_box.set(xlabel='')
            plt.show()
            if pict_sav:
                g.figure.savefig(col  + '.png')

## draw every variable with grouping (boxplots / barplots)

def draw_data_frame_group(df, col_lst, group, pict_sav=True, add_number=True):

    names = [str(x) + ' ' for x in  np.array(range(1, len(col_lst) + 1))]

    for col, name in zip(col_lst, names):
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
                    g.annotate(
                        str(format(p.get_height(), '.1f')) + (' %'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'center',
                    xytext = (0, 9),
                    textcoords = 'offset points')

                plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)## легенда снаружи
                plt.xlabel('')
                plt.ylabel(col)
                plt.show()

                if pict_sav:
                    if '/' in col:
                        col = col.replace('/', '_')
                    if add_number:
                        g.figure.savefig(name + col + '.png')
                    else:
                        g.figure.savefig(col  + '.png')

            else: pass

        else:
            sns.set(style='whitegrid')
            fig, ax = plt.subplots(figsize=(8,8))
            g=sns.boxplot(group, col, hue=group, data=df, ax=ax, dodge=False)
            ax.legend([],[], frameon=False)
            plt.ylabel(col)
            plt.xlabel('')
            plt.show()

            if pict_sav:
                if '/' in col:
                    col = col.replace('/', '_')
                if add_number:
                    g.figure.savefig(name + col + '.png')
                else:
                    g.figure.savefig(col  + '.png')


## Bland Altman Plot

def bland_altman_plot(data1, data2, x_label='', y_label='',save=False, name=None, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2
    md        = np.mean(diff)
    sd        = np.std(diff, axis=0)

    fig, ax = plt.subplots(figsize=(10,6))
    g=sns.scatterplot(mean, diff, *args, **kwargs, color='g')
    plt.axhline(md,           color='red', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save:
        g.figure.savefig(name  + '.png')


## Polar Circular Plot

def polar_plot_circular(
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
