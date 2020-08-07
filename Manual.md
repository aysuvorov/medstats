# M a n u a l

## Intro

This module was created for young researches in medicine and other human sciences to collect typical functions we use in clinical research. This project is like an analog of`coppareGroups` or `finalfit` libraries in R. 


## Import this repo from git 

For direct import and installation of this repo into Jupyter Notebook or Google Colab you can use:
```
!wget https://raw.githubusercontent.com/aysuvorov/medstats/Beta/medstats.py

!pip install medstats

import medstats as ms
```

or you can use 

```
!git clone https://github.com/aysuvorov/medstats.git

!cd /content/medstats

!python -m medstats

import medstats as ms
```
## List of functions
## Sample size

### `prop_size(p1, p2, alpha = 0.05, beta = 0.8, k = 1)`

Calculates sample size for 2 proportion test
Returns text of the trial parameters

- `p1`, `p2` - proportions of 2 drugs / drug-placebo. I.e. 0.5, 0.76, 0.4 ...
- `alpha` - type I error
- `beta` - 1 - type II error (trial power)
- `k` - test / placebo groups size ratio. I.e. 3/1, 1/2, etc.

### `mean_size(m1, m2, sd1, sd2, alpha = 0.05, beta = 0.8, k = 1)`

Calculates sample size for 2 means
Returns text of the trial parameters

- `m1`, `m2`, 'sd1', 'sd2' - known 2 means and sd's of groups
- `alpha` - type I error
- `beta` - 1 - type II error (trial power)
- `k` - test / placebo groups size ratio. I.e. 3/1, 1/2, etc.

## Misc: import, bootstrap comparison, Nan fill, CDF

### `cdf(array)`

- computes CDF of an array. Returns `x` - value and  `y` - share of this value. Youcan pass them directly to matplotlib.

### `import_gsheet(key, sheet = 0)`

- `key` is table UIN in google sheets:
F.e. https://docs.google.com/spreadsheets/d/`1CY3vBMfJNf55UkfrgdcefiUy6Jf2DZdOWDVtsan3T1dkS0`/edit#gid=12556364 the `key` is 1CY3vBMfJNf55UkfrgdcefiUy6Jf2DZdOWDVtsan3T1dkS0

- `sheet` is sheet number in your google table, starting from 0: 0,1,2..., in pythonists way

Returns pd.DataFrame
Sometimes makes errors in numeric vars (I didn't find any solution, I think it's not from python side)

### `filler(df, func)`

- `df` - data frame, needs to be dummified. Character vars wont work with the function
- `func` should be `df.mean` / `df.median` / `df.interpolate`

Returns `df` with no `NaN`

### `bs_multi(a,b, func, R = 100)`

Compares A and B arrays with bootstrap simulation 
Returns a list with 2.5, 50, 97.5 percentiles of p-value. 

- `a, b` - np.arrays
- `func` - np.mean, np.median, np.std, st.iqr, np.corrcoef - the func statistic you want to compare
- `R` - sqrt of number of simulations. Number of simulations is R x R, i.e. R = 100  is 100000 sims

### `bs_perc(a,b, perc, R = 100)`

Same function as `bs_multi` but for percentiles comparison
Returns a list with 2.5, 50, 97.5 percentiles of p-value. 

- `perc` is centile as 25 / 75 / 2.5, etc. 

### `bs_props(inv, inv_n, plac, plac_n, R=100)`

Bootstrap comparison for 2 proportions
Returns a list with 2.5, 50, 97.5 percentiles of p-value. 

- `inv` - successes in Intervention group, 30% will be `30.`, 75,1% - `75.1`
- `inv_n` - number of trials in Intervention group. 
- `plac` - successes in Placebo group, 30% will be `30.`, 75,1% - `75.1`
- `plac_n` - number of trials in Placebo group. 
- `R` - sqrt of number of simulations. Number of simulations is R x R, i.e. R = 100  is 100000 sims

## Table statistics and stuff - working with simple databases

### `dummification(df, cat_vars)`

- `df` - takes df
- `cat_vars` - list of vars to be dummified. Delets original vars.

Returns data frame.

### `summary(df, save_tab = False)` 

Provides summary table with var types (numeric / category - but in **RUSSIAN!**. You can simply change in the body of a function))
The function returns N - value counts for all columns in the data frame, 
- shares for categorical variables (2 uniques in the column), 
- median + iqr, mean + sd, min, max, pvalue of shapiro test for numeric values

Categorical variables must be dummified as 0/1. 
If some value is unique, the row has `Уникальная` category and the value is repeated for all statistics. 

- `df` - data frame to proceed
- `save_tab` - save to xlsx table, 'Описательные статистики.xlsx' means `Deascriptives.xlsx`

### `compare(df, group, gr_id_1 = 0, gr_id_2 = 1, name_1 = 'Группа 0', name_2 = 'Группа 1', test = 'mw', save_tab = False)`

Compares **2 groups**. 

Categorical dummified vars are compared with Fisher exact test. 
Numeric vars are compared with Mann-Whitney test if **test = 'mw'** and with Welch test (t test for unequal variances) if **test = 'tt'**.
Returns the table with value counts and shares for categorical variables, medains(25%, 75%) or means + sd for numerical variables, p-value of Mann-Whitney or Welch test.

- `df` - data frame
- `group` - the var for grouping. I.e. `group = 'GROUP'`
- `gr_id_1, gr_id_2` - categories of grouping var if not 0/1.
- `name_1, name_2` - names of groups in grouping var. I.e. **name_2 = 'Группа 1'**. They appear in result table.
- `test` - for unnormal distributed numerics and Mann - Whitney test use **test = 'mw'**, for normal distributed numerics - **test = 'tt'**
- `save_tab` - save to xlsx table

### `regr_onedim(df, group, adjusted = False, signif_only = False, age_col = 1, sex_col = 1, save_tab = False)`

One-dimensional regression analysis for every column in df.
Returns the table with calculated OR and its 95% CI, p-value. Function uses binomial GLM. 

- `df` - data frame
- `group` - the var for grouping. I.e. `group = 'GROUP'`
- `adjusted` - if `adjusted = True`, you can analyse every variable together with GENDER and AGE (typical in medicine, cardiology)
- `age_col = 1, sex_col = 1` - provided names in `**sex_col = 'GENDER'**` , `**age_col = 'AGE'**` format , for adjusted regression.
- `signif_only` - show only significant OR in result table
- `save_tab` - save to xlsx table

### `regr_multi(df, group, lst, save_tab = False)`

With significant factors from `regr_onedim` yuo can create multivariate regression with OR, 95% CI, p-values. Result is table. 

- `df` - data frame
- `group` - the var for grouping. I.e. `group = 'GROUP'`
- `lst` - list of significant factors
- `save_tab` - save to xlsx table

### `cox_onedim(df, group, time, adjusted = False, signif_only = False, age_col = 1, sex_col = 1, save_tab = False)`

One dimensional Cox regression analysis. Makes the run through all the data frame variables, like `reg_onedim`. Returns a table with factor name, HR, 95% CI, p-value of Cox regression.

- `df` - data frame
- `group` - the var of endpoint. I.e. `group = 'GROUP'`
- `time` - time to endpoint variable
- `adjusted` - if `adjusted = True`, you can analyse every variable together with GENDER and AGE (typical in medicine, cardiology)
- `age_col = 1, sex_col = 1` - provided names in `**sex_col = 'GENDER'**` , `**age_col = 'AGE'**` format , for adjusted regression.
- `signif_only` - show only significant OR in result table
- `save_tab` - save to xlsx table

## Backwise models and ROC-threshold cut offs
### `backwise(df, lst, group, time = 0, family = 'logistic', steps = 100, pmin = 0.05)`

Provides backwise selection of factors dropping largest p-values untill all the factors are significant. Returns table with names, HR/OR, 95% CI, p-values. 

- `family` - type of regression. Can be `'logistic'` or `'cox'`.
- `df` - data frame
- `group` - the var of endpoint. I.e. `group = 'GROUP'`
- `lst` - list of significant factors or variables of interest
- `time` - time variable, if we deal with `family = 'cox'`
- `steps` - number of steps for model selection. For many factors consider greater steps.
- `pmin` - threshold for significant p-value to select. Some researches consider that `p = 0.1` is also acceptable.

### `roc_cut(df, vars, group, time = 0, family = 'logistic', save_tab = False)`

Takes df, list of numeric vars, classifier and calculates optimal cut off for the classifier.
Returns table with name, AUC, optimal cut off, sensetivity, specificity. 

- `family` - type of regression. Can be `'logistic'` or `'cox'`.
- `df` - data frame
- `vars` - list of factors or variables of interest
- `group` - the classifier. I.e. `group = 'GROUP'`
- `time` - time variable, if we deal with `family = 'cox'`
- `save_tab` - save to xlsx table
