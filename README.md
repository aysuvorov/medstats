# Medstats python module

Python module for medical statistics. 



## `describe.py`

Main module for summary descriprive statistics and comparative analysis.

**Data cleaning**

- `columnn_normalizer` - deletes column delimeters, commas

- `column_categorical` - creates categorical variables

- `miss_counter` - counts absolute missing values and proportions columnwise

- `dplyr_filter` - same behavior as R's `dplyr %>% filter` function

- `dummification` - creates dummy variables from columns preserving `NaN`'s

**Descriptive statistics**

- `p_adjust` - corrects p-values accroding to multiple comparisons

- `summary_all`- summary statistics of a dataframe

- `compare_table_2g` - comparative analysis of 2 groups (criterial comparoisons)

- `compare_table_3g`- comparative analysis of 2 groups (criterial comparoisons) with/without corrections for multiple testing

- `numerics_95CI` - computes statistic (mean/median) and its 95% CI for numeric variables

- `binary_95CI` - computes proportions and 95% CI 

**Graphics**

- `dist_box` - plots boxplots and distributions of numeric variables

- `draw_data_frame` - plots boxplots and barplots for numeric/categorical variables in a dataframe

- `draw_data_frame_group`plots boxplots and barplots for numeric/categorical variables in a dataframe with a grouping variable

- `bland_altman_plot` - plots Bland Altman Plot

- `polar_plot_circular` - plots radial circular plot alike [this](https://i.stack.imgur.com/w5TtL.png)

## `regressions.py`

- `ModPerformance` - class for model performance evaluation
  
  - `auc` - computes AUC and 95% CI (DeLong) - using R's `pROC` library
  
  - `threshold_getter` - returns the dataframe with thresholds for numeric var/predictions and Se, Sp, PPV, NPV, Brier score, LR+, LR-.
  
  - `plot_roc` - plots AUC
  
  - `intrplt` - interpolates Sens/Spec when needed fixed Senc or Spec threshold

- `onedim_logregr` - univariate logistic regressions

- `onedim_coxregr` - univariate COX regressions

- `step_mass_R` - R stepwise selection algorithm using logistic regression

- `cox_multi` & `odds_multi` - COX and logistic multivariate models

## `matchit.py`

- `matchit` - propensity score matching algorithm using R's `MatchIt` library

## `parenclitic_oop.py`

Parenclitic network analysis and A.Gorban correlation graphs

- class`Zanin` - creates and fits Zanin parenclitic graphs

- class`Gorban` - creates and fits Gorban correlation graphs

- `graph_plotter` - plots individual graphs


