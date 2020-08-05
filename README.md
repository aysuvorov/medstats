# Medstats python module

Python module for medical statistics. This bunch of functions is dedicated to medical statistics I use in my work and for fun.

## List of functions:

### Import from google sheets:

- `import_gsheet` - imports gsheet table into workflow **when working in Colab**. To use the function you need to authenticate into your Google account with

```
from google.colab import auth
auth.authenticate_user()
```
As arguments you should pass table key:

https://docs.google.com/spreadsheets/d/`1CY3vBMfJNf55UkfrgdcefiUy6Jf2DZdOWDVtsan3T1dkS0`/edit#gid=12556364 as '1CY3vBMfJNf55UkfrgdcefiUy6Jf2DZdOWDVtsan3T1dkS0' and sheet number.

### Misc

- `filler` - fills NaN with `mean` / `median` / `interpolate`. Translates columns to numerics.

### Sample size:

- `prop_size` - calculates sample size with 2 known proportions
- `mean_size` - calculates sample size with 2 known means
- `cdf` - computes CDF

### Bootstrap comparison:

- `bs_multi` - compares 2 arrays using any stats functions you like (means, medians, iqr, R2, rho, slopes?)
- `bs_perc` - compares percentiles of 2 arrays
- `bs_props` - proportions comparison

### Table statistics and stuff - working with simple databases:

- `dummification` - creates dummies and deletes original vars 
- `summary` - provides simple summary table with var types (numeric / category).
- `compare` - statistical comparison between 2 groups. By default numerics compared using Mann-Whitney, shares - with Fisher exact test
- `regr_onedim` - provides one-dimensional logistic regression analysis. Sex and Age adjustment is available.

### Survival

- `cox_onedim` - onedimensional Cox regressions over every variable in data frame. Sex and Age adjustment is available.

### Models and quality

- `backwise` - provides backwise selection model in logistic regressions
- `roc_cut` - calculates cut-offs thresholds for significant numeric factors for logistic and Cox regression analysis
