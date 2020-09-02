# Medstats python module

Python module for medical statistics. This bunch of functions is dedicated to medical statistics I use in my work and for fun.

Module uses
```
numpy
pandas as pd
scipy
math
statsmodels
sklearn
dask
numba
lifelines
```
You need to have all theese modules installed. 

## List of functions: for details see [Manual](https://github.com/aysuvorov/medstats/blob/Beta_1/Manual.md)

### Misc

- `filler` - fills NaN with `df.mean` / `df.median` / `df.interpolate`. Translates columns to numerics.

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
- `compare` - statistical comparison between 2 groups. By default numerics compared using Mann-Whitney (or you can use Welch t-test), shares - with Fisher exact test
- `regr_onedim` - provides one-dimensional logistic regression analysis. Sex and Age adjustment is available.

### Survival

- `cox_onedim` - onedimensional Cox regressions over every variable in data frame. Sex and Age adjustment is available.

### Models and quality

- `backwise` - provides backwise selection model in logistic regressions
- `roc_cut` - calculates cut-offs thresholds for significant numeric factors for logistic and Cox regression analysis

### Graphics

- `forrest_plot` - draws simple Forrest plot
- `summary_graph` - Plots every column of dummified dataframe as histogram via pandas.plot. Settings are default. For dummified variables color is green, for numerics - blue
