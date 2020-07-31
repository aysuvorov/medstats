# Medstats python module

Python module for medical statistics. This bunch of functions is dedicated to medical statistics I use in my work and for fun.

## List of functions:

### Import from google sheets:

- `import_gsheet` - imports gsheet table into workflow **when working in Colab**. To use the function you need to authenticate into your Google account with

```
from google.colab import auth
auth.authenticate_user()
```

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
- `regr_onedim` - provides one-dimensional logistic regression analysis
- `backwise` - provides backwise selection model
- `roc_cut` - calculates cut-offs thresholds for significant numeric factors in regression analysis
