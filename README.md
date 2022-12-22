# Medstats module

Collection of scripts and modules for biostatistics and data science simple automation for research process.

Medstats is a child project of [**YASP!**](https://aysuvorov.github.io/) by [Aleksandr Suvorov](https://aysuvorov.github.io/docs/promotion/cv/cv_raw.html), senior statistician of [Center of analysis of Complex Systems](https://cacs.ai/). 

<p align="center">
    <a href="https://aysuvorov.github.io/" title="YASP">
        <img src="https://aysuvorov.github.io/docs/promotion/main_logo.png" width="200"/>
    </a>
    <a href="https://cacs.ai/" title="CACS">
        <img src="https://raw.githubusercontent.com/uwadim/cacs/main/images/main_logo.jpg" width="200"/>
    </a>
</p>


# Module contents

## `Describe`

Desribe is a module providing various basic statistical functions, plotting, comparative criterial approaches typically used in clinical trials.

Module uses `rpy2`-module and basic `R` libraries.

Module is under heavy readjustment.

## `ModPerf`

ModPerf is a module that assesses typical metrics known in medical trials for machine learning algorithms.

**Metric functions**:

- `ModPerf_AUC` - computes AUC and AUC 95%CI using bootstrap for binary classifier, probabilities for binary classifier, some quantitative variable;
- `ModPerf_Binary` - computes confusion matrix and Se, Sp, NPV, PPV and 95%CI using bootstrap for binary classifier output (not probabilities);
- `ModPerf_Multiclass` - computes confusion matrix and Se, Sp, NPV, PPV and 95%CI using bootstrap for vulticlass classifier output (not probabilities);
- `ModPerf_thresholds` - computes thresholds for probabilities for binary classifier, some quantitative variable with confusion matrix, Se, Sp, NPV, PPV and 95%CI using bootstrap;

**Plotting functions**

- `ROCPlotter_Binary` - plots AUC curve for binary classifier, probabilities for binary classifier, some quantitative variable;
- `ROCPlotter_Multiclass` - plots AUC curve for multiclass classifier, probabilities for multiclass classifier;

## `Parenclitic`

Various approaches for network analysis:

- parenclitic graphs with a threshold;
- weighted parenclitic graphs;
- synolitic graphs;
- correlation graphs for time series

**Common classes**

- `DataFrameLoader` - common class for loading data (for all types of approaches);
- `Prct` - common class for parenclytic approach;
- `Snltc` - common class for synolytic approach;
- `Corr` - common class for correlation approach;

**Common functions**

- `graph_plotter` - plots single weighted graph;
-  `chars` -computes various characteristics of a single graph; 

## `PhisioPatterns`

- test module for time series and longitudinal data (ECG, EEG). Under heavy development;

## Test platform

- `MLSelectionFlow_devel` - programm for automatic feature selection (SHAP) and model selection (optuna module);

