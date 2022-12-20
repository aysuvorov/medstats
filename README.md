# Medstats module

## Describe

Desribe is a module providing various basic statistical functions, plotting, comparative criterial approaches typically used in clinical trials.

Module is under heavy readjustment.

## ModPerf

ModPerf is a module that typical metrics known in medical trials for machine learning algorithms.

**Metric functions**:

- `ModPerf_AUC` - computes AUC and AUC 95%CI using bootstrap for binary classifier, probabilities for binary classifier, some quantitative variable;
- `ModPerf_Binary` - computes confusion matrix and Se, Sp, NPV, PPV and 95%CI using bootstrap for binary classifier output (not probabilities);
- `ModPerf_Multiclass` - computes confusion matrix and Se, Sp, NPV, PPV and 95%CI using bootstrap for vulticlass classifier output (not probabilities);
- `ModPerf_thresholds` - computes thresholds for probabilities for binary classifier, some quantitative variable with confusion matrix, Se, Sp, NPV, PPV and 95%CI using bootstrap;

**Plotting functions**

- `ROCPlotter_Binary` - plots AUC curve for binary classifier, probabilities for binary classifier, some quantitative variable;
- `ROCPlotter_Multiclass` - plots AUC curve for multiclass classifier, probabilities for multiclass classifier;


