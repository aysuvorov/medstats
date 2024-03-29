import pandas as pd
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.preprocessing import LabelBinarizer
from importlib import reload
from itertools import cycle

import warnings
warnings.filterwarnings("ignore")


# Module provides techniques for model quality or classification estimation

# +-----------------------------------------------------------------------------

# Functions for determining sensitivity, specificity, positive and negative 
# predictive value and a number of other characteristics of the model

# Evaluation of a binary classification or a quantitative predictor 
# in a binary classification

def ModPerf_AUC(real, pred, num_resamples = 1000, ci95 = True):

    if type(real) != np.ndarray:
        real = np.array(real)

    if type(pred) != np.ndarray:
        pred = np.array(pred)

    Y = np.array([real, pred]).T
    Y = Y[~np.any(np.isnan(Y), axis=1)]

    AUC = metrics.roc_auc_score(Y[:,0], Y[:,1])

    if AUC < 0.5:
        # print('Inverted real class!!!')
        
        Y[:,0] = np.abs(Y[:,0] - 1)
        AUC = metrics.roc_auc_score(Y[:,0], Y[:,1])

    if ci95:

        # Bootsrap CI-s
        random.seed(0)
        vals = []
        for i in range(num_resamples):
            a = np.array(random.choices(Y, k=len(Y)))
            try:
                vals = vals + [metrics.roc_auc_score(
                    a[:,0], 
                    a[:,1]
                )]
            except:
                pass

        vals = np.array(vals)

        auc_ci = np.percentile(vals, [2.5, 97.5])

        return[np.round(x, 3) for x in [AUC, auc_ci]]

    else:

        return np.round(AUC, 3)



def ModPerf_Binary(real, pred, num_resamples = 1000, data_frame = True, ci95 = True):

    if type(real) != np.ndarray:
        real = np.array(real)

    if type(pred) != np.ndarray:
        pred = np.array(pred)

    Y = np.array([real, pred]).T
    Y = Y[~np.any(np.isnan(Y), axis=1)]


    def _quality_point_est(reals, preds):
        """AI is creating summary for quality_point_est

        Args:
            real (np.array): [description]
            pred (np.array): [description]
        """
        tn, fp, fn, tp = metrics.confusion_matrix(reals, preds).ravel()

        auc = metrics.roc_auc_score(reals, preds)
        sens = tp/(tp + fn)
        spec = tn/(tn + fp)
        npv = tn/(tn + fn)
        ppv = tp/(tp + fp)
        return(auc, sens, spec, npv, ppv)

    # Point estimates:
    tn, fp, fn, tp = metrics.confusion_matrix(Y[:,0], Y[:,1]).ravel()
    auc, sens, spec, npv, ppv = _quality_point_est(Y[:,0], Y[:,1])

    if ci95:

        # Bootsrap CI-s
        random.seed(0)
        vals = []
        for i in range(num_resamples):
            a = np.array(random.choices(Y, k=len(Y)))
            try:
                vals = vals + [list(_quality_point_est(
                    a[:,0], 
                    a[:,1]
                ))]
            except:
                pass

        vals = np.array(vals)

        auc_ci, sens_ci, spec_ci, npv_ci, ppv_ci = \
            [np.percentile(vals[:,i][~np.isnan(vals)[:,i]], [2.5, 97.5]) for i in range(5)]

        auc, sens, spec, npv, ppv, auc_ci, sens_ci, spec_ci, npv_ci, ppv_ci = \
            [np.round(x, 3) for x in [auc, sens, spec, npv, ppv, auc_ci, sens_ci, \
                spec_ci, npv_ci, ppv_ci]]

        # Making final data frame
        names = ['tn', 'fp', 'fn', 'tp', 'auc', 'sens', 'spec', 'npv', 'ppv']
        pe = [tn, fp, fn, tp, auc, sens, spec, npv, ppv]
        ci = ['-', '-', '-', '-', auc_ci, sens_ci, spec_ci, npv_ci, ppv_ci]

        if data_frame:
            table = pd.DataFrame({'Chars': names, 'Point_est': pe, '95%CI': ci})
            return table
        else:
            return[tn, fp, fn, tp, auc, sens, spec, npv, ppv,\
                auc_ci, sens_ci, spec_ci, npv_ci, ppv_ci]

    else:

        auc, sens, spec, npv, ppv = \
            [np.round(x, 3) for x in [auc, sens, spec, npv, ppv]]

        return[tn, fp, fn, tp, auc, sens, spec, npv, ppv]
      

def ModPerf_Multiclass(real, pred, num_resamples = 1000): 

    if type(real) != np.ndarray:    
        real = np.array(real)

    if type(pred) != np.ndarray:
        pred = np.array(pred)

    Y = np.array([real, pred]).T
    Y = Y[~np.any(np.isnan(Y), axis=1)]   

    
    def _quality_point_est_multiclass_macro(reals, preds):

        lb = LabelBinarizer()

        cnf_matrix = metrics.confusion_matrix(reals, preds)

        real_lb = lb.fit_transform(reals)
        pred_lb = lb.transform(preds)

        auc = metrics.roc_auc_score(real_lb, pred_lb, average='macro', 
            multi_class = 'ovr')

        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        sens = TP/(TP+FN)
        spec = TN/(TN+FP) 
        ppv = TP/(TP+FP)
        npv = TN/(TN+FN)

        auc, sens, spec, ppv, npv = \
            [round(np.mean(x),3) for x in [auc, sens, spec, ppv, npv]]
        return(auc, sens, spec, npv, ppv)

    # Point estimates:
    auc, sens, spec, npv, ppv = _quality_point_est_multiclass_macro(Y[:,0], Y[:,1])

    # Bootstrap results
    vals = []
    for i in range(num_resamples):
        a = np.array(random.choices(Y, k=len(Y)))
        try:
            vals = vals + [list(_quality_point_est_multiclass_macro(
                a[:,0], 
                a[:,1]
            ))]
        except:
            pass

    # return vals
    vals = np.array(vals)
    auc_ci, sens_ci, spec_ci, npv_ci, ppv_ci = \
        [np.percentile(vals[:,i][~np.isnan(vals)[:,i]], [2.5, 97.5]) for i in range(5)]

    names = ['auc', 'sens', 'spec', 'npv', 'ppv']
    pe = [auc, sens, spec, npv, ppv]
    ci = [np.round(x,3) for x in [auc_ci, sens_ci, spec_ci, npv_ci, ppv_ci]]

    table = pd.DataFrame({'Chars': names, 'Point_est': pe, '95%CI': ci})

    return table


# +-----------------------------------------------------------------------------

# ROC AUC plots for binary classification and scores, and for 
# multiclass classifiers

def ROCPlotter_Binary(real, pred, title=None, plot=True, save_name=''):

    if metrics.roc_auc_score(real, pred) < 0.5:
        real = np.abs(real-1)
        print('Inverted class!')

    thresholds = np.sort(pred, axis=None)

    ROC = np.zeros((len(real) + 2, 2))
    ROC = np.append(ROC, [[1,1]],axis = 0)[::-1]

    for i in range(len(real)):
        t = thresholds[i]

        # Classifier / label agree and disagreements for current threshold.
        TP_t = np.logical_and( pred > t, real==1 ).sum()
        TN_t = np.logical_and( pred <=t, real==0 ).sum()
        FP_t = np.logical_and( pred > t, real==0 ).sum()
        FN_t = np.logical_and( pred <=t, real==1 ).sum()

        # Compute false positive rate for current threshold.
        FPR_t = FP_t / float(FP_t + TN_t)
        ROC[i+1,0] = FPR_t

        # Compute true  positive rate for current threshold.
        TPR_t = TP_t / float(TP_t + FN_t)
        ROC[i+1,1] = TPR_t

    fig = plt.figure(figsize=(6,6))
    plt.plot(ROC[:,0], ROC[:,1], lw=2)
    plt.plot([[0,0], [1,1]], linestyle='--', c = 'gray')
    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,1.1)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.grid()

    if title:
        plt.title(title + '\n\nAUC = %.3f'% metrics.roc_auc_score(real, pred))
    else:
        plt.title('ROC curve, AUC = %.3f'% metrics.roc_auc_score(real, pred))
    if plot:
        plt.show()

    if save_name != '':
        fig.savefig(save_name + '.png', facecolor='white', transparent=False)


def ROCPlotter_Multiclass(real, pred, n_classes, title=None, plot=True, save_name=''):

    if len(real.shape) < 2 or len(pred.shape) < 2:
        raise ValueError('ERROR: real and pred vectors must be binarized!')

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    lw = 2
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(real[:, i], pred[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(real.ravel(), pred.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    fig = plt.figure(figsize=(6,6))

    colors = cycle(["red", "green", "blue", "orange", "gray"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            linestyle=":",
            label="ROC curve of class {0} (AUC = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([-0.1, 1.0])
    plt.ylim([-0.1, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()

    if title:
        plt.title(title + '\n\nAUC = %.3f'%roc_auc["macro"])
    else:
        plt.title('ROC curve, AUC = %.3f'%roc_auc["macro"])
    if plot:
        plt.show()

    if save_name != '':
        fig.savefig(save_name + '.png', facecolor='white', transparent=False)

# +-----------------------------------------------------------------------------
# The function calculates quality scores for thresholds


def ModPerf_thresholds(real, pred):

    if type(real) != np.ndarray:
        real = np.array(real)

    if type(pred) != np.ndarray:
        pred = np.array(pred)

    Y = np.array([real, pred]).T
    Y = Y[~np.any(np.isnan(Y), axis=1)]

    g = pd.DataFrame()

    for score in sorted(set(Y[:,1])):
        tab = pd.Series(ModPerf_Binary(Y[:,0], \
            np.array((Y[:,1] >= score).astype(int)), ci95 = False))
        g = pd.concat([g, tab], axis=1)

    g.columns = sorted(set(Y[:,1]))
    g = g.T.reset_index()
    g.rename(columns={'index':'Thres'}, inplace=True)
    g.columns = ['Thres','tn', 'fp', 'fn', 'tp', 'auc', 'sens', 'spec', \
        'npv', 'ppv']
    g = g.drop('auc',1)
    return g

# +-----------------------------------------------------------------------------
# The function compares 2 ROC-curves with real values and McNemar test
# (Original in https://www.kaggle.com/code/ritvik1909/mcnemar-test-to-compare-classifiers/notebook)

def McNemar2AUCs(real, model_1_preds, model_2_preds):
    
    contingency_table = [[0, 0], [0, 0]]
    Y_ = pd.DataFrame(
        dict(
            ground_truth = real,
            model_1 = model_1_preds,
            model_2 = model_2_preds
        )
    )
    model_1_correct = Y_.apply(lambda row: int(row[ground_truth] == row[model_1]), axis=1)
    model_2_correct = Y_.apply(lambda row: int(row[ground_truth] == row[model_2]), axis=1)
    contingency_table[0][0] = Y_.apply(
        lambda row: int(row[model_1] == 0 and row[model_2] == 0), axis=1
    ).sum()
    contingency_table[0][1] = Y_.apply(
        lambda row: int(row[model_1] == 0 and row[model_2] == 1), axis=1
    ).sum()
    contingency_table[1][0] = Y_.apply( 
        lambda row: int(row[model_1] == 1 and row[model_2] == 0), axis=1
    ).sum()
    contingency_table[1][1] = Y_.apply(
        lambda row: int(row[model_1] == 1 and row[model_2] == 1), axis=1
    ).sum()
    
    test = mcnemar(contigency_table, exact=False, correction=True)
    print("P value:", test.pvalue)
    if test.pvalue < significance:
        print("Reject Null Hypotheis")
        print("Conclusion: Model have statistically different error rate")
    else:
        print("Accept Null Hypotheis")
        print("Conclusion: Model do not have statistically different error rate")


