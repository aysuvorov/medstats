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


        sens = tp/(tp + fn)
        spec = tn/(tn + fp)
        npv = tn/(tn + fn)
        ppv = tp/(tp + fp)
        return(sens, spec, npv, ppv)

    # Point estimates:
    tn, fp, fn, tp = metrics.confusion_matrix(Y[:,0], Y[:,1]).ravel()
    sens, spec, npv, ppv = _quality_point_est(Y[:,0], Y[:,1])

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

        sens_ci, spec_ci, npv_ci, ppv_ci = \
            [np.percentile(vals[:,i][~np.isnan(vals)[:,i]], [2.5, 97.5]) for i in range(4)]

        sens, spec, npv, ppv, sens_ci, spec_ci, npv_ci, ppv_ci = \
            [np.round(x, 3) for x in [sens, spec, npv, ppv, sens_ci, \
                spec_ci, npv_ci, ppv_ci]]

        # Making final data frame
        names = ['tn', 'fp', 'fn', 'tp', 'sens', 'spec', 'npv', 'ppv']
        pe = [tn, fp, fn, tp, sens, spec, npv, ppv]
        ci = ['-', '-', '-', '-', sens_ci, spec_ci, npv_ci, ppv_ci]

        if data_frame:
            table = pd.DataFrame({'Chars': names, 'Point_est': pe, '95%CI': ci})
            return table
        else:
            return[tn, fp, fn, tp, sens, spec, npv, ppv,\
                sens_ci, spec_ci, npv_ci, ppv_ci]

    else:

        sens, spec, npv, ppv = \
            [np.round(x, 3) for x in [sens, spec, npv, ppv]]

        return[tn, fp, fn, tp, sens, spec, npv, ppv]

def ModPerf_AUC(real, pred, num_resamples = 1000, ci95 = True):

    if type(real) != np.ndarray:
        real = np.array(real)

    if type(pred) != np.ndarray:
        pred = np.array(pred)

    Y = np.array([real, pred]).T
    Y = Y[~np.any(np.isnan(Y), axis=1)]

    AUC = np.round(metrics.roc_auc_score(Y[:,0], Y[:,1]), 3)

    if AUC < 0.5:
        # print('Inverted real class!!!')
        
        Y[:,0] = np.abs(Y[:,0] - 1)
        AUC = np.round(metrics.roc_auc_score(Y[:,0], Y[:,1]), 3)

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

        auc_l, auc_h = [np.round(x, 3) for x in np.percentile(vals, [2.5, 97.5])]

        return [AUC, auc_l, auc_h]

    else:

        return np.round(AUC, 3)


def ModPerf_thresholds(real, pred_probs):

    Y = np.array([real, pred_probs]).T
    Y = Y[~np.any(np.isnan(Y), axis=1)]

    g = pd.DataFrame()

    for score in sorted(set(Y[:,1])):
        tab = pd.Series(
            ModPerf_Binary(real = Y[:,0], \
            pred = np.array((Y[:,1] >= score).astype(int)), ci95 = False))
        g = pd.concat([g, tab], axis=1)

    g.columns = sorted(set(Y[:,1]))
    g = g.T.reset_index()
    g.rename(columns={'index':'Thres'}, inplace=True)
    g.columns = ['Thres','tn', 'fp', 'fn', 'tp', 'sens', 'spec', \
        'npv', 'ppv']
    g['Youden'] = g.sens + g.spec - 1

    g = g[['Thres','tn', 'fp', 'fn', 'tp', 'sens', 'spec', \
        'npv', 'ppv', 'Youden']]

    best_thres = g.sort_values('Youden', ascending = False)['Thres'].reset_index(drop = True)[0]

    pred_bin = pred_probs > best_thres

    return [g, best_thres, pred_bin]


def ModPerfBinProbsCalculator(real, pred_probs):
    AUC, auc_l, auc_h = ModPerf_AUC(real, pred_probs)
    g, _, pred_bin = ModPerf_thresholds(real, pred_probs)
    table = ModPerf_Binary(real, pred_bin)

    return AUC, auc_l, auc_h, table, g.drop('Youden', axis='columns')


def output_model_log(real, pred, f_path, best_dict_params):
    text_file = open(f_path + ".txt", "w")
    auc_med, auc_l, auc_h, table, g = ModPerfBinProbsCalculator(real, pred)
    text_file.write(f'\nMODEL PARAMS: {best_dict_params}')
    text_file.write(f'\n\nTest data ROC AUC score: {auc_med} [{auc_l}; [{auc_h}]')
    text_file.write(f'\n\nBest Threshold\n')
    text_file.write(f'{table}')
    text_file.write(f'\n\nThresholds\n')
    text_file.write(f'{g.to_markdown()}')
    text_file.close()

    with open(f_path + '.npy', 'wb') as f:
        np.save(f, np.array([real, pred]))
    f.close()

# def ModPerf_AUC(real, pred, num_resamples = 1000, ci95 = True):

#     if type(real) != np.ndarray:
#         real = np.array(real)

#     if type(pred) != np.ndarray:
#         pred = np.array(pred)

#     Y = np.array([real, pred]).T
#     Y = Y[~np.any(np.isnan(Y), axis=1)]

#     AUC = metrics.roc_auc_score(Y[:,0], Y[:,1])

#     if AUC < 0.5:
#         # print('Inverted real class!!!')
        
#         Y[:,0] = np.abs(Y[:,0] - 1)
#         AUC = metrics.roc_auc_score(Y[:,0], Y[:,1])

#     if ci95:

#         # Bootsrap CI-s
#         random.seed(0)
#         vals = []
#         for i in range(num_resamples):
#             a = np.array(random.choices(Y, k=len(Y)))
#             try:
#                 vals = vals + [metrics.roc_auc_score(
#                     a[:,0], 
#                     a[:,1]
#                 )]
#             except:
#                 pass

#         vals = np.array(vals)

#         auc_ci = np.percentile(vals, [2.5, 97.5])

#         return[np.round(x, 3) for x in [AUC, auc_ci]]

#     else:

#         return np.round(AUC, 3)



# def ModPerf_Binary(real, pred, num_resamples = 1000, data_frame = True, ci95 = True):

#     if type(real) != np.ndarray:
#         real = np.array(real)

#     if type(pred) != np.ndarray:
#         pred = np.array(pred)

#     Y = np.array([real, pred]).T
#     Y = Y[~np.any(np.isnan(Y), axis=1)]


#     def _quality_point_est(reals, preds):
#         """AI is creating summary for quality_point_est

#         Args:
#             real (np.array): [description]
#             pred (np.array): [description]
#         """
#         tn, fp, fn, tp = metrics.confusion_matrix(reals, preds).ravel()

#         auc = metrics.roc_auc_score(reals, preds)
#         sens = tp/(tp + fn)
#         spec = tn/(tn + fp)
#         npv = tn/(tn + fn)
#         ppv = tp/(tp + fp)
#         return(auc, sens, spec, npv, ppv)

#     # Point estimates:
#     tn, fp, fn, tp = metrics.confusion_matrix(Y[:,0], Y[:,1]).ravel()
#     auc, sens, spec, npv, ppv = _quality_point_est(Y[:,0], Y[:,1])

#     if ci95:

#         # Bootsrap CI-s
#         random.seed(0)
#         vals = []
#         for i in range(num_resamples):
#             a = np.array(random.choices(Y, k=len(Y)))
#             try:
#                 vals = vals + [list(_quality_point_est(
#                     a[:,0], 
#                     a[:,1]
#                 ))]
#             except:
#                 pass

#         vals = np.array(vals)

#         auc_ci, sens_ci, spec_ci, npv_ci, ppv_ci = \
#             [np.percentile(vals[:,i][~np.isnan(vals)[:,i]], [2.5, 97.5]) for i in range(5)]

#         auc, sens, spec, npv, ppv, auc_ci, sens_ci, spec_ci, npv_ci, ppv_ci = \
#             [np.round(x, 3) for x in [auc, sens, spec, npv, ppv, auc_ci, sens_ci, \
#                 spec_ci, npv_ci, ppv_ci]]

#         # Making final data frame
#         names = ['tn', 'fp', 'fn', 'tp', 'auc', 'sens', 'spec', 'npv', 'ppv']
#         pe = [tn, fp, fn, tp, auc, sens, spec, npv, ppv]
#         ci = ['-', '-', '-', '-', auc_ci, sens_ci, spec_ci, npv_ci, ppv_ci]

#         if data_frame:
#             table = pd.DataFrame({'Chars': names, 'Point_est': pe, '95%CI': ci})
#             return table
#         else:
#             return[tn, fp, fn, tp, auc, sens, spec, npv, ppv,\
#                 auc_ci, sens_ci, spec_ci, npv_ci, ppv_ci]

#     else:

#         auc, sens, spec, npv, ppv = \
#             [np.round(x, 3) for x in [auc, sens, spec, npv, ppv]]

#         return[tn, fp, fn, tp, auc, sens, spec, npv, ppv]
      

# # +-----------------------------------------------------------------------------

# # ROC AUC plots for binary classification and scores, and for 
# # multiclass classifiers

# def ROCPlotter_Binary(real, pred, title=None, plot=True, save_name=''):

#     if metrics.roc_auc_score(real, pred) < 0.5:
#         real = np.abs(real-1)
#         print('Inverted class!')

#     thresholds = np.sort(pred, axis=None)

#     ROC = np.zeros((len(real) + 2, 2))
#     ROC = np.append(ROC, [[1,1]],axis = 0)[::-1]

#     for i in range(len(real)):
#         t = thresholds[i]

#         # Classifier / label agree and disagreements for current threshold.
#         TP_t = np.logical_and( pred > t, real==1 ).sum()
#         TN_t = np.logical_and( pred <=t, real==0 ).sum()
#         FP_t = np.logical_and( pred > t, real==0 ).sum()
#         FN_t = np.logical_and( pred <=t, real==1 ).sum()

#         # Compute false positive rate for current threshold.
#         FPR_t = FP_t / float(FP_t + TN_t)
#         ROC[i+1,0] = FPR_t

#         # Compute true  positive rate for current threshold.
#         TPR_t = TP_t / float(TP_t + FN_t)
#         ROC[i+1,1] = TPR_t

#     fig = plt.figure(figsize=(6,6))
#     plt.plot(ROC[:,0], ROC[:,1], lw=2)
#     plt.plot([[0,0], [1,1]], linestyle='--', c = 'gray')
#     plt.xlim(-0.1,1.1)
#     plt.ylim(-0.1,1.1)
#     plt.xlabel('False positive rate')
#     plt.ylabel('True positive rate')
#     plt.grid()

#     if title:
#         plt.title(title + '\n\nAUC = %.3f'% metrics.roc_auc_score(real, pred))
#     else:
#         plt.title('ROC curve, AUC = %.3f'% metrics.roc_auc_score(real, pred))
#     if plot:
#         plt.show()

#     if save_name != '':
#         fig.savefig(save_name + '.png', facecolor='white', transparent=False)


# def ModPerf_thresholds(real, pred):

#     if type(real) != np.ndarray:
#         real = np.array(real)

#     if type(pred) != np.ndarray:
#         pred = np.array(pred)

#     Y = np.array([real, pred]).T
#     Y = Y[~np.any(np.isnan(Y), axis=1)]

#     g = pd.DataFrame()

#     for score in sorted(set(Y[:,1])):
#         tab = pd.Series(ModPerf_Binary(Y[:,0], \
#             np.array((Y[:,1] >= score).astype(int)), ci95 = False))
#         g = pd.concat([g, tab], axis=1)

#     g.columns = sorted(set(Y[:,1]))
#     g = g.T.reset_index()
#     g.rename(columns={'index':'Thres'}, inplace=True)
#     g.columns = ['Thres','tn', 'fp', 'fn', 'tp', 'auc', 'sens', 'spec', \
#         'npv', 'ppv']
#     g = g.drop('auc',1)
#     return g