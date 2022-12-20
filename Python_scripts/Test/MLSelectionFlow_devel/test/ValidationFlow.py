# train test validator
# utf8

import numpy as np
import pandas as pd
import json
from sklearn.metrics import roc_auc_score, confusion_matrix

from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb

import sys
sys.path.append('/home/guest/Yandex.Disk/Документы/Документы/Сокольская/workflow')
import raw_predict_collector as prdct

# +----------------------------------------------------------------------------
# LOADING PARAMETERS FROM JSON

f = open('search_results.json',)
parameters = json.load(f)
f.close()

predictors = parameters['predictors']
rs = parameters['seed']
classifier = parameters['classifier']

del parameters['predictors']
del parameters['classifier']
del parameters['seed']

print("Params loaded...")

# +-----------------------------------------------------------------------------
# DATA LOADING

path = '/home/guest/Yandex.Disk/Документы/Документы/Сокольская/'

X = pd.read_pickle(path + 'X_death.pkl')
y = pd.read_pickle(path + 'y_death.pkl')

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=rs, stratify=y)

del X
del y

X_train = X_train[predictors]
X_test = X_test[predictors]

# Лист с типами переменных
cat_lst = [x for x in X_train.columns if \
    pd.CategoricalDtype.is_dtype(X_train[x])==True]
sc_lst = [col for col in X_train.columns if col not in cat_lst]

print("Data loaded...")

# +-----------------------------------------------------------------------------
# Train Model

if classifier == "ELN":
    clf = LogisticRegression(**parameters, random_state = rs)
elif classifier == "RIDGE":
    clf = RidgeClassifier(**parameters, random_state = rs)
elif classifier == "SGD":
    clf = SGDClassifier(**parameters, random_state = rs)
elif classifier == "MLP":
    clf = MLPClassifier(**parameters, random_state = rs)
elif classifier == "ERT":
    clf = ExtraTreesClassifier(**parameters, random_state = rs)
elif classifier == "RFC":
    clf = RandomForestClassifier(**parameters, random_state = rs)
elif classifier == "XGB":
    clf = xgb.XGBClassifier(**parameters, max_depth=12,
        alpha = 0,random_state = rs)
else:
    clf = SVC(**parameters)


PR = prdct.Predictor(X_train, 
        y_train,
        sc_lst,
        [],
        predictors,
        rs)

pred_train = PR.run(clf, X_train, False)
pred_test = PR.run(clf, X_test, False)

# +-----------------------------------------------------------------------------
# Model Quality

train_auc = roc_auc_score(y_train, pred_train)
test_auc = roc_auc_score(y_test, pred_test)

quality_frame = pd.DataFrame()

def thres_getter(pred_test, thres):
    return((pred_test > thres).astype(int))

for i in sorted(list(set(pred_test))):
    tn, fp, fn, tp = confusion_matrix(
        y_test, 
        thres_getter(pred_test, i)).ravel() 
    ppv  = round( tp / (tp+fp), 4)
    npv  = round( tn / (tn+fn), 4)
    se = round( tp / (tp+fn), 4)
    sp = round( tn / (tn+fp), 4)

    quality_frame = quality_frame.append({
        'Threshold': i,
        'TN':tn,
        'FP':fp,
        'FN':fn,
        'TP':tp,
        'SE':se,
        'SP':sp,
        'PPV':ppv,
        'NPV':npv
        }, ignore_index = True)

youden = quality_frame['SE'] + quality_frame['SP'] - 1
quality_frame['Youden'] = (youden == np.max(youden)).astype(int)

quality_frame = str(quality_frame[['Threshold', 'TN','FP','FN', 'TP',\
    'SE','SP', 'PPV', 'NPV', 'Youden']].to_markdown())

# +-----------------------------------------------------------------------------
# Save Output

text_file = open("Output.txt", "w")
text_file.write(f'CLASSIFIER: {classifier}\n\nPREDICTORS: {predictors}')
text_file.write(f'\n\nTRAIN\n')
text_file.write(f'-------------------------------------------------')
text_file.write(f'\nTrain ROC AUC = {train_auc}')
text_file.write(f'\n\nTEST\n')
text_file.write(f'-------------------------------------------------')
text_file.write(f'\nTest ROC AUC = {test_auc}')
text_file.write(f'\n\nTest Data Thresholds\n')
text_file.write(f'\n{quality_frame}')
text_file.close()




