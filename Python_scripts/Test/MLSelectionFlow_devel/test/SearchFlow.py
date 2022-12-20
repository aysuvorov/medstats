# best params selector
# utf8

import pandas as pd
import warnings
import json

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import sys
sys.path.append('/home/guest/Yandex.Disk/GitHub/medstats/src/cxs/test')
import worker_ml_sokolskaya as wkr

sys.path.append('/home/guest/Yandex.Disk/Документы/Документы/Сокольская/workflow')
import feature_selector as fs
import optuna_selector as optsel

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
warnings.filterwarnings("ignore")

# +-----------------------------------------------------------------------------
# DATA LOADING

path = '/home/guest/Yandex.Disk/Документы/Документы/Сокольская/'

X = pd.read_pickle(path + 'X_death.pkl')
y = pd.read_pickle(path + 'y_death.pkl')

print("Select seed:")
rs = int(input())
print("Select number of trials:")
OPTUNA_TRIAL_NUMBER = int(input())

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=rs, stratify=y)

# Лист с типами переменных
cat_lst = [x for x in X.columns if pd.CategoricalDtype.is_dtype(X[x])==True]
sc_lst = [col for col in X.columns if col not in cat_lst]

del X
del y


# +-----------------------------------------------------------------------------
# FEATURE SELECTION BY SHAP

model = LogisticRegression(
    penalty = 'l2', 
    random_state=rs,
    class_weight={0:0.1,
             1:0.9}
            )

FS = fs.ShapModelSelector(
        X_train=X_train, 
        y_train = y_train, 
        cat_lst = [], 
        num_lst = sc_lst,
        n_chars = 10)

FS.run(model)
predictors = FS.most_important
print("Predictors are defined...:")
print(predictors)

# +-----------------------------------------------------------------------------
# OPTUNA MODEL SEARCH

# Select only predictors for model building...
X_train = X_train[predictors]
X_test = X_test[predictors]

del cat_lst
del sc_lst

cat_lst = [x for x in X_train.columns if \
    pd.CategoricalDtype.is_dtype(X_train[x])==True]
sc_lst = [col for col in X_train.columns if col not in cat_lst]


search = optsel.OptSearch(
    X_train, 
    y_train, 
    sc_lst, 
    [],
    OPTUNA_TRIAL_NUMBER, 
    rs,
    {0:0.1, 1:0.9})
best_params = search.run(False)

print("Best parameters...:")
best_params['predictors'] = predictors
best_params['seed'] = rs
print(best_params)

# +-----------------------------------------------------------------------------
# SAVE SEARCH RESULTS

def save():
    print("Save search: print y/n")
    reaction = str(input())
    if reaction == 'y':
        with open("search_results.json", 'w', encoding='utf8') as f: 
            json.dump(best_params, f, ensure_ascii=False)
    elif reaction == 'n':
        pass
    else:
        save()

save()


