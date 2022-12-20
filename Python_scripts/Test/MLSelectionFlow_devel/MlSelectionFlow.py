import pandas as pd
import yaml

from omegaconf import DictConfig, OmegaConf
import hydra

import workers.optuna_selector as optsel
import workers.feature_selector as fs
import workers.validator as validator

from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

from sklearn.metrics import roc_auc_score, confusion_matrix

from datetime import datetime

# +-----------------------------------------------------------------------------
# DATA LOADING

class SearchFlow:
    def __init__(self):
        pass

    def load_globals(self, cfg):       
        # GLOBALS
        self.X = pd.read_pickle(cfg.PATH_TO_X)
        self.Y = pd.read_pickle(cfg.PATH_TO_Y)
        self.RS = cfg.RANDOM_STATE
        self.OPTUNA_TRIAL_NUMBER = cfg.OPTUNA_TRIAL_NUMBER
        self.TEST_SIZE = cfg.TEST_SIZE
        self.STRATIFY = cfg.STRATIFY
        self.CLASS_WEIGHT = dict(cfg.CLASS_WEIGHT)
        self.N_PREDICTORS = cfg.N_PREDICTORS
        self.BALANCED = cfg.BALANCED
        self.PATH_TO_OUTPUT = cfg.PATH_TO_OUTPUT
        self.PATH_TO_PLOT = cfg.PATH_TO_PLOT
        self.SCORER = cfg.SCORER
        print("Globals loaded...")


    def splitter(self):
        if self.STRATIFY == 1:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(
                    self.X, 
                    self.Y, 
                    test_size=self.TEST_SIZE, 
                    random_state=self.RS, 
                    stratify=self.Y)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(
                    self.X, 
                    self.Y, 
                    test_size=self.TEST_SIZE, 
                    random_state=self.RS)

        self.cat_lst = [x for x in self.X.columns if \
            pd.CategoricalDtype.is_dtype(self.X[x])==True]
        self.sc_lst = [col for col in self.X.columns if \
            col not in self.cat_lst]

        del self.X
        del self.Y
        print("Data loaded and splitted...")


    def feature_selection(self):
        model = LogisticRegression(
            penalty = 'l2', 
            random_state=self.RS,
            class_weight=self.CLASS_WEIGHT)
        
        FS = fs.ShapModelSelector(
            X_train=self.X_train, 
            y_train = self.y_train, 
            cat_lst = [], 
            num_lst = self.sc_lst,
            n_chars = self.N_PREDICTORS)

        FS.run(model, self.PATH_TO_PLOT)
        self.predictors = FS.most_important

        self.X_train = self.X_train[self.predictors]
        self.X_test = self.X_test[self.predictors]

        del self.cat_lst
        del self.sc_lst

        self.cat_lst = [x for x in self.X_train.columns if \
            pd.CategoricalDtype.is_dtype(self.X_train[x])==True]
        self.sc_lst = [col for col in self.X_train.columns if \
            col not in self.cat_lst]

        print("Predictors are defined...:")
        print(self.predictors)

class OptSearch:
    def __init__(self):
        pass

    def search(self, data):
        search = optsel.OptSearch(
            data.X_train, 
            data.y_train, 
            data.sc_lst, 
            [],
            data.OPTUNA_TRIAL_NUMBER, 
            data.RS,
            data.CLASS_WEIGHT,
            data.SCORER)

        if data.BALANCED == 0:
            self.best_params = search.run(False)
        else:
            self.best_params = search.run()

        self.classifier = self.best_params['classifier']
        del self.best_params['classifier']

        print("Best parameters fitted...")


class ModelTraining:
    def __init__(self):
        pass

    def fit(self, data, model):
        if model.classifier == "ELN":
            self._clf = LogisticRegression(**model.best_params, \
                random_state = data.RS)
        elif model.classifier == "RIDGE":
            self._clf = RidgeClassifier(**model.best_params, \
                random_state = data.RS)
        elif model.classifier == "SGD":
            self._clf = SGDClassifier(**model.best_params, \
                random_state = data.RS)
        elif model.classifier == "MLP":
            self._clf = MLPClassifier(**model.best_params, \
                random_state = data.RS)
        elif model.classifier == "ERT":
            self._clf = ExtraTreesClassifier(**model.best_params, \
                random_state = data.RS)
        elif model.classifier == "RFC":
            self._clf = RandomForestClassifier(**model.best_params, \
                random_state = data.RS)
        elif model.classifier == "XGB":
            self._clf = xgb.XGBClassifier(**model.best_params, \
                random_state = data.RS)
        else:
            self._clf = SVC(**model.best_params, \
                random_state = data.RS)      

        runner = validator.Validator(
            data.X_train, 
            data.y_train,
            data.sc_lst,
            [],
            data.predictors,
            data.RS)

        if data.BALANCED == 0:
            self.pred_train = runner.fit(self._clf, data.X_train, False)
            self.pred_test = runner.fit(self._clf, data.X_test, False)
        else:
            self.pred_train = runner.fit(self._clf, self.X_train)
            self.pred_test = runner.fit(self._clf, self.X_test)


class ModelQuality:
        def __init__(self):
            pass

        def fit(self, data, model, train):
            def thres_getter(pred_test, thres):
                return((pred_test > thres).astype(int))

            train_auc = roc_auc_score(data.y_train, train.pred_train)
            test_auc = roc_auc_score(data.y_test, train.pred_test)

            quality_frame = pd.DataFrame()          

            for i in sorted(list(set(train.pred_test))):
                tn, fp, fn, tp = confusion_matrix(
                    data.y_test, 
                    thres_getter(train.pred_test, i)).ravel() 
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
            quality_frame['Youden'] = (youden == max(youden)).astype(int)

            quality_frame = str(quality_frame[['Threshold', 'TN','FP','FN', \
            'TP', 'SE','SP', 'PPV', 'NPV', 'Youden']].to_markdown())

            # Save Output
            text_file = open(data.PATH_TO_OUTPUT + str(datetime.now()) + ".txt", "w")
            text_file.write(f'CLASSIFIER: {model.classifier}')
            text_file.write(f'\nMODEL PARAMS: {model.best_params}')
            text_file.write(f'\n\nPREDICTORS: {data.predictors}')    
            text_file.write(f'\n\nTRAIN\n')
            text_file.write(f'-------------------------------------------------')
            text_file.write(f'\nTrain ROC AUC = {train_auc}')
            text_file.write(f'\n\nTEST\n')
            text_file.write(f'-------------------------------------------------')
            text_file.write(f'\nTest ROC AUC = {test_auc}')
            text_file.write(f'\n\nTest Data Thresholds\n')
            text_file.write(f'\n{quality_frame}')
            text_file.close()

@hydra.main(version_base=None, 
    config_path="../MLSelectionFlow_devel/config", 
    config_name="config")
def main(cfg: DictConfig):
    data = SearchFlow()
    data.load_globals(cfg)
    data.splitter()
    data.feature_selection()

    model = OptSearch()
    model.search(data)

    train = ModelTraining()
    train.fit(data, model)

    quality = ModelQuality()
    quality.fit(data, model, train)

if __name__ == "__main__":
    main()

