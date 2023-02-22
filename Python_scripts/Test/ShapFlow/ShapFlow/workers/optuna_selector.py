import optuna
from optuna.samplers import TPESampler

from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import StratifiedKFold

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
warnings.filterwarnings("ignore")

# +----------------------------------------------------------------------------

class OptSearch:
    def __init__(self,
        X_train, 
        y_train,
        num_lst,
        cat_lst,
        n_trials,
        random_state,
        class_weight,
        scorer_param
        ):

        self.X_train = X_train
        self.y_train = y_train
        self.n_trials = n_trials
        self.random_state = random_state
        self.num_lst = num_lst
        self.cat_lst = cat_lst
        self.class_weight = class_weight
        self.scorer_param = scorer_param


    def run(self, balanced = True):

        def objective(trial):

            classifier = trial.suggest_categorical("classifier", 
                ["ELN", "RIDGE", "SGD", "MLP", "ERT", "RFC", "XGB", "SVC"])
                
            if classifier == "ELN":

                c_const = int(trial.suggest_categorical('C', \
                    [1, 10, 100, 1000]))
                l1_ratio = int(trial.suggest_categorical('l1_ratio', \
                    [0.3, 0.5, 0.7]))
                penalty = trial.suggest_categorical('penalty', \
                    ['elasticnet', 'elasticnet'])
                solver = trial.suggest_categorical('solver', \
                    ['saga', 'saga'])
                clf = LogisticRegression(
                    solver = solver,
                    C=c_const, 
                    penalty = penalty,
                    class_weight = self.class_weight,
                    l1_ratio = l1_ratio,
                    random_state=self.random_state)

            elif classifier == "RIDGE":

                ridge_alpha = int(trial.suggest_categorical('alpha', \
                    [1, 100, 1000]))
                clf = RidgeClassifier(
                    alpha=ridge_alpha,
                    class_weight = self.class_weight,
                    random_state=self.random_state)

            elif classifier == "SGD":
                sgd_loss = trial.suggest_categorical('loss', \
                    ['hinge', 'huber', 'log_loss'])
                max_iter = int(trial.suggest_categorical('max_iter', \
                    [1000, 1000]))
                learning_rate = trial.suggest_categorical('learning_rate', \
                    ['optimal', 'optimal'])
                shuffle= trial.suggest_categorical('shuffle', \
                    [True, True])

                clf = SGDClassifier(
                        loss=sgd_loss, 
                        max_iter=max_iter, 
                        shuffle=shuffle, 
                        n_jobs=-1,
                        early_stopping=False, 
                        learning_rate = learning_rate,
                        random_state=self.random_state,
                        class_weight = self.class_weight,
                        average = True
                        )

            elif classifier == 'MLP':

                hidden_layer_sizes = trial.suggest_categorical(
                    'hidden_layer_sizes', 
                        [(90, 90, 30),
                        (100, 50, 50)])
                activation = trial.suggest_categorical('activation', \
                    ['logistic', 'tanh', 'relu'])        

                clf = MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=activation,
                    random_state=self.random_state)


            elif classifier == "ERT":
                
                ert_max_depth = int(trial.suggest_categorical('max_depth', \
                    [15, 18, 22]))
                n_estimators=int(trial.suggest_categorical('n_estimators', \
                    [1000, 1000]))
                clf = ExtraTreesClassifier(
                    n_estimators=n_estimators,
                    max_depth=ert_max_depth,
                    n_jobs=-1,
                    class_weight = self.class_weight,
                    random_state=self.random_state)

            elif classifier == "RFC":
                
                ert_max_depth = int(trial.suggest_categorical('max_depth', \
                    [15, 18, 22]))
                n_estimators=int(trial.suggest_categorical('n_estimators', \
                    [1000, 1000]))
                clf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=ert_max_depth,
                    n_jobs=-1,
                    class_weight = self.class_weight,
                    random_state=self.random_state)

            elif classifier == "XGB":
                
                reg_lambda = int(trial.suggest_categorical('lambda', \
                    [1, 100, 1000]))
                max_depth=int(trial.suggest_categorical('max_depth', \
                    [15, 18, 22]))
                # alpha = float(trial.suggest_categorical('alpha', \
                #     [0, 0]))
                enable_categorical = trial.suggest_categorical(
                    'enable_categorical', [True, True])
                n_estimators = int(trial.suggest_categorical('n_estimators', \
                    [1000, 1000]))

                clf = xgb.XGBClassifier(
                            verbosity=0,
                            n_estimators = n_estimators,
                            reg_lambda=reg_lambda,
                            nthread=-1,
                            max_depth=max_depth,
                            alpha = 0,
                            use_label_encoder=False,
                            # enable_categorical = enable_categorical,
                            random_state=self.random_state)

            else:

                kernel = trial.suggest_categorical('kernel', ['poly', 'rbf'])
                C = float(trial.suggest_categorical('C', [1, 10, 100, 1000]))
                clf = SVC(
                    C=C,
                    kernel=kernel,
                    class_weight = self.class_weight,
                    random_state=self.random_state)


            numeric_transformer = Pipeline(steps=[('scaler', 
                    StandardScaler())])
            categ_transformer = Pipeline(steps=[('oh_encoder', 
                    OneHotEncoder())])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, self.num_lst),
                    ("cat", categ_transformer, self.cat_lst)
                    ],
                    remainder='passthrough'
                    )

            smote = ADASYN(random_state=self.random_state)

            pca_transformer = PCA(n_components=0.95, 
                random_state=self.random_state)

            if balanced:
                pipe = Pipeline([
                    ('preprocessor', preprocessor), 
                    ("pca_transformer", pca_transformer),
                    ('clf', clf)
                    ])
            else:
                pipe = Pipeline([
                    ('preprocessor', preprocessor), 
                    ("smote", smote),
                    ("pca_transformer", pca_transformer),
                    ('clf', clf)
                    ])

            cv = StratifiedKFold(5, shuffle=True, 
                random_state=self.random_state)

            return cross_val_score(
                pipe, self.X_train, self.y_train, n_jobs=-1, 
                cv=cv, scoring = self.scorer_param).mean()

        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)

        trial = study.best_trial

        # ## Лучшие параметры
        best_dict_params = trial.params
        
        return(best_dict_params)