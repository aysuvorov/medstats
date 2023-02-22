
# utf8

from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
# from sklearn.pipeline import Pipeline
# we call Pipeline from imblearn module:...
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN

# +----------------------------------------------------------------------------

class Validator:
    def __init__(self,
        X_train, 
        y_train,
        num_lst,
        cat_lst,
        predictors,
        random_state
        ):

        self.X_train = X_train
        self.y_train = y_train
        self.random_state = random_state
        self.predictors = predictors
        self.num_lst = num_lst
        self.cat_lst = cat_lst


    def fit(self, clf, X_test, balanced = True):   

        self.X_test = X_test  
        self.clf = clf   

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


        pca_transformer = PCA(n_components=0.95, 
          random_state=self.random_state)

        cv = StratifiedKFold(10, shuffle=True, random_state=self.random_state)
        calibrator = CalibratedClassifierCV(self.clf, 
            method = 'sigmoid', cv=cv)

        smt = ADASYN(random_state=self.random_state)

        if balanced:

            pipe = Pipeline([
                ("preprocessor", preprocessor), 
                ("pca_transformer", pca_transformer),
                ('calibrator', calibrator)
                ])

        else:

            pipe = Pipeline([
                ("preprocessor", preprocessor), 
                ('smt', smt),
                ("pca_transformer", pca_transformer),
                ('calibrator', calibrator)
                ])


        pipe.fit(self.X_train[self.predictors], self.y_train)
        pred = pipe.predict_proba(X_test[self.predictors])[:,1]

        return(pred)