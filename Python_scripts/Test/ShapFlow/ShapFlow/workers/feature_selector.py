# utf 8
# shap feature selector

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# +----------------------------------------------------------------------------

class ShapModelSelector:
    def __init__(self, 
        X_train, 
        y_train, 
        cat_lst, 
        num_lst,
        n_chars = 10
        ):

        self.X_train = X_train 
        self.y_train = y_train 
        self.cat_lst = cat_lst 
        self.num_lst = num_lst
        self.n_chars = n_chars

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

        self.pipe = Pipeline([
            ("preprocessor", preprocessor)
            ])

        self.pipe.fit(self.X_train)
        
        self.features = self.pipe[-1].get_feature_names_out()
        self.features = [f.replace("num__", "").replace("remainder__", \
            "").replace("cat__", "") for f in self.features]
        
        self.observations = self.pipe.fit_transform(X_train)
        

    
    def run(self, algo, path):

        algo.fit(self.observations, self.y_train)

        self.observations = pd.DataFrame(self.observations, 
            columns=self.features)
        for col in self.observations.columns:
            if col in self.num_lst:
                self.observations[col] = self.observations[col].\
                    astype(float)
            else:
                self.observations[col] = self.observations[col].\
                    astype(float) # astype('category')
        explainer = shap.Explainer(algo.predict, self.observations)
        self.shap_values = explainer(self.observations)

        shap.plots.waterfall(self.shap_values[0], max_display=\
            self.n_chars + 1, show=False)  
        plt.savefig(path + "/Plot_SHAP.png", dpi=150, 
            bbox_inches='tight')  

        feature_names = self.shap_values.feature_names
        shap_df = pd.DataFrame(self.shap_values.values, 
            columns=feature_names)
        vals = np.abs(shap_df.values).mean(0)
        shap_importance = pd.DataFrame(list(zip(feature_names, \
            vals)), columns=['col_name', 'feature_importance_vals'])
        shap_importance.sort_values(by=\
            ['feature_importance_vals'], ascending=False, \
                inplace=True)

        self.most_important = \
            list(shap_importance[:self.n_chars]['col_name'])

