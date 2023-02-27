from joblib import dump

import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")


class Loader:
    def __init__(self):
        pass     


    def fit(self, data: pd.core.frame.DataFrame):
        self.data = data
        self.nodes_lst = data.columns
        self.edges_lst = [i for i in list(combinations(self.nodes_lst, r = 2))]


class Snltc:
    def __init__(self):
        pass


    def fit(self, data, labels, random_state, clf=None):
        dfl = Loader()
        self.data = data
        dfl.fit(self.data)
        
        self.nodes_lst = dfl.nodes_lst
        self.edges_lst = dfl.edges_lst
        self.labels = labels
        self.random_state = random_state

        model_lst = []

        for i in self.edges_lst: 
            if clf:
                clf = clf
            else:
                # clf = SVC(probability = True, random_state=self.random_state)
                # clf = XGBClassifier(
                #     n_estimators = 500,
                #     random_state = self.random_state,
                #     tree_method = 'hist',
                #     n_jobs = -1, 
                #     enable_categorical = True
                #         )
                clf = LogisticRegression(
                    C = 1,
                    penalty = 'l1',
                    solver = 'saga',
                    random_state = self.random_state)

            model_lst = model_lst + [clf.fit(self.data[[i[0], i[1]]], self.labels)]
        self._model_lst = model_lst


    def transform(self, newdata, newindex=None):      

        new_dfl = Loader()
        self.newdata = newdata

        new_dfl.fit(self.newdata)
        
        self.new_nodes_lst = new_dfl.nodes_lst
        self.new_edges_lst = new_dfl.edges_lst

        assert self.new_nodes_lst.all() == self.nodes_lst.all(), f"Train data and new data have different features-nodes"
        assert self.new_edges_lst == self.edges_lst, f"Train data and new data have different features-edges"

        del self.new_nodes_lst
        del self.new_edges_lst

        _new_epsilons = np.array([])
        
        for i, model in zip(self.edges_lst, self._model_lst): 
            # print(self.newdata[[i[0], i[1]]])
            _new_epsilons = np.hstack([_new_epsilons, 
                np.array(model.predict_proba(self.newdata[[i[0], i[1]]])[:,1])])
                # np.array(model.predict(self.newdata[[i[0], i[1]]]))])
        
        self.new_epsilons = _new_epsilons.reshape(len(self._model_lst), 
            int(len(_new_epsilons)/len(self._model_lst))).T

        if newindex:
            self.index = newindex
        else:
            self.index = np.array(range(self.new_epsilons.shape[0]))


    def ntwrk_construct(self):

        self.graphs = dict()

        for index, s_idx in zip([x for x in range(len(self.index))], [x for x in self.index]):              
            G = nx.Graph()
            G.add_nodes_from(self.nodes_lst)
            for edge, position in [[self.edges_lst[i],i] for i in range(len(self.edges_lst))]:
                G.add_edge(edge[0], edge[1], weight = self.new_epsilons[index, position])

            self.graphs[s_idx] = G


class AdjMatrixComputer:
    def __init__(
        self, 
        processed_data_save_path, 
        adj_mtx_path,
        random_state
        ):
        self.random_state = random_state
        self.adj_mtx_path = adj_mtx_path
        self.X_train = pd.read_pickle(processed_data_save_path + "/X_train.pkl")
        self.y_train = pd.read_pickle(processed_data_save_path + "/y_train.pkl")
        self.X_test = pd.read_pickle(processed_data_save_path + "/X_test.pkl")
        self.y_test = pd.read_pickle(processed_data_save_path + "/y_test.pkl")


    def __call__(self):

        self.X_train.index = [str(x) + 'train' for x in range(len(self.X_train.index))]
        self.X_test.index = [str(x) + 'test' for x in self.X_test.index]

        self.y_train.index = self.X_train.index
        self.y_test.index = self.X_test.index


        X_full = pd.concat([
            self.X_train,
            self.X_test
            ])

        y_full = pd.concat([
            self.y_train,
            self.y_test
        ])

        # Initialise and train synolytic
        snc = Snltc()
        snc.fit(self.X_train, self.y_train, self.random_state)
        
        # Transform all the data
        snc.transform(X_full, newindex=list(X_full.index))
        snc.ntwrk_construct()

        XY_g = pd.DataFrame(
            dict(
                idx = list(snc.graphs.keys()),
                Matrices = [nx.to_numpy_array(x) for x in list(snc.graphs.values())]
            )
        )

        XY_g.index = list(snc.graphs.keys())

        XY_full = XY_g.merge(pd.DataFrame(y_full), left_index=True, right_index=True)
        XY_full = XY_full.drop('idx', 1)

        XY_full.columns = ['Matrices', 'Group']

        XY_train = XY_full.loc[[x for x in XY_full.index if 'train' in x],:]
        XY_test = XY_full.loc[[x for x in XY_full.index if 'test' in x],:]

        X_train_adjmat, X_test_adjmat = [x.drop('Group', 1) for x in [XY_train, XY_test]]
        y_train_adjmat, y_test_adjmat = [x['Group'] for x in [XY_train, XY_test]]

        X_train_adjmat.to_pickle(self.adj_mtx_path + "/X_train.pkl")
        y_train_adjmat.to_pickle(self.adj_mtx_path + "/y_train.pkl")
        
        X_test_adjmat.to_pickle(self.adj_mtx_path + "/X_test.pkl")
        y_test_adjmat.to_pickle(self.adj_mtx_path + "/y_test.pkl")
