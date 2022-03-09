# coding: utf-8

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sma 

from itertools import combinations
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

#+----------------------------------------------
######### Classes ########


class DataFrameLoader(object):
    
    def __init__(
            self, 
            X_fitter: pd.core.frame.DataFrame,
            nodes_lst: list,
            tuples_lst, 
            feature_lst

        )-> None:
        
        self.X_fitter = X_fitter
        self.nodes_lst = nodes_lst
        self.tuples_lst = tuples_lst
        self.feature_lst = feature_lst

    @classmethod
    def load(
        cls, 
        X_fitter: pd.core.frame.DataFrame,
        nodes_lst: list
        ):

        feature_lst = []
        tuples_lst = []

        for i in list(combinations(nodes_lst, r = 2)):
            tuples_lst = tuples_lst + [i]
            feature_lst = feature_lst + [i[0] + '-' + i[1]]

        return DataFrameLoader(X_fitter, nodes_lst,
            tuples_lst, feature_lst)


class Model(object):
    """[summary]

    Args:
        object ([type]): [description]
    """   
    def __init__(
            self, 
            model_lst,
            zanin_loaded_fitted
            ) -> None:
        self.model_lst = model_lst
        self.zanin_loaded_fitted = zanin_loaded_fitted

    @classmethod
    def linear_fit(
        cls,
        zanin_loaded: DataFrameLoader       
        ) -> None:

        model_lst = []

        for i in zanin_loaded.tuples_lst: 
            X = sma.add_constant(zanin_loaded.X_fitter[i[1]])
            y = zanin_loaded.X_fitter[i[0]]
            model_ols = sma.OLS(y, X).fit()
            model_lst = model_lst + [model_ols]

        data = zanin_loaded.X_fitter

        for i, model in zip(zanin_loaded.tuples_lst, model_lst): 
            data[i[0] + '-' + i[1]] = model.predict(
                                sma.add_constant(data[i[1]])) - data[i[0]]

        zanin_loaded_fitted = data.drop(zanin_loaded.nodes_lst, axis=1)
        
        return(Model(model_lst, zanin_loaded_fitted))


class NewData(object):
    def __init__(
            self, 
            newdata_fitted_scaled,
            uin: list
            ) -> None:
        self.newdata = newdata_fitted_scaled
        self.uin = uin


    @classmethod
    def linear_fit(
            cls, 
            zanin_loaded: DataFrameLoader,
            models: Model,
            newdata: pd.core.frame.DataFrame,
            uin: list
        ) -> pd.core.frame.DataFrame:

        # fit new data with regression models
        for i, model in zip(zanin_loaded.tuples_lst, models.model_lst): 
            newdata[i[0] + '-' + i[1]] = model.predict(
                                sma.add_constant(newdata[i[1]])) - newdata[i[0]]
        
        newdata_fitted = newdata.drop(zanin_loaded.nodes_lst, axis=1)

        # scale newdata with healthy data fitted
        cols = zanin_loaded.feature_lst
        sc = StandardScaler()
        sc = sc.fit(models.zanin_loaded_fitted[cols])
        newdata_fitted = newdata_fitted[cols]
        newdata_fitted_scaled = abs(sc.transform(newdata_fitted[cols]))

        return(NewData(newdata_fitted_scaled, uin))


class ZaninDict(object):

    def __init__(
            self, 
            newdata_fitted_scaled: NewData,
            zanin_loaded: DataFrameLoader
        ) -> None:
        
        self.data = newdata_fitted_scaled.newdata
        self.uin = newdata_fitted_scaled.uin
        self.nodes = zanin_loaded.nodes_lst
        self.tuples_lst = zanin_loaded.tuples_lst
        self.feature_lst = zanin_loaded.feature_lst


    @classmethod
    def fit(
            cls,
            newdata_fitted_scaled: NewData,
            zanin_loaded: DataFrameLoader
        ):
 
        graphs = dict()

        for elems, i in zip(
                newdata_fitted_scaled.uin, 
                range(len(newdata_fitted_scaled.newdata))):
            G = nx.Graph()
            G.add_nodes_from(zanin_loaded.nodes_lst)
            for e, w in zip(
                    zanin_loaded.tuples_lst, 
                    newdata_fitted_scaled.newdata[i]):
                G.add_edge(e[0], e[1], weight = w)

            graphs[elems] = G

        return graphs

#+----------------------------------------------
class Zanin(object):
    """ 
    clf = prct.Zanin(X_healthy, nodes_lst)

    clf.fit(X_test.iloc[1:4, :],[132, 133, 134])

    clf.graphs

    """ 
    def __init__(
        self,
        healthy_X,
        nodes
    ) -> None:
        
        self.healthy = DataFrameLoader.load(healthy_X, nodes)
        print('Data for healthy loaded successfully')

        self.models = Model.linear_fit(self.healthy)
        print('Models fitted successfully')
        self.tuples_lst = self.healthy.tuples_lst
        self.feature_lst = self.healthy.feature_lst

    def fit(
            self,
            newdata,
            index
        ):

        self.index = index

        newdata = NewData.linear_fit(
                    self.healthy,
                    self.models,
                    newdata,
                    self.index
                )

        self.graphs = ZaninDict.fit(newdata, self.healthy)
        print('Zanin dict of graphs is ready ...')

        
#+----------------------------------------------
#### Common functions ####


def graph_plotter(
    G, 
    add_edge_labels = True, 
    add_nodes_size = True,
    figsize=(10,10), 
    title='',
    save = False
    ):

    """
    Creates circular graph plot with node size 
    (through weighted degree) and edge width depending
    on edge weights

    Parameters
    ----------
    :G: networkx.classes.graph.Graph

    :add_edge_labels: bool
        edge labels will be shown

    :add_nodes_size: bool
        node size will adjusted according to weighted
        node degree * 50
    
    :figsize: set
        set of plot size for matplotlib plt.subplots() function

    :title: str
        plot title

    Returns
    ----------
    Plot of a graph

    """
    
    pos = nx.circular_layout(G)   

    edge_labels = dict([((n1, n2), d['weight']) for 
        n1, n2, d in G.edges(data=True)])

    degree_dict = dict(G.degree(weight='weight'))

    plt.figure(figsize=figsize)
    plt.title(title)

    nx.draw(G, pos, with_labels = True, node_size=[v * 50 for v in degree_dict.values()])

    if add_edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5)

    if add_nodes_size:
        for edge in G.edges(data='weight'):
            nx.draw_networkx_edges(G, pos, edgelist=[edge], width=edge[2])

    if save:
        plt.savefig(title + '.png')#, bbox_inches="tight", transparent=False)

    plt.show()






