# coding: utf-8

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sma 
import networkx as nx

from sklearn.preprocessing import StandardScaler
from seaborn import palettes
from itertools import combinations
from scipy.stats import pearsonr, spearmanr
warnings.filterwarnings("ignore")

############################## Parenclitic graphs ##############################

class DataFrameLoader(object):

    def __init__(
            self
        )-> None:
        pass     

    def fit(self, data: pd.core.frame.DataFrame):
        self.data = data
        self.nodes_lst = data.columns
        self.edges_lst = [i for i in list(combinations(self.nodes_lst, r = 2))]


class Prct(object):

    def __init__(self):
        pass


    def fit(self, data):

        dfl = DataFrameLoader()
        self.data = data
        dfl.fit(self.data)
        
        self.nodes_lst = dfl.nodes_lst
        self.edges_lst = dfl.edges_lst

        model_lst = []

        for i in self.edges_lst: 

            _X = sma.add_constant(self.data[i[1]])
            _y = self.data[i[0]]
            model_ols = sma.OLS(_y, _X).fit()
            model_lst = model_lst + [model_ols]
        self._model_lst = model_lst

        _epsilons = np.array([])

        for i, model in zip(self.edges_lst, self._model_lst): 
            _epsilons = np.append(_epsilons, 
                np.array(model.predict(
                                sma.add_constant(self.data[i[1]])) - self.data[i[0]]), axis =0
            )
        self.epsilons = _epsilons.reshape(int(len(_epsilons)/len(self._model_lst)), len(self._model_lst))      


    def transform(self, newdata, newindex=None):      

        sc = StandardScaler()

        new_dfl = DataFrameLoader()
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
            _new_epsilons = np.append(_new_epsilons, 
                np.array(model.predict(
                                sma.add_constant(self.newdata[i[1]])) - self.newdata[i[0]]), axis =0
            )
        self.new_epsilons = _new_epsilons.reshape(int(len(_new_epsilons)/len(self._model_lst)), len(self._model_lst))

        sc.fit(self.epsilons)
        self.new_epsilons = np.abs(sc.transform(self.new_epsilons))

        if newindex:
            self.index = newindex
        else:
            self.index = np.array(range(self.new_epsilons.shape[0]))
        

    def ntwrk_construct(self, type = 'thres'):

        self.graphs = dict()

        if type == 'weighted':

            for index in [x for x in range(len(self.index))]:              
                G = nx.Graph()
                G.add_nodes_from(self.nodes_lst)
                for edge, position in [[self.edges_lst[i],i] for i in range(len(self.edges_lst))]:
                    print(edge, position, self.new_epsilons[index, position])
                    G.add_edge(edge[0], edge[1], weight = self.new_epsilons[index, position])

                self.graphs[index] = G

        elif type == 'thres':

            for index in [x for x in range(len(self.index))]:              
                G = nx.Graph()
                G.add_nodes_from(self.nodes_lst)
                for edge, position in [[self.edges_lst[i],i] for i in range(len(self.edges_lst))]:
                    if float(self.new_epsilons[index, position]) > 2:
                        G.add_edge(edge[0], edge[1])

                self.graphs[index] = G

        elif type == 'weighted_thres':

            for index in [x for x in range(len(self.index))]:              
                G = nx.Graph()
                G.add_nodes_from(self.nodes_lst)
                for edge, position in [[self.edges_lst[i],i] for i in range(len(self.edges_lst))]:
                    if float(self.new_epsilons[index, position]) > 2:
                        G.add_edge(edge[0], edge[1], weight = float(self.new_epsilons[index, position]))

                self.graphs[index] = G

        else:
            raise KeyboardError('Type must be `weighted`, `thres` or `weighted_thres`')

############################### Synolitic graphs ###############################

class Snltc(object):

    def __init__(self):
        pass


    def fit(self, data, labels, clf=None):



        dfl = DataFrameLoader()
        self.data = data
        dfl.fit(self.data)
        
        self.nodes_lst = dfl.nodes_lst
        self.edges_lst = dfl.edges_lst
        self.labels = labels

        model_lst = []

        for i in self.edges_lst: 
            if clf:
                pass
            else:
                clf = SVC(probability = True, random_state=0)

            model_lst = model_lst + [clf.fit(self.data[[i[0], i[1]]], self.labels)]
        self._model_lst = model_lst


    def transform(self, newdata, newindex=None):      

        new_dfl = DataFrameLoader()
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
            _new_epsilons = np.hstack([_new_epsilons, 
                np.array(model.predict_proba(self.newdata[[i[0], i[1]]])[:,1])])
        
        self.new_epsilons = _new_epsilons.reshape(len(self._model_lst), 
            int(len(_new_epsilons)/len(self._model_lst))).T

        if newindex:
            self.index = newindex
        else:
            self.index = np.array(range(self.new_epsilons.shape[0]))


    def ntwrk_construct(self, type = 'thres'):

        self.graphs = dict()

        if type == 'weighted':

            for index in [x for x in range(len(self.index))]:              
                G = nx.Graph()
                G.add_nodes_from(self.nodes_lst)
                for edge, position in [[self.edges_lst[i],i] for i in range(len(self.edges_lst))]:
                    G.add_edge(edge[0], edge[1], weight = self.new_epsilons[index, position])

                self.graphs[index] = G

        elif type == 'thres':

            for index in [x for x in range(len(self.index))]:              
                G = nx.Graph()
                G.add_nodes_from(self.nodes_lst)
                for edge, position in [[self.edges_lst[i],i] for i in range(len(self.edges_lst))]:
                    if float(self.new_epsilons[index, position]) > .5:
                        G.add_edge(edge[0], edge[1])

                self.graphs[index] = G

        elif type == 'weighted_thres':

            for index in [x for x in range(len(self.index))]:              
                G = nx.Graph()
                G.add_nodes_from(self.nodes_lst)
                for edge, position in [[self.edges_lst[i],i] for i in range(len(self.edges_lst))]:
                    if float(self.new_epsilons[index, position]) > .5:
                        G.add_edge(edge[0], edge[1], weight = float(self.new_epsilons[index, position]))

                self.graphs[index] = G

        else:
            raise KeyboardError('Type must be `weighted`, `thres` or `weighted_thres`')


####################### Alexander Gorban correlation graphs ####################
# 
# Working sample
# Under further development...


# class CorrModel(Model):
#     """[summary]

#     Args:
#         Model ([type]): [description]
#     """    
#     def __init__(self, training_data, features):
#         super().__init__(training_data, features)

#     def fit(self) -> None:
#         """[summary]
#         """        
#         self.weights = []
        
#         for i in self.tuples_lst:
#             par = pearsonr(
#                     self.training_data[i[0]], 
#                     self.training_data[i[1]]
#                     )
#             #par = spearmanr(
#             #        self.training_data[i[0]], 
#             #        self.training_data[i[1]]
#             #        )

#             weight = abs(par[0])
#             #par = pearsonr(
#             #        self.training_data[i[0]], 
#             #        self.training_data[i[1]]
#             #    )
#             #if par[1] < 0.05: 
#             #    weight = abs(par[0])
#             #else: 
#             #    weight = 0

#                 #spearmanr(
#                 #    self.training_data[i[0]],
#                 #    self.training_data[i[1]]
#                 #)[0]

#             #)
#             self.weights = self.weights + [weight]
#         self.weights = np.array(self.weights)


# class CorrGraf(object):
#     """[summary]

#     Args:
#         object ([type]): [description]
#     """    
#     def __init__(
#             self, 
#             networkx_graph: nx.classes.graph.Graph
#         ) -> None:
#         self.graph = networkx_graph

#     @classmethod
#     def fit(
#             cls,
#             features: Features,
#             weights: CorrModel,
#             threshold: float = 0
#         ) -> nx.classes.graph.Graph:

#         G = nx.Graph()
#         G.add_nodes_from(features.nodes)

#         for e, w in zip(features.tuples_lst, weights.weights):
#             if w >= threshold: 
#                 G.add_edge(e[0], e[1], weight = w)
#         return(CorrGraf(G))       

# ##### Classes to make it all work #######                    
# ## For Zanin ##

# class Zanin(object):
#     """
#     ## Zanin algorithm for parenclitic graphs

#     Parameters
#     ----------
#     :X_fitter: pd.core.frame.DataFrame
#         Data frame with features from healthy subjects
#         'id' can be set as an index
    

#     :nodes_lst: list with feature names - they should be in columns
#         Theese features will be used to construct
#         graphs

#     Examples
#     --------
#     first fit regression models from healthy subjects
#     >>> healthy_data_fitted = Zanin(X_healthy, nodes)

#     add new data (old or new, graphs will be created)
#     >>> clf = DataFitter(healthy_data_fitted, some_data_to_make_graphs)

#     get characteristics of graphs(strength, centralities...)
#     >>> clf.create_chars()

#     """  
#     def __init__(
#         self,
#         healthy_X,
#         nodes
#     ) -> None:

#         self.healthy = DataFrameLoader(healthy_X, nodes)
#         print('Data for healthy loaded successfully')

#         self.features = Features(self.healthy)
#         self.features.fit()
#         print('Features computed successfully')
    
#         self.models = Model(self.healthy, self.features)
#         self.models.fit() 
#         print('Models fitted successfully')
#         print('Zanin is ready ...')


# class DataFitter(object):
#     """[summary]

#     Args:
#         object ([type]): [description]
#     """    
#     def __init__(
#         self,
#         dataloader: DataFrameLoader,
#         newdata
#     ) -> None:

#         self.train_data = NewData(newdata, dataloader.healthy, dataloader.features, dataloader.models)
#         self.train_data.fit()
#         print('New data fitted successfully')

#         self.graphs = GraphBasket(self.train_data, dataloader.features)
#         self.graphs.fit()
#         print('Graphs created...')

#     def create_chars(self):

#         self.graphs.compute()
#         return self.graphs.chars

# ## For Gorban ##

# class Gorban(object):
#     """
#     ## Gorban algorithm for correlated graphs
#     Each run makes one correlated graph

#     Parameters
#     ----------
#     :data: pd.core.frame.DataFrame
#         Data frame with features to make graph - time window is passed,
#         where data frame index is usually pd.DatetimeIndex
    

#     :nodes_lst: list with feature names - they should be in columns
#         Theese features will be used to construct graphs

#     :threshold: float - threshold for edge construction. 
#         Represents correlation coeff {0,1}. To construct an edge, 
#         correlated weight must be strictly above threshold, otherwise
#         edge will not be made

#     Examples
#     --------
#     fit correlation coeffs and construct graphs
#     >>> graph_data = DataLoader(time_window_dataframe, nodes, 0.5)

#     get nx.classes.graph.Graph object:
#     >>> G = graph_data.graph

#     """


#     def __init__(
#             self,
#             data: DataFrameLoader,
#             nodes: list,
#             threshold: float,
#             verbose=False
#         ) -> None:
        
#         self.nodes = nodes
#         self.data = DataFrameLoader(data, self.nodes)
#         if verbose:
#             print('Data for healthy loaded successfully')

#         self.features = Features(self.data)
#         self.features.fit()
#         if verbose:
#             print('Features computed successfully')
    
#         self.correlations = CorrModel(self.data, self.features)
#         self.correlations.fit() 

#         corrgraph = CorrGraf.fit(self.features, 
#             self.correlations, threshold)

#         self.graph = corrgraph.graph
#         if verbose:
#             print('Corrs fitted successfully')
#             print('Gorban is ready ...')
        
################################################################################
############################### Common functions ###############################
################################################################################

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
        plt.savefig(title, bbox_inches="tight", transparent=False)
    
    else:
        plt.show()