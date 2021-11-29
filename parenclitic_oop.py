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
from scipy.stats import pearsonr, spearmanr
warnings.filterwarnings("ignore")

##########################
######### Classes ########
##########################

######### Zanin parenclitic graphs #######

class DataFrameLoader(object):
    """[summary]

    Args:
        object ([type]): [description]
    """    

    def __init__(
            self, 
            X_fitter: pd.core.frame.DataFrame,
            nodes_lst: list
        )-> None:
        """[summary]

        Args:
            X_fitter (pd.core.frame.DataFrame): [description]
            nodes_lst (list): [description]
        """        

        self.X_fitter = X_fitter
        self.nodes_lst = nodes_lst


class Features(object):
    """[summary]

    Args:
        object ([type]): [description]
    """    
    def __init__(
            self, 
            nodes: DataFrameLoader
            ) -> None:
        
        self.nodes = nodes.nodes_lst


    def fit(self):
        feature_lst = []
        tuples_lst = []

        for i in list(combinations(self.nodes, r = 2)):
            tuples_lst = tuples_lst + [i]
            feature_lst = feature_lst + [i[0] + '-' + i[1]]

        self.feature_lst = feature_lst
        self.tuples_lst = tuples_lst


class Model(object):
    """[summary]

    Args:
        object ([type]): [description]
    """    
    def __init__(
            self, 
            training_data: DataFrameLoader,
            features: Features
            ) -> None:
        self.training_data = training_data.X_fitter
        self.tuples_lst = features.tuples_lst
    

    def fit(self) -> None:
        """[summary]
        """        
        model_lst = []

        for i in self.tuples_lst: 

            X = sma.add_constant(self.training_data[i[1]])
            y = self.training_data[i[0]]
            model_ols = sma.OLS(y, X).fit()
            model_lst = model_lst + [model_ols]
        self.model_lst = model_lst


class NewData(object):
    """[summary]

    Args:
        object ([type]): [description]
    """    
    def __init__(
            self, 
            newdata: pd.core.frame.DataFrame,
            zanin_fitted: DataFrameLoader,
            features: Features, 
            models: Model
        ) -> None:
        self.newdata = newdata
        self.nodes = features.nodes
        self.tuples_lst = features.tuples_lst
        self.feature_lst = features.feature_lst
        self.model_lst = models.model_lst
        self.X_fitter = zanin_fitted.X_fitter


    def newdata_fit(
            self, 
            newdata: pd.core.frame.DataFrame,
        ) -> pd.core.frame.DataFrame:
        """[summary]

        Args:
            newdata (pd.core.frame.DataFrame): [description]

        Returns:
            pd.core.frame.DataFrame: [description]
        """        
        for i, model in zip(self.tuples_lst, self.model_lst): 
            newdata[i[0] + '-' + i[1]] = model.predict(
                                sma.add_constant(newdata[i[1]])) - newdata[i[0]]
        return(newdata.drop(self.nodes, axis=1))


    def fit(self) -> None:
        """[summary]
        """        
        self.newdata = self.newdata_fit(self.newdata)
        cols = [x for x in self.feature_lst]
        healthy_data = self.newdata_fit(self.X_fitter)
        sc = StandardScaler()
        sc = sc.fit(healthy_data[cols])
        self.newdata[cols] = abs(sc.transform(self.newdata[cols]))


class GraphBasket(object):
    """[summary]

    Args:
        object ([type]): [description]
    """    
    def __init__(
            self, 
            data: NewData,
            features: Features
        ) -> None:
        
        self.data = data.newdata
        self.index = self.data.index
        self.nodes = features.nodes
        self.tuples_lst = features.tuples_lst
        self.feature_lst = features.feature_lst

    def fit(self)-> None:
        """[summary]
        """        
        self.graphs = dict()

        for elems in self.index:
            G = nx.Graph()
            G.add_nodes_from(self.nodes)
            for e, w in zip(self.tuples_lst, self.feature_lst):
                G.add_edge(e[0], e[1], weight = \
                    float(self.data.loc[elems, w]))

            self.graphs[elems] = G

    def params(self, gg)-> np.array:
        """[summary]

        Args:
            gg ([type]): [description]

        Returns:
            np.array: [description]
        """        
        weight = 'weight'
        norm = False

        clsns = nx.closeness_centrality(gg, distance=weight, wf_improved=norm).values()
        btwnns = nx.betweenness_centrality(gg, weight=weight, normalized=norm).values()
        edge_btwnns = nx.edge_betweenness_centrality(gg, weight=weight, normalized=norm).values()
        pgrnk = nx.pagerank(gg, weight=weight).values()
        eign = nx.eigenvector_centrality(gg, weight=weight).values()
        auth = nx.hits(gg)[1].values()
        strength = dict(gg.degree(weight=weight)).values()

        combo = [btwnns, clsns, edge_btwnns, pgrnk, eign, auth, strength]
        combo = [np.fromiter(x, dtype=float) for x in combo]
        subarray = np.ravel([[np.mean(x), np.median(x), np.std(x), np.max(x), np.max(x)] for x in combo])       
        return(subarray)
    
    
    def compute(self) -> None:
        """[summary]
        """        
        names = ['btwnns', 'clsns', 'edge_btwnns', 'pgrnk', 'eign', 'auth', 'strength']
        func_names = ['_mean', '_median', '_std', '_min', '_max']
        columns = [x+y for x in names for y in func_names]

        mtx = []

        for graph in range(len(self.graphs)):
            gg = [*self.graphs.values()][graph]
            row = self.params(gg)
            mtx = mtx + [row]

        self.chars = pd.DataFrame(
                    data=np.array(mtx), 
                    index=self.index, 
                    columns=columns
                )

######### Gorban correlation graphs #######

class CorrModel(Model):
    """[summary]

    Args:
        Model ([type]): [description]
    """    
    def __init__(self, training_data, features):
        super().__init__(training_data, features)

    def fit(self) -> None:
        """[summary]
        """        
        self.weights = []
        
        for i in self.tuples_lst:
            par = pearsonr(
                    self.training_data[i[0]], 
                    self.training_data[i[1]]
                    )
            #par = spearmanr(
            #        self.training_data[i[0]], 
            #        self.training_data[i[1]]
            #        )

            weight = abs(par[0])
            #par = pearsonr(
            #        self.training_data[i[0]], 
            #        self.training_data[i[1]]
            #    )
            #if par[1] < 0.05: 
            #    weight = abs(par[0])
            #else: 
            #    weight = 0

                #spearmanr(
                #    self.training_data[i[0]],
                #    self.training_data[i[1]]
                #)[0]

            #)
            self.weights = self.weights + [weight]
        self.weights = np.array(self.weights)


class CorrGraf(object):
    """[summary]

    Args:
        object ([type]): [description]
    """    
    def __init__(
            self, 
            networkx_graph: nx.classes.graph.Graph
        ) -> None:
        self.graph = networkx_graph

    @classmethod
    def fit(
            cls,
            features: Features,
            weights: CorrModel,
            threshold: float = 0
        ) -> nx.classes.graph.Graph:

        G = nx.Graph()
        G.add_nodes_from(features.nodes)

        for e, w in zip(features.tuples_lst, weights.weights):
            if w >= threshold: 
                G.add_edge(e[0], e[1], weight = w)
        return(CorrGraf(G))       

##### Classes to make it all work #######                    
## For Zanin ##

class Zanin(object):
    """
    ## Zanin algorithm for parenclitic graphs

    Parameters
    ----------
    :X_fitter: pd.core.frame.DataFrame
        Data frame with features from healthy subjects
        'id' can be set as an index
    

    :nodes_lst: list with feature names - they should be in columns
        Theese features will be used to construct
        graphs

    Examples
    --------
    first fit regression models from healthy subjects
    >>> healthy_data_fitted = Zanin(X_healthy, nodes)

    add new data (old or new, graphs will be created)
    >>> clf = DataFitter(healthy_data_fitted, some_data_to_make_graphs)

    get characteristics of graphs(strength, centralities...)
    >>> clf.create_chars()

    """  
    def __init__(
        self,
        healthy_X,
        nodes
    ) -> None:

        self.healthy = DataFrameLoader(healthy_X, nodes)
        print('Data for healthy loaded successfully')

        self.features = Features(self.healthy)
        self.features.fit()
        print('Features computed successfully')
    
        self.models = Model(self.healthy, self.features)
        self.models.fit() 
        print('Models fitted successfully')
        print('Zanin is ready ...')


class DataFitter(object):
    """[summary]

    Args:
        object ([type]): [description]
    """    
    def __init__(
        self,
        dataloader: DataFrameLoader,
        newdata
    ) -> None:

        self.train_data = NewData(newdata, dataloader.healthy, dataloader.features, dataloader.models)
        self.train_data.fit()
        print('New data fitted successfully')

        self.graphs = GraphBasket(self.train_data, dataloader.features)
        self.graphs.fit()
        print('Graphs created...')

    def create_chars(self):

        self.graphs.compute()
        return self.graphs.chars

## For Gorban ##

class Gorban(object):
    """
    ## Gorban algorithm for correlated graphs
    Each run makes one correlated graph

    Parameters
    ----------
    :data: pd.core.frame.DataFrame
        Data frame with features to make graph - time window is passed,
        where data frame index is usually pd.DatetimeIndex
    

    :nodes_lst: list with feature names - they should be in columns
        Theese features will be used to construct graphs

    :threshold: float - threshold for edge construction. 
        Represents correlation coeff {0,1}. To construct an edge, 
        correlated weight must be strictly above threshold, otherwise
        edge will not be made

    Examples
    --------
    fit correlation coeffs and construct graphs
    >>> graph_data = DataLoader(time_window_dataframe, nodes, 0.5)

    get nx.classes.graph.Graph object:
    >>> G = graph_data.graph

    """


    def __init__(
            self,
            data: DataFrameLoader,
            nodes: list,
            threshold: float,
            verbose=False
        ) -> None:
        
        self.nodes = nodes
        self.data = DataFrameLoader(data, self.nodes)
        if verbose:
            print('Data for healthy loaded successfully')

        self.features = Features(self.data)
        self.features.fit()
        if verbose:
            print('Features computed successfully')
    
        self.correlations = CorrModel(self.data, self.features)
        self.correlations.fit() 

        corrgraph = CorrGraf.fit(self.features, 
            self.correlations, threshold)

        self.graph = corrgraph.graph
        if verbose:
            print('Corrs fitted successfully')
            print('Gorban is ready ...')
        
##########################
#### Common functions ####
##########################

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
