"""
Parenclitic functions
"""

import numpy as np
import pandas as pd
import networkx as nx
import statsmodels.api as sma
import warnings
warnings.filterwarnings("ignore")

from itertools import combinations
from sklearn.preprocessing import StandardScaler
#from sklearn.neighbors import KernelDensity


"""
Zanin method with linear regressions
"""

def zanin_preprocess_scaler(X_train, X_test, y_train, nodes_lst, training_index=None):

    sc = StandardScaler()

    sc = sc.fit(X_train.loc[training_index,:][nodes_lst])
    X_train[nodes_lst] = abs(sc.transform(X_train[nodes_lst]))
    
    #if X_test.shape[0] > 0:
    try: 
        if X_test.shape:
            X_test[nodes_lst] = abs(sc.transform(X_test[nodes_lst]))
    except:
        pass

    return(X_train, X_test)


def zanin_model_fitter(X_train, y_train, nodes_lst, training_index=None):

    X_train_regr = X_train.loc[training_index,:]
    feature_lst = []
    model_lst = []

    for i in list(combinations(nodes_lst, r = 2)): 

        a = list(i)
        feature_lst = feature_lst + [a[0] + '-' + a[1]]

        X = sma.add_constant(X_train_regr[a[1]])
        y = X_train_regr[a[0]]
        model_ols = sma.OLS(y, X).fit()

        model_lst = model_lst + [model_ols]     
        
    return(feature_lst, model_lst)


def zanin_fit(df, model_lst, nodes_lst, drop_nodes=True):

    for i, mod in zip(list(combinations(nodes_lst, r = 2)), model_lst): 

        a = list(i)

        df[a[0] + '-' + a[1]] = mod.predict(
                            sma.add_constant(df[a[1]])) - df[a[0]]

    if drop_nodes:
        df = df.drop(nodes_lst, axis=1)

    return(df)


def graph_basket(df_matrix, nodes_lst, tuples_lst, feature_lst, id):
    """
    df_matrix - таблица с фичами, связями и весами для графов, уникальным номером пациента
    nodes_lst - список с иемнами узлов
    tuples_lst - список с кортежами, содержащий связи между 2-мя узлами (отражает будущее ребро)
    feature_lst - список с именами ребер, соответствующий элементам списка с кортежами tuples_lst
    id - колонка с уникальным номером пациента
    """
    G_list = []
    uins_lst = []

    for elems in df_matrix[id]:

        # создание графа
        G = nx.Graph()
        # Создание узлов
        G.add_nodes_from(nodes_lst)
        # Создание каждого ребра из tuples_lst
        for e, w in zip(tuples_lst, feature_lst):
            G.add_edge(e[0], e[1], weight = float(df_matrix[df_matrix[id]==elems][w]))

        # Заполнение списков с пациентами и с графами:
        uins_lst = uins_lst + [elems]
        G_list = G_list + [G]

    return(uins_lst, G_list)


def graph_multi_centralities(uins_lst, G_list):

    btwnns = []
    clsns = []
    edge_btwnns = []
    pgrnk = []
    eign = []
    auth = []
    strength = []

    future_data = {}

    for gg in G_list:
        clsns = clsns + [list(nx.closeness_centrality(gg, distance='weight', wf_improved=False).values())]
        btwnns = btwnns + [list(nx.betweenness_centrality(gg, weight='weight', normalized=False).values())]
        edge_btwnns = edge_btwnns + [list(nx.edge_betweenness_centrality(gg, weight='weight', normalized=False).values())]
        pgrnk = pgrnk + [list(nx.pagerank(gg, weight='weight').values())]
        eign = eign + [list(nx.eigenvector_centrality(gg, weight='weight').values())]
        auth = auth + [list(nx.hits(gg)[1].values())]
        strength = strength + [list(dict(gg.degree(weight='weight')).values())]

    for cent, name in zip([btwnns, clsns, edge_btwnns, pgrnk, eign, auth, strength], \
        ['btwnns', 'clsns', 'edge_btwnns', 'pgrnk', 'eign', 'auth', 'strength']):

        future_data[name + '_mean']=np.array(cent).mean(axis=1)
        future_data[name + '_sd']=np.array(cent).std(axis=1)
        future_data[name + '_min']=np.array(cent).min(axis=1)
        future_data[name + '_max']=np.array(cent).max(axis=1)

    future_data['id'] = uins_lst

    return(pd.DataFrame(future_data))


def zanin_X_train_fit(X_train, y_train, nodes_lst, training_index, pre_scale=False):

    if pre_scale:
        X_train, _ = zanin_preprocess_scaler(
                        X_train, 
                        None, 
                        y_train, 
                        nodes_lst, 
                        training_index)

    feature_lst, model_lst = zanin_model_fitter(
                                X_train, 
                                y_train, 
                                nodes_lst,
                                training_index)

    tuples_lst = [tuple(i.split('-')) for i in feature_lst]

    fitter = zanin_fit(X_train, model_lst, nodes_lst, drop_nodes=False)

    return(fitter, feature_lst, model_lst, tuples_lst)
    

def zanin_transform(
        fitter, 
        X_test, 
        nodes_lst, 
        training_index, 
        feature_lst, 
        model_lst, 
        tuples_lst,
        pre_scale=False):

    if pre_scale:
        _, X_test = zanin_preprocess_scaler(
            X_train, 
            X_test, 
            y_train, 
            nodes_lst,
            training_index)

    X_test = zanin_fit(X_test, model_lst, nodes_lst)


    _, X_test = zanin_preprocess_scaler(
                    X_train[feature_lst], 
                    X_test, 
                    y_train, 
                    feature_lst,
                    training_index)

    X_test['id'] = X_test.index

    uins_lst, G_list = graph_basket(X_test, nodes_lst, tuples_lst, feature_lst, id='id')

    graph_chars = graph_multi_centralities(uins_lst, G_list)

    return graph_chars



"""
KDE method


def kde_fitter(x,y, metric='euclidean', kernel='gaussian', algorithm='ball_tree', band_method='silverman'):

    if type(x) == np.ndarray:
        pass
    else:
        x = x.to_numpy()

    if type(y) == np.ndarray:
        pass
    else:
        y = y.to_numpy()

    xy = np.vstack([x,y])

    d = xy.shape[0]
    n = xy.shape[1]

    # best bandwidth estimation
    if 'silverman':
        bw = (n * (d + 2) / 4.)**(-1. / (d + 4)) # silverman
    else:
        bw = n**(-1./(d+4)) # scott

    # kde fit
    kde = KernelDensity(bandwidth=bw, metric='euclidean', kernel='gaussian')
    kde.fit(xy.T)

    # return fitted kde
    return(kde.fit(xy.T), bw)

def kde_compute(clf, x, y):

    return(np.exp(clf.score_samples(np.vstack([x,y]).T)))
    


def kde_feature_creator(X_train, X_test, y_train, nodes_lst, scale_data=True):

    X_train_regr = X_train.loc[y_train[y_train == 0].index,:]
    feature_lst = []

    for i in list(combinations(nodes_lst, r = 2)): 

        a = list(i)
        clf, _ = kde_fitter(X_train_regr[a[1]],X_train_regr[a[0]])
        #clf, _ = kde_fitter(X_train[a[1]],X_train[a[0]])

        for df in [X_train, X_test]:
            df[a[0] + '-' + a[1]] = kde_compute(clf, df[a[1]], df[a[0]]) 
             

        feature_lst = feature_lst + [a[0] + '-' + a[1]]

    if scale_data:

        sc = StandardScaler()
        X_train_sc = X_train.loc[y_train[y_train == 0].index,:]
        X_train_sc[feature_lst] = abs(sc.fit_transform(X_train_sc[feature_lst]))
        X_train[feature_lst] = abs(sc.transform(X_train[feature_lst]))
        X_test[feature_lst] = abs(sc.transform(X_test[feature_lst]))

    return(X_train[feature_lst], X_test[feature_lst], feature_lst)



def graph_basket(df_matrix, nodes_lst, tuples_lst, feature_lst, id):
    """
"""     df_matrix - таблица с фичами, связями и весами для графов, уникальным номером пациента
    nodes_lst - список с иемнами узлов
    tuples_lst - список с кортежами, содержащий связи между 2-мя узлами (отражает будущее ребро)
    feature_lst - список с именами ребер, соответствующий элементам списка с кортежами tuples_lst
    id - колонка с уникальным номером пациента """
"""
    G_list = []
    uins_lst = []

    for elems in df_matrix[id]:

        # создание графа
        G = nx.Graph()
        # Создание узлов
        G.add_nodes_from(nodes_lst)
        # Создание каждого ребра из tuples_lst
        for e, w in zip(tuples_lst, feature_lst):
            G.add_edge(e[0], e[1], weight = float(df_matrix[df_matrix[id]==elems][w]))

        # Заполнение списков с пациентами и с графами:
        uins_lst = uins_lst + [elems]
        G_list = G_list + [G]

    return(uins_lst, G_list)
    

def graph_multi_centralities(uins_lst, G_list):

    btwnns = []
    clsns = []
    edge_btwnns = []
    pgrnk = []
    eign = []
    auth = []
    strength = []

    future_data = {}

    for gg in G_list:
        clsns = clsns + [list(nx.closeness_centrality(gg, distance='weight', wf_improved=True).values())]
        btwnns = btwnns + [list(nx.betweenness_centrality(gg, weight='weight', normalized=False).values())]
        edge_btwnns = edge_btwnns + [list(nx.edge_betweenness_centrality(gg, weight='weight', normalized=False).values())]
        pgrnk = pgrnk + [list(nx.pagerank(gg, weight='weight').values())]
        eign = eign + [list(nx.eigenvector_centrality(gg, weight='weight').values())]
        auth = auth + [list(nx.hits(gg)[1].values())]
        strength = strength + [list(dict(gg.degree(weight='weight')).values())]

    for cent, name in zip([btwnns, clsns, edge_btwnns, pgrnk, eign, auth, strength], \
        ['btwnns', 'clsns', 'edge_btwnns', 'pgrnk', 'eign', 'auth', 'strength']):

        future_data[name + '_mean']=np.array(cent).mean(axis=1)
        future_data[name + '_sd']=np.array(cent).std(axis=1)
        future_data[name + '_min']=np.array(cent).min(axis=1)
        future_data[name + '_max']=np.array(cent).max(axis=1)

    future_data['id'] = uins_lst

    return(pd.DataFrame(future_data))


def kde_data_preprocessor(data_frame, nodes_lst, test_train_var='label', test_train_labels=['TRAIN', 'TEST'], unit='id', outcome='score'):
    """
"""     Zanin Data Preprocessor

    data_frame - original dataframe 
    nodes_lst - list of features to be processed, also - future nodes
    test_train_var - variable name with test/train labels
    test_train_labels - test/train labels in test_train_var, as list, first - train, second - test
    unit - variable with patients id
    outcome - variable with predictor class  """
    
"""


    # first test/train split
    X_train = data_frame[data_frame[test_train_var] == test_train_labels[0]].set_index(unit).drop([test_train_var, outcome], axis=1)
    X_test = data_frame[data_frame[test_train_var] == test_train_labels[1]].set_index(unit).drop([test_train_var, outcome], axis=1)
    y_train = data_frame[data_frame[test_train_var] == test_train_labels[0]].set_index(unit)[outcome]

    # transform/scale data
    X_train, X_test = zanin_scaler_transformer(X_train, X_test, y_train, nodes_lst)

    # creation of regressions, weights, feature_lst and tuples_lst
    X_train, X_test, feature_lst = kde_feature_creator(X_train, X_test, y_train, nodes_lst)
    tuples_lst = [tuple(i.split('-')) for i in feature_lst]

    # create full dataframe with new features
    X_train = X_train.reset_index()
    X_test = X_test.reset_index()
    full_data = X_train.append(X_test)

    # list of graphs creation
    uins_lst, G_list = graph_basket(full_data, nodes_lst, tuples_lst, feature_lst, unit)
    # chars dataframe creation
    new_data = graph_multi_centralities(uins_lst, G_list).merge(data_frame[[unit, test_train_var, outcome]], on=unit)

    # Final train / test split
    X_train = new_data[new_data[test_train_var] == test_train_labels[0]].set_index(unit).drop([test_train_var, outcome], axis=1)
    X_test = new_data[new_data[test_train_var] == test_train_labels[1]].set_index(unit).drop([test_train_var, outcome], axis=1)
    y_train = new_data[new_data[test_train_var] == test_train_labels[0]].set_index(unit)[outcome]
    y_test = new_data[new_data[test_train_var] == test_train_labels[1]].set_index(unit)[outcome]

    return(X_train, X_test, y_train, y_test)

"""
