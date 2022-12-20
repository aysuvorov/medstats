from sklearn.metrics import pairwise_distances
from itertools import combinations

class Point:
    def __init__(self, x, y, metric = 'euclidean'):
        
        self.x = np.asarray(x).reshape(-1, 1)
        self.y = np.asarray(y).reshape(-1, 1)
        self.metric = metric
    
        try:
            self.distance = float(pairwise_distances(self.x, self.y, 
                    metric = self.metric)
                    )
        except:
            self.distance = float('inf')

    def __str__(self):
        return f'x: {float(self.x)} \n' \
                f'y: {float(self.y)} \n' \
                f'dist: {float(self.distance)} \n' \
                f'metric: {self.metric} \n'


class PatientGraph:
    def __init__(
        self,
        data_series: pd.core.series.Series,
        node_lst: list,
        metric: str
        ):

        self.data = data_series
        self.nodes = node_lst
        self._metric = metric

    def fit(self):
        tuples_lst = []
        for i in list(combinations(self.data.index, r = 2)):
            tuples_lst = tuples_lst + [i]

        self.graph = nx.Graph()
        for edge in tuples_lst:
            self.graph.add_edge(edge[0], edge[1], weight = 
                Point(self.data.loc[edge[0]], 
                    self.data.loc[edge[1]], self._metric).distance)
        return self.graph


def data_transformer(dataframe, metric = 'euclidean'):
    
    train_data = pd.DataFrame()

    for idx in dataframe.index:
        G = PatientGraph(dataframe.loc[idx, :], 
            predictors, metric).fit()
        train_data = train_data.append(
            {'degree': 
                np.max(np.fromiter(dict(G.degree(weight='weight')).values(), 
                dtype=float)),
            'pgrnk': np.max(np.fromiter(nx.pagerank(G, 
                weight='weight').values(), 
                dtype=float)),
            'l_1': np.linalg.norm(
                np.asarray(
                    nx.to_numpy_matrix(G, weight='weight')).ravel(), 
                    ord = 1),
            'l_2': np.linalg.norm(
                np.asarray(
                    nx.to_numpy_matrix(G, weight='weight')).ravel(), 
                    ord = 2),
            'btwn_cent': np.max(np.fromiter(nx.betweenness_centrality(G, 
                weight='weight', 
                normalized=False).values(), 
                dtype=float))        
            }, ignore_index=True)

    train_data.index = dataframe.index
    return train_data