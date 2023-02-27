import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import joblib
import itertools

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.svm import SVC
from sklearn.utils import resample

import scipy.stats as st
import networkx as nx

from seaborn import palettes
from itertools import combinations
from numpy import linalg as LA
warnings.filterwarnings("ignore")

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.append('/home/guest/Документы/medstats/Python_scripts')
import ModPerf as mdp


# Random state
RS = 101 

SELECTED_VARS = ['Age', 'QRS11energy', 'TA', 'Tfi', 'J80A', 'DiabMell', 'TE1', 'RA', 'RoffsF', 'Sex', 'SA', 'RonsF', 'Tpeak', 'Pan.1', 'VAT', 'Pfi', 'HFQRS', 'PpeakP', 'Pst', 'QRSw', 'Pan', 'Tons', 'PpeakN', 'Sbeta', 'Speak']
SELECTED_VARS_LEN = [15, 20]
THRES_TYPE = 'weighted'
INPUT_DIMS = [256, 576]
KERNEL_SIZE = [4]
MAX_POOL = 6
BATCH = 64
EARL_STOP = 5
N_EPOCHS = 1000
LEARN_RATE = [0.001, .0001]
MODEL_NUMBER = 0

# batch = 64 / krn = 3 / lr = 0.001 / inp_dims = 1600, 3136, 5184
# batch = 64 / krn = 3 / lr = 0.0001 / inp_dims = 1600, 3136, 5184

# batch = 64 / krn = 4 / lr = 0.001 / inp_dims = 1600, 3136, 4096
# batch = 64 / krn = 4 / lr = 0.0001 / inp_dims = 1600, 3136, 4096

# batch = 64 / krn = 3(4) mxpool = 4/ lr = 0.001 / inp_dims = 1024, 1600, 2304
# batch = 64 / krn = 3(4) mxpool = 4/ lr = 0.0001 / inp_dims = 1024, 1600, 2304

# batch = 96 / krn = 3 / lr = 0.001 / inp_dims = 1600, 3136, 4096
# batch =  96 / krn = 3 / lr = 0.0001 / inp_dims = 1600, 3136, 4096

# batch = 32 / krn = 4 / lr = 0.001 / inp_dims = 1600, 3136, 4096
# batch = 32 / krn = 4 / lr = 0.0001 / inp_dims = 1600, 3136, 4096

# Load data
df = pd.read_pickle('./data/01_raw_data/data.pkl')

# Split data for test / train
X = df.drop('Y_diast', 1)
Y = df['Y_diast']



for selected_var_len, input_dims in zip(SELECTED_VARS_LEN, INPUT_DIMS):

    selected = SELECTED_VARS[0:selected_var_len]

    # Split data to train and test
    X_train, X_test, y_train, y_test = train_test_split(X[selected], Y, test_size=0.3, random_state=RS)

    # X_train = X_train
    # X_test = X_test[selected]

    target_0_df = X_train.loc[y_train[y_train == 0].index, :]
    target_1_df = X_train.loc[y_train[y_train == 1].index, :]

    # Create resampled dataset with 500 patients and 1:1 target classes 
    X_train_res = pd.concat([
        resample(target_0_df, n_samples = 350, random_state = RS),
        resample(target_1_df, n_samples = 350, random_state = RS)
    ])

    y_train_res = np.hstack([np.zeros(350), np.ones(350)])

    y_train = pd.Series(y_train_res, index = X_train_res.index)

    train_index = [str(x) + 'train' for x in range(len(X_train_res.index))]
    test_index = [str(x) + 'test' for x in X_test.index]

    y_train.index = train_index
    y_test.index = test_index

    # Create `preprocessor` pipeline for data preprocessing
    # Imputing, scaling, one-hot-encoding

    category_cols = [col for col in X_train_res.columns if X_train_res[col].dtype == 'category']
    numeric_cols = [col for col in X_train_res.columns if col not in category_cols]


    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
                                ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(
                handle_unknown='ignore', 
                drop = 'if_binary',
                sparse = False))
    ])

    preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_cols),
                    ("cat", categorical_transformer, category_cols)
                    ],
                    remainder='passthrough'
                    )

    X_train_res = pd.DataFrame(preprocessor.fit_transform(X_train_res))
    X_test = pd.DataFrame(preprocessor.transform(X_test))

    colnames = [list(y.get_feature_names_out()) for x,y in preprocessor.named_transformers_.items()]

    colnames = list(itertools.chain(*colnames))

    X_train_res.columns = colnames
    X_test.columns = colnames

    X_train_res.index = train_index
    X_test.index = test_index

    X_full = pd.concat([
        X_train_res,
        X_test
        ])

    y_full = pd.concat([
        y_train,
        y_test
    ])

    class DataFrameLoader(object):

        def __init__(
                self
            )-> None:
            pass     

        def fit(self, data: pd.core.frame.DataFrame):
            self.data = data
            self.nodes_lst = data.columns
            self.edges_lst = [i for i in list(combinations(self.nodes_lst, r = 2))]


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
                    clf = clf
                else:
                    clf = SVC(probability = True, random_state=RS)
                    # clf = SGDClassifier(random_state = RS,
                    #     n_jobs = -1,
                    #     loss='huber')

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


        def ntwrk_construct(self, type = 'thres'):

            self.graphs = dict()

            if type == 'weighted':

                for index, s_idx in zip([x for x in range(len(self.index))], [x for x in self.index]):              
                    G = nx.Graph()
                    G.add_nodes_from(self.nodes_lst)
                    for edge, position in [[self.edges_lst[i],i] for i in range(len(self.edges_lst))]:
                        G.add_edge(edge[0], edge[1], weight = float(self.new_epsilons[index, position]))

                    self.graphs[s_idx] = G

            elif type == 'thres':

                for index, s_idx in zip([x for x in range(len(self.index))], [x for x in self.index]):              
                    G = nx.Graph()
                    G.add_nodes_from(self.nodes_lst)
                    for edge, position in [[self.edges_lst[i],i] for i in range(len(self.edges_lst))]:
                        # print(float(self.new_epsilons[index, position]))
                        if float(self.new_epsilons[index, position]) > EDGE_THRES:
                            G.add_edge(edge[0], edge[1])

                    self.graphs[s_idx] = G

            elif type == 'weighted_thres':

                for index, s_idx in zip([x for x in range(len(self.index))], [x for x in self.index]):              
                    G = nx.Graph()
                    G.add_nodes_from(self.nodes_lst)
                    for edge, position in [[self.edges_lst[i],i] for i in range(len(self.edges_lst))]:
                        if float(self.new_epsilons[index, position]) > EDGE_THRES:
                            G.add_edge(edge[0], edge[1], weight = float(self.new_epsilons[index, position]))
                        else:
                            G.add_edge(edge[0], edge[1], weight = 0)

                    self.graphs[s_idx] = G

            else:
                raise KeyboardError('Type must be `weighted`, `thres` or `weighted_thres`')


    snc = Snltc()

    snc.fit(X_train_res, y_train)

    snc.transform(X_full, newindex=list(X_full.index))

    snc.ntwrk_construct(type = THRES_TYPE)

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

    XY_train = XY_train.sample(frac = 1, random_state = RS)

    X_train, X_test = [x.drop('Group', 1) for x in [XY_train, XY_test]]
    y_train, y_test = [x['Group'] for x in [XY_train, XY_test]]

    X_train, X_test = [torch.FloatTensor(x) for x in \
        [X_train['Matrices'], X_test['Matrices']]]
    y_train, y_test = [torch.LongTensor(x) for x in [y_train.astype(int), y_test.astype(int)]]

    X_train = torch.reshape(X_train, (-1, 1, selected_var_len, selected_var_len))
    X_test = torch.reshape(X_test, (-1, 1, selected_var_len, selected_var_len))

    # Defining CNN...

    def init_weights(self):
        for module in self.modules():
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)):
                nn.init.xavier_uniform(module.weight)
                module.bias.data.fill_(0.01)

    class Net(nn.Module):
        def __init__(self, kernel_size, input_dims):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 64, kernel_size, 1, 2)
            self.conv1_drop = nn.Dropout2d(.5)
            # # self.conv2 = nn.Conv2d(64, 64, kernel_size, 1, 2)
            # self.conv2_drop = nn.Dropout2d(.6)
            self.bnm1 = nn.BatchNorm1d(input_dims, momentum=0.1)    
            self.fc1 = nn.Linear(input_dims, 1024)
            self.bnm2 = nn.BatchNorm1d(1024, momentum=0.1)
            self.fc2 = nn.Linear(1024, 256)
            self.bnm3 = nn.BatchNorm1d(256, momentum=0.1)
            self.fc3 = nn.Linear(256, 64)
            self.fc4 = nn.Linear(64, 2)

        def forward(self, x):
            x = F.max_pool2d(self.conv1_drop(self.conv1(x)), MAX_POOL)
            # x = F.max_pool2d(self.conv2_drop(self.conv2(x)), 3)
            x = x.view(x.size(0), -1)
            x = F.relu(F.dropout(self.fc1(self.bnm1(x)), 0.5))
            x = F.relu(F.dropout(self.fc2(self.bnm2(x)), 0.5))
            x = F.relu(F.dropout(self.fc3(self.bnm3(x)), 0.5))
            x = self.fc4(x)
            return F.log_softmax(x)

    # +----------------------------------

    batch_size = BATCH

    # Loading data

    train_loader = torch.utils.data.DataLoader(
        TensorDataset(X_train, y_train),
    batch_size=batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        TensorDataset(X_test, y_test),
    batch_size=batch_size, shuffle=True)

    # Calculationg loss

    # GREAT FOR

    for kernel_size in KERNEL_SIZE:
        for learning_rate in LEARN_RATE:

            loss_per_epoch_train = []
            loss_per_epoch_val = []

            torch.manual_seed(RS)
            network = Net(kernel_size, input_dims)
            network.apply(init_weights)

            loss_fn=nn.CrossEntropyLoss()
            optimizer = optim.Adam(network.parameters(), lr=learning_rate)

            early_stop_thres = EARL_STOP
            early_stop_counter = 0

            for epoch in range(N_EPOCHS):

                loss_per_batch_train = []
                loss_per_batch_val = []

                for x, y in train_loader:          
                
                    optimizer.zero_grad()
                    pred = network(x)
                    loss=loss_fn(pred, y)
                    loss_per_batch_train = loss_per_batch_train + [loss.item()]
                    loss.backward()
                    optimizer.step()

                loss_per_epoch_train = loss_per_epoch_train + [np.array(loss_per_batch_train).mean()]

                network.eval()

                for x, y in val_loader:

                    pred_val = network(x)
                    loss_eval=loss_fn(pred_val, y)
                    loss_per_batch_val = loss_per_batch_val + [loss_eval.item()]

                last_epoch = epoch

                last_loss = np.array(loss_per_batch_val).mean()

                if len(loss_per_epoch_val) > 0:
                    max_loss = max(loss_per_epoch_val[-20:])
                    min_loss = min(loss_per_epoch_val)
                else:
                    max_loss = 0
                    # min_loss = np.Inf

                loss_per_epoch_val = loss_per_epoch_val + [last_loss]  

                if last_loss > max_loss:
                # if last_loss > min_loss:

                    early_stop_counter += 1

                    if early_stop_counter == early_stop_thres:

                        break

            save_name = 'Model_' + str(MODEL_NUMBER)
            
            torch.save(network, './Test/Models/' + save_name + '.pth')

            # Plotting loss curves          

            fig = plt.figure()
            plt.plot(
                np.array(range(last_epoch + 1)),
                np.array(loss_per_epoch_train),
                label = f'TRAIN loss / LR {learning_rate}'
                )
            plt.legend()

            plt.plot(
                np.array(range(last_epoch + 1)),
                np.array(loss_per_epoch_val),
                label = f'VALIDATION loss / LR {learning_rate}'
                )
            plt.legend()
            fig.savefig('./Test/Graphics/' + save_name + '_LOSS.png', facecolor='white', transparent=False)

            # Let`s plot the ROC - curve

            y_pred = torch.sigmoid(network(X_test)).detach().numpy()[:,1]
            print(f'Model {MODEL_NUMBER}')
            print(f'Test data ROC AUC score: {roc_auc_score(y_test.detach().numpy(), y_pred)}\n')

            mdp.ROCPlotter_Binary(
                y_test.detach().numpy(), 
                y_pred,
                plot=False, 
                save_name='./Test/Graphics/' + save_name
                )

            MODEL_NUMBER += 1