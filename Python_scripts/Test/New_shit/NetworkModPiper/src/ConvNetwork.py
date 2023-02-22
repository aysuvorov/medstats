import numpy as np
import pandas as pd

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

import src.ModPerf as mdp


class CNNTrainer:
    def __init__(self):
        pass

    def fit(self,
        adj_mtx_path,
        cnn_model_path,
        learning_rate,
        n_epochs,
        batch_size,
        random_state):

        X_train = pd.read_pickle(adj_mtx_path + "/X_train.pkl")
        y_train = pd.read_pickle(adj_mtx_path + "/y_train.pkl")
        X_test = pd.read_pickle(adj_mtx_path + "/X_test.pkl")
        y_test = pd.read_pickle(adj_mtx_path + "/y_test.pkl")

        dims = X_train['Matrices'][0].shape
        print(dims)

        X_train, X_test = [torch.FloatTensor(x) for x in \
            [X_train['Matrices'], X_test['Matrices']]]
        y_train, y_test = [torch.LongTensor(x) for x in [y_train.astype(int), y_test.astype(int)]]

        X_train = torch.reshape(X_train, (-1, 1, dims[0], dims[1]))
        X_test = torch.reshape(X_test, (-1, 1, dims[0], dims[1]))

        train_loader = torch.utils.data.DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True
            )

        val_loader = torch.utils.data.DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=batch_size,
            shuffle=True
            )

        # Neural net classifier

        def init_weights(self):
            for module in self.modules():
                if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)):
                    nn.init.xavier_uniform(module.weight)
                    module.bias.data.fill_(0.01)

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1, 2)
                self.conv2 = nn.Conv2d(32, 32, 3, 1, 2)
                self.conv2_drop = nn.Dropout2d(.5)
                self.bnm1 = nn.BatchNorm1d(288, momentum=0.1)    
                self.fc1 = nn.Linear(288, 128)
                self.bnm2 = nn.BatchNorm1d(128, momentum=0.1)
                self.fc2 = nn.Linear(128, 64)
                self.bnm3 = nn.BatchNorm1d(64, momentum=0.1)
                self.fc3 = nn.Linear(64, 64)
                self.fc4 = nn.Linear(64, 2)
            

            def forward(self, net):
                net = F.max_pool2d(self.conv1(net), 3)
                net = F.max_pool2d(self.conv2_drop(self.conv2(net)), 3)
                net = net.view(net.size(0), -1)
                net = F.relu(F.dropout(self.fc1(self.bnm1(net)), 0.5))
                net = F.relu(F.dropout(self.fc2(self.bnm2(net)), 0.5))
                net = F.relu(F.dropout(self.fc3(self.bnm3(net)), 0.5))
                net = self.fc4(net)
                return F.log_softmax(net)

        # Calculationg loss

        loss_per_epoch_train = []
        loss_per_epoch_val = []

        torch.manual_seed(random_state)
        network = Net()
        network.apply(init_weights)

        loss_fn=nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            network.parameters(),
            lr=learning_rate
            )

        early_stop_thres = 3
        early_stop_counter = 0

        for epoch in range(n_epochs):

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
                max_loss = max(loss_per_epoch_val)
            else:
                max_loss = 0
            loss_per_epoch_val = loss_per_epoch_val + [last_loss]

            if last_loss > max_loss:
                early_stop_counter += 1
                if early_stop_counter == early_stop_thres:

                    break

        y_pred = torch.sigmoid(network(X_test)).detach().numpy()[:,1]

        model_name = 'model_' + str(roc_auc_score(y_test.detach().numpy(), y_pred))

        best_dict_params = {
            'LR': learning_rate,
            'Batch': batch_size,
            'Epochs': last_epoch
        }

        mdp.output_model_log(
            y_test.detach().numpy(),
            y_pred,
            cnn_model_path + '/Model_info' + model_name,
            best_dict_params)

        torch.save(
            network.state_dict(),
            cnn_model_path + '/' + model_name + '.pth')

        fig = plt.figure()
        plt.plot(
            np.array(range(last_epoch + 1)),
            np.array(loss_per_epoch_train)
            )
        plt.legend()

        plt.plot(
            np.array(range(last_epoch + 1)),
            np.array(loss_per_epoch_val)
            )
        plt.legend()
        fig.savefig(cnn_model_path + '/Model_info' + model_name + '_LOSS.png', facecolor='white', transparent=False)