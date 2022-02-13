#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import numpy as np
from tqdm import tqdm
import rdflib as rl
import torch
import torchtuples as tt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, GENConv, GATConv, SAGPooling
from torch_geometric.nn import global_max_pool as gmp
from torch.nn.parallel import DistributedDataParallel as DDP
#from SAGPooling import SAGPooling
import click as ck
import gzip
import pickle
import sys
import matplotlib.pyplot as plt
import statistics
import pandas as pd
import random
# from torch.utils.tensorboard import SummaryWriter


# In[2]:


cancer_types = [str(sys.argv[1])]

#cancer_types = ['TCGA-BRCA']


# In[3]:


device = 'cuda:0'


# In[4]:


proteins_df = pd.read_pickle('data/proteins_700.pkl')
interactions_df = pd.read_pickle('data/interactions_700.pkl')
proteins = {row.proteins: row.ids for row in proteins_df.itertuples()}
edge_index = [interactions_df['protein1'].values, interactions_df['protein2'].values]
edge_index = torch.LongTensor(edge_index).to(device)


# In[5]:


class MyNet(nn.Module):
    def __init__(self, num_nodes, edge_index):
        super(MyNet, self).__init__()
        self.num_nodes = num_nodes
        self.edge_index = edge_index
       # self.conv1 = GraphConv(6, 6)
#         self.pool1 = SAGPooling(6, ratio=0.1, GNN=GraphConv)
        self.fc1 = nn.Linear(6*num_nodes, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(1)
        self.dropout1 = nn.Dropout(0.1)
#         self.fc2 = nn.Linear(1024, 512, bias=False)
#         self.bn2 = nn.BatchNorm1d(512)
#         self.dropout2 = nn.Dropout(0.1)
#         self.fc3 = nn.Linear(512, 1, bias=False)
#         self.bn3 = nn.BatchNorm1d(1)
#         self.dropout3 = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.batches = {}

    def forward(self, data):
        j = torch.cuda.memory_stats('cuda:0')
        print(j['allocated_bytes.all.current'],len(self.batches))
        batch_size = data.shape[0]
        x = data[:, :self.num_nodes * 6]
        x = x.reshape(batch_size, self.num_nodes, 6)
        batch=''
        if True:
            l = []
            for i in range(batch_size):
                l.append(Data(x=x[i], edge_index=self.edge_index))
#             if batch_size not in self.batches:
#                 print(x[i])
#                 print(self.edge_index)
            batch = Batch.from_data_list(l)
            self.batches[batch_size] = True
        batch = batch.to(device)
        x = x.to(device)
        x = x.reshape(-1, 6)
       # x = F.relu(self.conv1(x=x, edge_index=batch.edge_index))
#         x, edge_index, _, batch, perm, score = self.pool1(
#             x, batch.edge_index, None, batch.batch)
        #print(x.shape)
        x = x.view(batch_size, -1)
        x = self.dropout1(self.bn1(torch.relu(self.fc1(x))))
#         x = self.dropout2(self.bn2(torch.relu(self.fc2(x))))
#         x = self.dropout3(self.bn3(self.fc3(x)))
        return x
    
net = MyNet(len(proteins), edge_index).to(device)
net


# In[6]:


def normalize(data, minx=None, maxx=None):
    if minx is None:
        minx = np.min(data)
        maxx = np.max(data)
    if minx == maxx:
        return data
    return (data - minx) / (maxx - minx)
        
def normalize_by_row(data):
    for i in range(data.shape[0]):
        data[i, :] = normalize(data[i, :])
    return data

def normalize_by_column(data):
    for i in range(data.shape[1]):
        data[:, i] = normalize(data[:, i])
    return data


# In[7]:


lr_ = float(sys.argv[2])
pat_ = int(sys.argv[3])
norm_ = globals()[sys.argv[4]]

#lr_ = 0.01
#pat_ = 10
#norm_ = normalize_by_column

for i, cancer_type in enumerate(cancer_types):
    # Create a new model for each cancer type
    df = pd.read_pickle(f'preprocessing_codes/{cancer_type}_avg.pkl')
    print(df)
    dataset = np.stack(df['features'].values).reshape(len(df), -1)
    print(dataset.shape)
    in_features = dataset.shape[1]
    labels_days = df['duration'].values
    labels_surv = df['survival'].values
    censored_index = []
    uncensored_index = []
    for i in range(len(dataset)):
        if labels_surv[i] == 1:
            censored_index.append(i)
        else:
            uncensored_index.append(i)
#                 print('Censored', len(censored_index))
#                 print('Uncensored', len(uncensored_index))
    censored_index = np.array(censored_index)
    uncensored_index = np.array(uncensored_index)
    ev_ = []
    splits = 5
    best_cindex = 0
    num_features = 6
    num_nodes = 17185
    for fold in range(splits):
#         del net
        torch.manual_seed(0)
#         model = MyNet(len(proteins), edge_index)
#         if torch.cuda.device_count() > 1:
#             print("Let's use", torch.cuda.device_count(), "GPUs!")
#             model = nn.DataParallel(model)
        net = MyNet(len(proteins), edge_index).to(device)
#         net = tt.practical.MLPVanilla(in_features, [1024, 512], 1, True, lr_, output_bias=False)
        model = CoxPH(net, tt.optim.Adam(lr_))
        # Censored split
        n = len(censored_index)
        index = np.arange(n)
        i = n // 5
        np.random.seed(seed=0)
        np.random.shuffle(index)
        if fold < 4:
            ctest_idx = index[fold * i: fold * i + i]
            ctrain_idx = np.concatenate((index[:fold * i],index[fold * i + i:]))
        else:
            ctest_idx = index[fold * i:]
            ctrain_idx = index[:fold * i]
        ctrain_n = len(ctrain_idx)
        cvalid_n = ctrain_n // 10
        cvalid_idx = ctrain_idx[:cvalid_n]
        ctrain_idx = ctrain_idx[cvalid_n:]

        # Uncensored split
        n = len(uncensored_index)
        index = np.arange(n)
        i = n // 5
        np.random.seed(seed=0)
        np.random.shuffle(index)
        if fold < 4:
            utest_idx = index[fold * i: fold * i + i]
            utrain_idx = np.concatenate((index[:fold * i],index[fold * i + i:]))
        else:
            utest_idx = index[fold * i:]
            utrain_idx = index[:fold * i]
        utrain_n = len(utrain_idx)
        uvalid_n = utrain_n // 10
        uvalid_idx = utrain_idx[:uvalid_n]
        utrain_idx = utrain_idx[uvalid_n:]
        train_idx = np.concatenate((
            censored_index[ctrain_idx], uncensored_index[utrain_idx]))
        np.random.seed(seed=0)
        np.random.shuffle(train_idx)
        valid_idx = np.concatenate((
            censored_index[cvalid_idx], uncensored_index[uvalid_idx]))
        np.random.seed(seed=0)
        np.random.shuffle(valid_idx)
        test_idx = np.concatenate((
            censored_index[ctest_idx], uncensored_index[utest_idx]))
        np.random.seed(seed=0)
        np.random.shuffle(test_idx)
        normalize_func = norm_
        train_data = dataset[train_idx]
        # Normalize by rows (genes)
        train_data = train_data.reshape(-1, num_nodes, num_features)
        for i in range(num_features):
            train_data[:, :, i] = normalize_func(train_data[:, :, i])
#             if i <= 5:
#                 train_data[:, :, i] = normalize_by_column(train_data[:, :, i])
#             elif i > 5 and i <= 11:
#                 train_data[:, :, i] = normalize_by_row(train_data[:, :, i])
#             else:
#                 train_data[:, :, i] = normalize(train_data[:, :, i])
        train_data = train_data.reshape(-1, num_nodes * num_features)
        train_labels_days = labels_days[train_idx]
        train_labels_surv = labels_surv[train_idx]
        train_labels = (train_labels_days, train_labels_surv)
        val_data = dataset[valid_idx]
        val_data = val_data.reshape(-1, num_nodes, num_features)
        for i in range(num_features):
            val_data[:, :, i] = normalize_func(val_data[:, :, i])
#             if i <= 5:
#                 val_data[:, :, i] = normalize_by_column(val_data[:, :, i])
#             elif i > 5 and i <= 11:
#                 val_data[:, :, i] = normalize_by_row(val_data[:, :, i])
#             else:
#                 val_data[:, :, i] = normalize(val_data[:, :, i])
        val_data = val_data.reshape(-1, num_nodes * num_features)
        val_labels_days = labels_days[valid_idx]
        val_labels_surv = labels_surv[valid_idx]
        test_data = dataset[test_idx]
        test_data = test_data.reshape(-1, num_nodes, num_features)
        for i in range(num_features):
            test_data[:, :, i] = normalize_func(test_data[:, :, i])
#             if i <= 5:
#                 test_data[:, :, i] = normalize_by_column(test_data[:, :, i])
#             elif i > 5 and i <= 11:
#                 test_data[:, :, i] = normalize_by_row(test_data[:, :, i])
#             else:
#                 test_data[:, :, i] = normalize(test_data[:, :, i])
        test_data = test_data.reshape(-1, num_nodes * num_features)
        test_labels_days = labels_days[test_idx]
        test_labels_surv = labels_surv[test_idx]
        val_labels = (val_labels_days, val_labels_surv)
        callbacks = [tt.callbacks.EarlyStopping(patience=pat_)]
        batch_size = 32
        epochs = 100
        val = (val_data, val_labels)
        log = model.fit(
            train_data, train_labels, batch_size, epochs, callbacks, verbose=True,
            val_data=val,
            val_batch_size=batch_size)
        train = train_data, train_labels
        # Compute the evaluation measurements
        _ = model.compute_baseline_hazards(input=train_data, target=train_labels, batch_size=batch_size)
        surv = model.predict_surv_df(input=test_data, batch_size=batch_size)
        ev = EvalSurv(surv, test_labels_days, test_labels_surv)
        result = ev.concordance_td()
        print('Concordance', result)
        ev_.append(result)
    print(cancer_type)
    print('lr = '+str(lr_))
    print('patience = '+str(pat_))
    print('normalization = '+str(norm_))
    print(str(statistics.mean(ev_))+"["+str(min(ev_))+"-"+str(max(ev_))+"]")

