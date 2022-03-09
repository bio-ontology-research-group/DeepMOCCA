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
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, SAGPooling
from torch_geometric.nn import global_max_pool as gmp
import click as ck
import gzip
import pickle
import sys
import matplotlib.pyplot as plt
import statistics
import pandas as pd


# In[2]:


cancer_types = [
    "TCGA-ACC", "TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC",
    "TCGA-CHOL", "TCGA-COAD", "TCGA-DLBC", "TCGA-ESCA",
    "TCGA-GBM", "TCGA-HNSC", "TCGA-KICH", "TCGA-KIRC",
    "TCGA-KIRP", "TCGA-LAML","TCGA-LGG","TCGA-LIHC",
    "TCGA-LUAD","TCGA-LUSC","TCGA-MESO","TCGA-OV",
    "TCGA-PAAD","TCGA-PCPG","TCGA-PRAD","TCGA-READ",
    "TCGA-SARC","TCGA-SKCM","TCGA-STAD","TCGA-TGCT",
    "TCGA-THCA","TCGA-THYM","TCGA-UCEC","TCGA-UCS","TCGA-UVM"]


# In[3]:


device = 'cuda:0'
proteins_df = pd.read_pickle('data/proteins_700.pkl')
interactions_df = pd.read_pickle('data/interactions_700.pkl')
proteins = {row.proteins: row.ids for row in proteins_df.itertuples()}
edge_index = [interactions_df['protein1'].values, interactions_df['protein2'].values]
edge_index = torch.LongTensor(edge_index).to(device)


# In[4]:


class MyNet(nn.Module):
    def __init__(self, num_nodes, edge_index):
        super(MyNet, self).__init__()
        self.num_nodes = num_nodes
        self.edge_index = edge_index
        
        self.linears = nn.ModuleList([nn.Linear(6, 6) for i in range(num_nodes)])
#         self.nbatches = nn.ModuleList([nn.BatchNorm1d(6) for i in range(num_nodes)])
#         self.dropouts = nn.ModuleList([nn.Dropout(0.1) for i in range(num_nodes)])

#         self.fc1 = nn.Linear(17185*2, 17185*2)
#         self.bn1 = nn.BatchNorm1d(17185*2)
#         self.dropout1 = nn.Dropout(0.1)
        
#         self.fc2 = nn.Linear(17185*6, 1024)
#         self.bn2 = nn.BatchNorm1d(1024)
#         self.dropout2 = nn.Dropout(0.1)
        
        
#         self.fc3 = nn.Linear(1024, 256)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.dropout3 = nn.Dropout(0.1)

        self.nbatches = nn.BatchNorm1d(6*num_nodes)
        self.dropouts = nn.Dropout(0.1)

        self.fc_final = nn.Linear(6*num_nodes, 1)
        self.bn_final = nn.BatchNorm1d(1)
        self.dropout_final = nn.Dropout(0.1)
        
        self.sigmoid = nn.Sigmoid()
        self.batches = {}
        
    def forward(self, data):
#         print('maybe',data.shape)
        j = torch.cuda.memory_stats('cuda:0')
        print(j['allocated_bytes.all.current'])
        for i,l in enumerate(self.linears):
            datat=data[:,6*i:6*i+6]
#             print('datat: ',datat.shape)
            if i==0:
                xo=l(datat)
#                 print('xo: ',xo)
            else:
                xo=torch.hstack((xo,l(datat)))
#         print(xo)
#         print(xo.shape)
#         if xo.shape[0] == 32:
#             name = datetime.datetime.utcnow().strftime('%Y-%m-%d %H_%M_%S.%f')[:-3]
#             filename = "rank/"+'ACC'+"/%s.txt"% name
#             xx = torch.sigmoid(xo)
#             xx = torch.mean(xx, 0)
#             print(xx.shape)
#             with open(filename, 'w') as filehandle:
#                 for listitem in xx.tolist():
#                     filehandle.write('%s\n' % listitem)
        x=self.dropouts(self.nbatches(torch.relu(xo)))
#         print(x)
#         print(x.shape)
        return torch.sigmoid(self.dropout_final(self.bn_final(self.fc_final(x))))
    """
#         print(data.shape)
        j = torch.cuda.memory_stats('cuda:0')
        print(j['allocated_bytes.all.current'], len(self.batches))
        batch_size = data.shape[0]
        x = data[:, :self.num_nodes * 6]
#         print(x.shape)
#         x = x.reshape(batch_size, 2)
#         print(x.shape)
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
        
        for i, l in enumerate(self.linears):
            ind = (batch.edge_index[0] == i).nonzero(as_tuple=True)[0]
            x = self.dropouts[i](self.nbatches[i](torch.relu(self.linears[i](x))))
            if ind.nelement() != 0:
                for j, l in enumerate(ind): # iterate over all neighbors of node X
                    num = (batch.edge_index[1])[l] # Yth neighbor
                    x = self.dropouts[num](self.nbatches[num](torch.relu(self.linears[num](x))))
            
#         print(x.get_device())
#         print(adj.get_device())
#         print(self.fc1.weight.get_device())
#         print(self.fc1.bias.get_device())
#         print(x.shape)
#         print(adj.shape)
#         print(self.fc1.weight.shape)
#         print(self.fc1.bias.shape)
#         x = torch.matmul(x, torch.transpose(torch.mul(self.fc1.weight, adj.to(device)), dim0=0,dim1=1)) + self.fc1.bias
#         print(x)
#         x = self.dropout1(self.bn1(torch.relu(x)))
#         x = self.dropout2(self.bn2(torch.relu(self.fc2(x))))
#         x = self.dropout3(self.bn3(torch.relu(self.fc3(x))))
        x = self.dropout_final(self.bn_final(self.fc_final(x)))
        return x

# net = MyNet(len(proteins), edge_index).to(device)
# net"""


# In[5]:


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


# In[6]:


data = {}
dfs = []
#ty_ = 0
#lr_ = 0.01
#pat_ = 10
#norm_ = normalize_by_column
ty_ = int(sys.argv[1])
lr_ = float(sys.argv[2])
pat_ = int(sys.argv[3])
norm_ = globals()[sys.argv[4]]
for i, cancer_type in enumerate(cancer_types):
    # Create a new model for each cancer type
    print(f'Training for {cancer_type}')
    
    df = pd.read_pickle(f'preprocessing_codes/{cancer_type}_avg.pkl')
    df['cancer_type'] = [i] * len(df)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
# print(df)
can = df.index[df['cancer_type'] == ty_].tolist()
# print(can)
dataset = np.stack(df['features'].values).reshape(len(df), -1)
# dataset = torch.FloatTensor(dataset)
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
print('Censored', len(censored_index))
print('Uncensored', len(uncensored_index))
censored_index = np.array(censored_index)
uncensored_index = np.array(uncensored_index)
ev_ = []
splits = 5
best_cindex = 0
num_features = 6
num_nodes = 17185
for fold in range(splits):
#     del net
    torch.manual_seed(0)
    net = MyNet(len(proteins), edge_index).to(device)
#     net = tt.practical.MLPVanilla(in_features, [1024, 512], 1, True,
#                              0.1, output_bias=False)
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
    train_data = train_data.reshape(-1, num_nodes * num_features)
    train_labels_days = labels_days[train_idx]
    train_labels_surv = labels_surv[train_idx]
    train_labels = (train_labels_days, train_labels_surv)
    val_data = dataset[valid_idx]
    val_data = val_data.reshape(-1, num_nodes, num_features)
    for i in range(num_features):
        val_data[:, :, i] = normalize_func(val_data[:, :, i])
    val_data = val_data.reshape(-1, num_nodes * num_features)
    val_labels_days = labels_days[valid_idx]
    val_labels_surv = labels_surv[valid_idx]
    
    test_idx = [x for x in can if x in test_idx]
#     print(test_idx)
    test_data = dataset[test_idx]
    test_data = test_data.reshape(-1, num_nodes, num_features)
    for i in range(num_features):
        test_data[:, :, i] = normalize_func(test_data[:, :, i])
    test_data = test_data.reshape(-1, num_nodes * num_features)
    test_labels_days = labels_days[test_idx]
    test_labels_surv = labels_surv[test_idx]
    val_labels = (val_labels_days, val_labels_surv)
    print(val_labels)
    print('Training data', train_data.shape)
    print('Validation data', val_data.shape)
    print('Testing data', test_data.shape)
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
print('cancer_type = '+str(ty_))
print('lr = '+str(lr_))
print('patience = '+str(pat_))
print('normalization = '+str(norm_))           
print(str(statistics.mean(ev_))+"["+str(min(ev_))+"-"+str(max(ev_))+"]")


# In[ ]:
