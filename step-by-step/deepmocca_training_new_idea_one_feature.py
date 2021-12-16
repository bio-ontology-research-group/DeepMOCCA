#!/usr/bin/env python
# coding: utf-8

# # Training individual models

# ## Imports

# In[1]:


# CUDA_VISIBLE_DEVICES=0,1
# CUDA_LAUNCH_BLOCKING=1
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
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, GENConv, GATConv
from SAGPooling import SAGPooling
from torch_geometric.nn import global_max_pool as gmp
from torch.nn.parallel import DistributedDataParallel as DDP
import click as ck
import gzip
import pickle
import sys
import matplotlib.pyplot as plt
import statistics
import pandas as pd
import random
# from torch.utils.tensorboard import SummaryWriter


# ## Constant variables

# In[2]:


# Manually categorized cancer subtypes
CANCER_SUBTYPES = [
    [0,12,7,14,4,1,6,2,3],
    [4],
    [5,4,14,6],
    [6,4,12,7],
    [4],
    [6,4,12,7],
    [8],
    [6,4,12],
    [9],
    [6],
    [4],
    [4],
    [4],
    [10],
    [9],
    [4],
    [4,11,12],
    [6],
    [13],
    [12],
    [0,4,12,14],
    [15],
    [4,0,12],
    [4,12],
    [16,17,18,19,20],
    [20],
    [4,12],
    [22],
    [4,14],
    [23],
    [4,12,14],
    [24],
    [21]
]

CELL_TYPES = [
    0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0, 4, 2, 0,
    0, 0, 5, 0, 0, 6, 0, 0, 7, 8, 0, 9, 0, 0, 0, 0,
    8]

# cancer_types = [
#     "TCGA-ACC", "TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC",
#     "TCGA-CHOL", "TCGA-COAD", "TCGA-DLBC", "TCGA-ESCA",
#     "TCGA-GBM", "TCGA-HNSC", "TCGA-KICH", "TCGA-KIRC",
#     "TCGA-KIRP", "TCGA-LAML","TCGA-LGG","TCGA-LIHC",
#     "TCGA-LUAD","TCGA-LUSC","TCGA-MESO","TCGA-OV",
#     "TCGA-PAAD","TCGA-PCPG","TCGA-PRAD","TCGA-READ",
#     "TCGA-SARC","TCGA-SKCM","TCGA-STAD","TCGA-TGCT",
#     "TCGA-THCA","TCGA-THYM","TCGA-UCEC","TCGA-UCS","TCGA-UVM"]

cancer_types = [str(sys.argv[1])]

# cancer_types = ['TCGA-GBM']


# ## Load proteins and interactions

# In[3]:


device = 'cuda:0'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'

#     # initialize the process group
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)

# def cleanup():
#     dist.destroy_process_group()

# torch.cuda.set_device(device)

proteins_df = pd.read_pickle('data/proteins_700.pkl')
interactions_df = pd.read_pickle('data/interactions_700.pkl')
# interactions_df = pd.read_csv('pr.tsv', header=None, sep='\t')
# interactions_df.columns = ['protein1','protein2']
proteins = {row.proteins: row.ids for row in proteins_df.itertuples()}
edge_index = [interactions_df['protein1'].values, interactions_df['protein2'].values]
edge_index = torch.LongTensor(edge_index).to(device)

# edge_index = torch.empty_like(edge_index)
# proteins, edge_index


# In[3]:


# device = 'cuda:0'

# proteins_df = pd.read_pickle('data/proteins_co_small.pkl')
# interactions_df = pd.read_pickle('data/interactions_co_small.pkl')
# proteins = {row.proteins: row.ids for row in proteins_df.itertuples()}
# edge_index = [interactions_df['protein1'].values, interactions_df['protein2'].values]
# edge_index = torch.LongTensor(edge_index).to(device)

# # proteins, edge_index


# In[ ]:


# device = torch.device('cpu')

# cancer_type_vector = np.zeros((33,), dtype=np.float32)
# cancer_type_vector[cancer_type] = 1

# cancer_subtype_vector = np.zeros((25,), dtype=np.float32)
# for i in CANCER_SUBTYPES[cancer_type]:
#     cancer_subtype_vector[i] = 1

# anatomical_location_vector = np.zeros((52,), dtype=np.float32)
# anatomical_location_vector[0] = 1
# cell_type_vector = np.zeros((10,), dtype=np.float32)
# cell_type_vector[CELL_TYPES[cancer_type]] = 1

# pt_tensor_cancer_type = torch.FloatTensor(cancer_type_vector).to(device)
# pt_tensor_cancer_subtype = torch.FloatTensor(cancer_subtype_vector).to(device)
# pt_tensor_anatomical_location = torch.FloatTensor(anatomical_location_vector).to(device)
# pt_tensor_cell_type = torch.FloatTensor(cell_type_vector).to(device)
# edge_index = torch.LongTensor(edge_index).to(device)


# In[4]:


# class MyNet(nn.Module):
#     def __init__(self, num_nodes, edge_index):
#         super(MyNet, self).__init__()
#         self.num_nodes = num_nodes
#         self.edge_index = edge_index
#         self.conv1 = GraphConv(6, 6)
#         self.pool1 = SAGPooling(6, ratio=0.1, GNN=GraphConv)
#         self.fc1 = nn.Linear(5460, 1024, bias=False)
#         self.bn1 = nn.BatchNorm1d(1024)
#         self.dropout1 = nn.Dropout(0.1)
#         self.fc2 = nn.Linear(1024, 512, bias=False)
#         self.bn2 = nn.BatchNorm1d(512)
#         self.dropout2 = nn.Dropout(0.1)
#         self.fc3 = nn.Linear(512, 1, bias=False)
#         self.bn3 = nn.BatchNorm1d(1)
#         self.dropout3 = nn.Dropout(0.1)
#         self.sigmoid = nn.Sigmoid()
#         self.batches = {}
#     def forward(self, data):
# #         j = torch.cuda.memory_stats('cuda:0')
# #         print(j['allocated_bytes.all.current'],len(self.batches))
#         batch_size = data.shape[0]
#         x = data[:, :self.num_nodes * 6]
#         x = x.reshape(batch_size, self.num_nodes, 6)
#         batch=''
#         if True:
#             l = []
#             for i in range(batch_size):
#                 l.append(Data(x=x[i], edge_index=self.edge_index))
# #             if batch_size not in self.batches:
# #                 print(x[i])
# #                 print(self.edge_index)
#             batch = Batch.from_data_list(l)
#             self.batches[batch_size] = True
#         batch = batch.to(device)
#         x = x.to(device)
#         x = x.reshape(-1, 6)
#         x = F.relu(self.conv1(x=x, edge_index=batch.edge_index))
#         x, edge_index, _, batch, perm, score = self.pool1(
#             x, batch.edge_index, None, batch.batch)
#         x = x.view(batch_size, -1)
#         x = self.dropout1(self.bn1(torch.relu(self.fc1(x))))
#         x = self.dropout2(self.bn2(torch.relu(self.fc2(x))))
#         x = self.dropout3(self.bn3(self.fc3(x)))
#         return x
    
# net = MyNet(len(proteins), edge_index).to(device)
# net


# In[11]:


def get_adj(nb=1):
    adj=[[0 for i in range(len(proteins))] for i in range(len(proteins))]
    for i in range(17185):
        adj[i][i] = 1
    for index, row in interactions_df.iterrows():
        adj[row['protein1']][row['protein2']] = 1
    adj=torch.tensor(adj, dtype=torch.float)
    if nb==0:
        return adj
    else:
        return adj.repeat(nb,nb)
num = 1
adj=get_adj(num)


# In[12]:


print(adj.shape)


# ## Model class

# In[20]:


class MyNet(nn.Module):
    def __init__(self, num_nodes, edge_index):
        super(MyNet, self).__init__()
        self.num_nodes = num_nodes
        self.edge_index = edge_index
#         self.conv1 = GraphConv(6, 6)
#         self.pool1 = SAGPooling(6, ratio=0.1, GNN=GraphConv)
        self.fc1 = nn.Linear(17185, 17185)
        self.bn1 = nn.BatchNorm1d(17185)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(17185, 1)
        self.bn3 = nn.BatchNorm1d(1)
        self.dropout3 = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.batches = {}
    def forward(self, data):
#         j = torch.cuda.memory_stats('cuda:0')
#         print(j['allocated_bytes.all.current'],len(self.batches))
        batch_size = data.shape[0]
        x = data[:, :self.num_nodes * 1]
#         x = x.reshape(batch_size, self.num_nodes, 6)
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
#         x = x.reshape(-1, 6)
#         x = F.relu(self.conv1(x=x, edge_index=batch.edge_index))
#         x, edge_index, _, batch, perm, score = self.pool1(
#             x, batch.edge_index, None, batch.batch)
#         x = x.view(batch_size, -1)
#         print(self.fc1.weight.shape)
#         print(adj.to(device).shape)
        x = torch.matmul(x, torch.transpose(torch.mul(self.fc1.weight, adj.to(device)), dim0=0,dim1=1)) + self.fc1.bias
        x = self.dropout1(self.bn1(torch.relu(x)))
#         x = self.dropout2(self.bn2(torch.relu(self.fc2(x))))
        x = self.dropout3(self.bn3(self.fc3(x)))
        return x
    
net = MyNet(len(proteins), edge_index).to(device)
net


# In[21]:


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

# def normalize_by_row_and_column(data):
#     data[:, len(proteins)*0:len(proteins)*6] = normalize_by_row(data[:, len(proteins)*0:len(proteins)*6])
#     data[:, len(proteins)*6:len(proteins)*12] = normalize_by_column(data[:, len(proteins)*6:len(proteins)*12])
#     return data

# def normalize_by_row_and_column_and_matrix(data):
#     data[:, len(proteins)*0:len(proteins)*6] = normalize_by_row(data[:, len(proteins)*0:len(proteins)*6])
#     data[:, len(proteins)*6:len(proteins)*12] = normalize_by_column(data[:, len(proteins)*6:len(proteins)*12])
#     data[:, len(proteins)*12:len(proteins)*18] = normalize(data[:, len(proteins)*12:len(proteins)*18])
#     return data


# In[22]:


# def normalize_train(data):
#     minx = np.min(data)
#     maxx = np.max(data)
#     return (data - minx) / (maxx - minx), minx, maxx

# def normalize(data, minx, maxx):
#     return (data - minx) / (maxx - minx)
        
# def normalize_by_row(data):
#     for i in range(data.shape[0]):
#         data[i, :] = normalize(data[i, :])
#     return data

# def normalize_by_column(data):
#     for i in range(data.shape[1]):
#         data[:, i] = normalize(data[:, i])
#     return data

# # def normalize_by_row_and_column(data):
# #     data[:, len(proteins)*0:len(proteins)*6] = normalize_by_row(data[:, len(proteins)*0:len(proteins)*6])
# #     data[:, len(proteins)*6:len(proteins)*12] = normalize_by_column(data[:, len(proteins)*6:len(proteins)*12])
# #     return data

# # def normalize_by_row_and_column_and_matrix(data):
# #     data[:, len(proteins)*0:len(proteins)*6] = normalize_by_row(data[:, len(proteins)*0:len(proteins)*6])
# #     data[:, len(proteins)*6:len(proteins)*12] = normalize_by_column(data[:, len(proteins)*6:len(proteins)*12])
# #     data[:, len(proteins)*12:len(proteins)*18] = normalize(data[:, len(proteins)*12:len(proteins)*18])
# #     return data


# In[23]:


# def normalize(data):
#     data_mean = np.mean(data)
#     data_std = np.std(data)
#     if data_std == 0:
#         print('mean = ',data_mean)
#         print('std = ',data_std)
# #     if minx is None:
# #         minx = np.min(data)
# #         maxx = np.max(data)
# #     if minx == maxx:
# #         return data
#         print('results = ',(data - data_mean) / (data_std))
#     return (data - data_mean) / (data_std)


# # def normalize(data, data_mean, data_std):
# # #     data_mean = np.mean(data)
# # #     data_std = np.std(data)
# # #     if data_std == 0:
# # #         print('mean = ',data_mean)
# # #         print('std = ',data_std)
# # #     if minx is None:
# # #         minx = np.min(data)
# # #         maxx = np.max(data)
# # #     if minx == maxx:
# # #         return data
# # #         print('results = ',(data - data_mean) / (data_std))
# #     return (data - data_mean) / (data_std)
        
# def normalize_by_row(data):
#     for i in range(data.shape[0]):
#         data[i, :] = normalize(data[i, :])
#     return data

# def normalize_by_column(data):
#     for i in range(data.shape[1]):
#         data[:, i] = normalize(data[:, i])
#     return data


# In[24]:


# lr_ = 0.01
# pat_ = 10
# norm_ = normalize

# # one = 0
# # two = 1
# for i, cancer_type in enumerate(cancer_types):
#     # Create a new model for each cancer type

#     df = pd.read_pickle(f'preprocessing_codes/{cancer_type}.pkl')
# #     print(df)

#     dataset = np.stack(df['features'].values).reshape(len(df), -1)
#     print(dataset.shape)

#     d = []
#     for j in range(17185):
#         print(j)
#         dat = dataset[:, (6*j)+2]
#         print(dat.shape)
#         print(dat)
#         dat = np.log(dat + 1)
#         print(dat)
#         d.append(dat)
# #         print(type(d))

# #     print(dataset)
#     print('meth_data: min = ', np.min(d), 'max = ', np.max(d))


# ## Train a model for each cancer type

# In[25]:


# lr_ = float(sys.argv[2])
# pat_ = int(sys.argv[3])
# norm_ = sys.argv[4]

# one = int(sys.argv[5])
# two = int(sys.argv[6])

lr_ = float(sys.argv[2])
pat_ = int(sys.argv[3])
norm_ = globals()[sys.argv[4]]

# one = 0
# two = 1
for i, cancer_type in enumerate(cancer_types):
    # Create a new model for each cancer type

    df = pd.read_pickle(f'preprocessing_codes/{cancer_type}_exp.pkl')
    print(df)

    dataset = np.stack(df['features'].values).reshape(len(df), -1)
    print(dataset.shape)
#     print(type(dataset))
    
    
#     arr = np.array([])
#     for j in range(17185):
#         arr = np.append(arr, dataset[:, (6*j)+2])
    
#     dataset = arr
#     for j in range(17185):
#         dataset[:, (6*j)+2] = np.log(dataset[:, (6*j)+2] + 1)
#         dataset[:, (6*j)+3] = np.log(dataset[:, (6*j)+3] + 1)
#         dataset[:, (6*j)+5] = np.log(dataset[:, (6*j)+5] + 14)


#     dataset = dataset[:, len(proteins)*2:len(proteins)*3]
#     print(type(dataset))
#     result = np.argwhere(dataset == 0)
#     print(result)
#     print(dataset.shape)
#     print('meth_data: min = ', dataset.min(axis=0), 'max = ', dataset.max(axis=0))
#     dataset = np.concatenate((dataset, dataset, dataset), axis=1)

#     dataset = np.concatenate((dataset, dataset), axis=1)

#                 print(dataset.shape)

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

#     print(dataset)
#     print(dataset.shape)

    ev_ = []
    splits = 5
    best_cindex = 0

    num_features = 1
    num_nodes = 17185

    for fold in range(splits):
        del net
        torch.manual_seed(0)
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

#                     print(val_labels)
#                     print('Training data', train_data.shape)
#                     print('Validation data', val_data.shape)
#                     print('Testing data', test_data.shape)
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

#         if result > best_cindex:
#             best_cindex = result

#             np.savetxt('test'+str(fold+1)+'.csv', test_data, delimiter="\t")

#             np.savetxt('test_labels_days'+str(fold+1)+'.csv', test_labels_days, delimiter="\t")

#             np.savetxt('test_labels_surv'+str(fold+1)+'.csv', test_labels_surv, delimiter="\t")

    print(cancer_type)
    print('lr = '+str(lr_))
    print('patience = '+str(pat_))
    print('normalization = '+str(norm_))
    print(str(statistics.mean(ev_))+"["+str(min(ev_))+"-"+str(max(ev_))+"]")
