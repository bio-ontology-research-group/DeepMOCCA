# coding: utf-8

#!/usr/bin/env python
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
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, SAGPooling
from torch_geometric.nn import global_max_pool as gmp
import click as ck
import gzip
import pickle
import sys
import matplotlib.pyplot as plt

class MyNet(nn.Module):
    def __init__(self, edge_index):
        super(MyNet, self).__init__()
        self.edge_index = edge_index
        self.conv1 = GCNConv(6,64)
        self.pool1 = SAGPooling(64, ratio=0.70, GNN=GCNConv)
        self.conv2 = GCNConv(64,32)
        self.fc1 = nn.Linear(32,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        batch_size = data.shape[0]
        x = data[:, :103116]
        metadata = data[:, 103116:]
        input_size = 17186
        x = x.reshape(-1, 6)
        batches = []
        for i in range(batch_size):
            tr = torch.ones(input_size, dtype=torch.int64) * i
            batches.append(tr)
        batch = torch.cat(batches, 0).to(device)
        #x = torch.from_numpy(x).to(device)
        x = F.relu(self.conv1(x, self.edge_index))
        x, edge_index, _, batch, perm, score = self.pool1(x, self.edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index))
        x = gmp(x, batch)
        #features = x
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x


def features(net, data):
    batch_size = data.shape[0]
    x = data[:, :103116]
    metadata = data[:, 103116:]
    input_size = 17186
    x = x.reshape(-1, 6)
    batches = []
    for i in range(batch_size):
        tr = torch.ones(input_size, dtype=torch.int64) * i
        batches.append(tr)
    batch = torch.cat(batches, 0).to(device)
    x = F.relu(net.conv1(x, net.edge_index))
    x, edge_index, _, batch, perm, score = net.pool1(x, net.edge_index, None, batch)
    x = F.relu(net.conv2(x, edge_index))
    x = gmp(x, batch)
    return x

global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

@ck.command()
@ck.option('--data-root', '-dr', default='data/', help='Data root folder', required=True)
@ck.option('--in-file', '-if', help='Input file', required=True)
@ck.option('--model-file', '-mf', default='model.pt', help='Pytorch model file')
@ck.option('--cancer-type-flag', '-ct', help='Cancer type flag', required=True)
@ck.option('--anatomical-part-flag', '-ap', help='Anatomical part flag', required=True)
@ck.option('--out-file', '-of', default='results.tsv', help='Output result file')
def main(data_root, in_file, model_file, cancer_type_flag, anatomical_part_flag, out_file):
    # Check data folder and required files
    try:
        if os.path.exists(data_root):
            model_file = os.path.join(data_root, model_file)
            if not os.path.exists(in_file):
                raise Exception(f'Input file ({in_file}) is missing!')
            if not os.path.exists(model_file):
                raise Exception(f'Model file ({model_file}) is missing!')
        else:
            raise Exception(f'Data folder {data_root} does not exist!')
    except Exception as e:
        logging.error(e)
        sys.exit(1)

    # Read input data
    cancer_type_flag = int(cancer_type_flag)
    anatomical_part_flag = int(anatomical_part_flag)
    data, clinical, surv_time, edge_index = load_data(data_root, in_file, cancer_type_flag, anatomical_part_flag)
    # Load and Run GCN model
    output = load_model(model_file, data, clinical, surv_time, edge_index)
    # Write the results to a file
    print_results(data, output, out_file, in_file)

    

def load_data(data_root, in_file, cancer_type_flag, anatomical_part_flag, proteins_nodes = 'seen.pkl', edges_indeces = 'ei.pkl', conv_prot = 'ens_dic.pkl'):
    """This function load input data and formats it
    """    
    # Import PPI network
    f=open(os.path.join(data_root, proteins_nodes),'rb')
    seen=pickle.load(f)
    f.close()
    f=open(os.path.join(data_root, edges_indeces),'rb')
    ei=pickle.load(f)
    f.close()
    #############
    # device = torch.device('cpu')

    cancer_type_vector = np.zeros((33,), dtype=np.float32)
    cancer_type_vector[cancer_type_flag] = 1
    
    cancer_subtype_vector = np.zeros((25,), dtype=np.float32)
    for i in CANCER_SUBTYPES[cancer_type_flag]:
        cancer_subtype_vector[i] = 1
    
    anatomical_location_vector = np.zeros((52,), dtype=np.float32)
    anatomical_location_vector[anatomical_part_flag] = 1
    cell_type_vector = np.zeros((10,), dtype=np.float32)
    cell_type_vector[CELL_TYPES[cancer_type_flag]] = 1

    pt_tensor_cancer_type = torch.FloatTensor(cancer_type_vector).to(device)
    pt_tensor_cancer_subtype = torch.FloatTensor(cancer_subtype_vector).to(device)
    pt_tensor_anatomical_location = torch.FloatTensor(anatomical_location_vector).to(device)
    pt_tensor_cell_type = torch.FloatTensor(cell_type_vector).to(device)
    edge_index = torch.LongTensor(ei).to(device)
    
    #############
    # Import a dictionary that maps protiens to their coresponding genes from Ensembl database
    f = open(os.path.join(data_root, conv_prot),'rb')
    dicty = pickle.load(f)
    f.close()
    dic = {}
    for d in dicty:
        key = dicty[d]
        if key not in dic:
            dic[key] = {}
            dic[key][d] = 1
    #############
    clin_vec = [] # for clinical data (i.e. number of days to survive, days to death for dead patients and days to last followup for alive patients)
    feat_vecs = [] # list of lists ([[patient1],[patient2],.....[patientN]]) -- [patientX] = [gene_expression_value, diff_gene_expression_value, methylation_value, diff_methylation_value, VCF_value, CNV_value]
    suv_time = [] # list that include wheather a patient is alive or dead (i.e. 0 for dead and 1 for alive)
    f = open(in_file)
    line = f.readlines()
    f.close()
    data = [[0,0,0,0,0,0] for j in range(len(seen)+1)]
    feat_vecs = np.zeros((1, 17186 * 6 + 120), dtype=np.float32)
    for l in line:
        gene, exp, diffexp, methyl, diffmethyl, cnv, snv, clin, surv = l.split('\t')
        exp = float(exp)
        diffexp = float(diffexp)
        methyl = float(methyl)
        diffmethyl = float(diffmethyl)
        cnv = float(cnv)
        snv = float(snv)
        clin = float(clin)
        surv = float(surv)
        if gene in dic:
            for p in dic[gene]:
                if p in seen:
                    data[seen[p]][0] = exp
                    data[seen[p]][1] = diffexp
                    data[seen[p]][2] = methyl
                    data[seen[p]][3] = diffmethyl
                    data[seen[p]][4] = cnv
                    data[seen[p]][5] = snv
#                     data[seen[p]][6] = clin
#                     data[seen[p]][7] = surv
    clin_vec.append(clin)
    suv_time.append(surv)
    vec = np.array(data, dtype=np.float32)
    vec = vec.flatten()
    vec = np.concatenate([
    vec, cancer_type_vector, cancer_subtype_vector,
    anatomical_location_vector, cell_type_vector])
    feat_vecs[0, :] = vec
    labels_days = []
    labels_surv = []
    for days, surv in zip(clin_vec, suv_time):
        labels_days.append(float(days))
        labels_surv.append(float(surv))

    # Train by batch
    dataset = feat_vecs
    #print(dataset.shape)
    labels_days = np.array(labels_days)
    labels_surv = np.array(labels_surv)
    #print(dataset.shape)
    #edge = torch.tensor(ei,dtype=torch.long)
    #x = torch.tensor(data,dtype=torch.float)
    #dataset = Data(x = x,edge_index = edge)
    return dataset, labels_days, labels_surv, edge_index

def load_model(model_file, data, clinical, surv_time, edge_index):
    """The function for loading a pytorch model
    """
    #############
    m = MyNet(edge_index).to(device)
    model = CoxPH(m, tt.optim.Adam(0.0001))    
    #_, features = m(data)
    #print(features)
    model.load_net(model_file)
    prediction = model.predict_surv_df(data)
    #print(prediction)
    fs = features(model.net, torch.from_numpy(data).to(device))
    #print(fs)
    #ev = EvalSurv(prediction, clinical, surv_time)
    #prediction = ev.concordance_td()

    return prediction, fs

def print_results(dataset, results, out_file, in_file):
    """Write results to a file
    """
    prediction, features = results
    file_name = os.path.splitext(in_file)[0] + '_' + out_file
    features = features.data.cpu().numpy()
    with open(file_name, 'w') as f:
        f.write(os.path.splitext(in_file)[0] + '\t')
        f.write(str(prediction) + '\t')
        f.write(str(features) + '\n')

    print(f'***DONE***\n***The results have been written to {file_name}***')
    

if __name__ == '__main__':
    main()