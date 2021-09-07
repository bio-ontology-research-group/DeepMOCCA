#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import statistics


# In[ ]:


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
        x = F.relu(self.conv1(x, self.edge_index))
        x, edge_index, _, batch, perm, score = self.pool1(x, self.edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index))
        x = gmp(x, batch)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x


# In[ ]:


# Import the pre-processed methylation data
def myth_data(fname, seen, d, dic):
    f=open(fname)
    line=f.readlines()
    f.close()
    output=[[0,0,0,0,0,0] for j in range(len(seen)+1)]
    for l in line:
        temp=[]
        trans,myth=l.split('\t')
        temp=trans.split(';')
        myth=float(myth)
        for x in temp:
            index=x.find('.')
            if index<1:
                index=len(x)
            x=x[:index]
            if x in d:
                gen = d[x]
            if gen in dic:
                for p in dic[gen]:
                    if p in seen:
                        output[seen[p]][0]=myth
    return output

# Import the pre-processed gene expression files

def get_data(expname,diffexpname,diffmethyname,cnvname,vcfname,output, seen, dic):
    f=gzip.open(expname,'rt')
    line=f.readlines()
    f.close()
    for l in line:
        gene,exp=l.split('\t')
        prev=gene
        index=gene.find('.')
        if index<1:
            index=len(gene)
        gene=gene[:index]
        exp=float(exp)
        if gene in dic:
            for p in dic[gene]:
                if p in seen:
                    output[seen[p]][1]=exp

    # Import the pre-processed differential gene expression files            
    f=gzip.open(diffexpname,'rt')
    line=f.readlines()
    f.close()    
    for l in line:
        gene,diffexp=l.split('\t')
        prev=gene
        index=gene.find('.')
        if index<1:
            index=len(gene)
        gene=gene[:index]
        diffexp=float(diffexp)
        if gene in seen:
            output[seen[gene]][2]=diffexp

    # Import the pre-processed differential methylation files           
    f=open(diffmethyname)
    line=f.readlines()
    f.close()    
    for l in line:
        gene,diffmethy=l.split('\t')
        prev=gene
        index=gene.find('.')
        if index<1:
            index=len(gene)
        gene=gene[:index]
        diffmethy=float(diffmethy)
        if gene in seen:
            output[seen[gene]][3]=diffmethy
    # Import the pre-processed CNV files
    f=open(cnvname)
    line=f.readlines()
    f.close()    
    for l in line:
        gene,cnv=l.split('\t')
        prev=gene
        index=gene.find('.')
        if index<1:
            index=len(gene)
        gene=gene[:index]
        cnv=float(cnv)
        if gene in dic:
            for p in dic[gene]:
                if p in seen:
                    output[seen[p]][4]=cnv                        
    # Import the pre-processed VCF files          
    f=open(vcfname)
    line=f.readlines()
    f.close()    
    for l in line:
        gene,score=l.split('\t')
        score=float(score)
        if gene in dic:
            for p in dic[gene]:
                if p in seen:
                    output[seen[p]][5]=score
                
    return output


# In[ ]:


# This value should be changed for train each cancer type (0--32)
# cancer_type = 0
can_types = ["TCGA-BRCA","TCGA-ACC","TCGA-BLCA","TCGA-CESC","TCGA-CHOL","TCGA-COAD","TCGA-DLBC","TCGA-ESCA","TCGA-GBM","TCGA-HNSC","TCGA-KICH","TCGA-KIRC","TCGA-KIRP","TCGA-LAML","TCGA-LGG","TCGA-LIHC","TCGA-LUAD","TCGA-LUSC","TCGA-MESO","TCGA-OV","TCGA-PAAD","TCGA-PCPG","TCGA-PRAD","TCGA-READ","TCGA-SARC","TCGA-SKCM","TCGA-STAD","TCGA-TGCT","TCGA-THCA","TCGA-THYM","TCGA-UCEC","TCGA-UCS","TCGA-UVM"]
# anatomical_location = 0
# For each cancer type this needs to be changed
data_root = 'data_root/'
# def main(cancer_type, anatomical_location):

# Import the RDF graph for PPI network
f = open(data_root+'seen.pkl','rb')
seen = pickle.load(f)
f.close()
#####################

f = open(data_root+'ei.pkl','rb')
ei = pickle.load(f)
f.close()

global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# cancer_type_vector = np.zeros((33,), dtype=np.float32)
# cancer_type_vector[cancer_type] = 1

# cancer_subtype_vector = np.zeros((25,), dtype=np.float32)
# for i in CANCER_SUBTYPES[cancer_type]:
#     cancer_subtype_vector[i] = 1

# anatomical_location_vector = np.zeros((52,), dtype=np.float32)
# anatomical_location_vector[anatomical_location] = 1
# cell_type_vector = np.zeros((10,), dtype=np.float32)
# cell_type_vector[CELL_TYPES[cancer_type]] = 1

# pt_tensor_cancer_type = torch.FloatTensor(cancer_type_vector).to(device)
# pt_tensor_cancer_subtype = torch.FloatTensor(cancer_subtype_vector).to(device)
# pt_tensor_anatomical_location = torch.FloatTensor(anatomical_location_vector).to(device)
# pt_tensor_cell_type = torch.FloatTensor(cell_type_vector).to(device)
edge_index = torch.LongTensor(ei).to(device)

# Import a dictionary that maps protiens to their coresponding genes by Ensembl database
f = open(data_root+'ens_dic.pkl','rb')
dicty = pickle.load(f)
f.close()
dic = {}
for d in dicty:
    key=dicty[d]
    if key not in dic:
        dic[key]={}
    dic[key][d]=1

# Build a dictionary from ENSG -- ENST
d = {}
with open(data_root+'prot_names1.txt') as f:
    for line in f:
        tok = line.split()
        d[tok[1]] = tok[0]


# In[ ]:


clin = [] # for clinical data (i.e. number of days to survive, days to death for dead patients and days to last followup for alive patients)
feat_vecs = [] # list of lists ([[patient1],[patient2],.....[patientN]]) -- [patientX] = [gene_expression_value, diff_gene_expression_value, methylation_value, diff_methylation_value, VCF_value, CNV_value]
suv_time = [] # list that include wheather a patient is alive or dead (i.e. 0 for dead and 1 for alive)
for ii in range(len(can_types)):
    # file that contain patients ID with their coressponding 6 differnt files names (i.e. files names for gene_expression, diff_gene_expression, methylation, diff_methylation, VCF and CNV)
    f = open('mapping_files/' + can_types[ii] + '.tsv')
    lines = f.read().splitlines()
    f.close()
    lines = lines[1:]
    count = 0
    feat_vecs = np.zeros((len(lines), 17186 * 6), dtype=np.float32)
    i = 0
    for l in tqdm(lines):
        l = l.split('\t')
        clinical_file = l[6]
        surv_file = l[2]
        myth_file = 'cancer_types/' + can_types[ii] + '/myth/' + l[3]
        diff_myth_file = 'cancer_types/' + can_types[ii] + '/diff_myth/' + l[1]
        exp_norm_file = 'cancer_types/' + can_types[ii] + '/exp_upper/row/' + l[-1]
        diff_exp_norm_file = 'cancer_types/' + can_types[ii] + '/diff_exp/' + l[0]
        cnv_file = 'cancer_types/' + can_types[ii] + '/cnv/' + l[4] + '.txt'
        vcf_file = 'cancer_types/' + can_types[ii] + '/vcf/output/' + 'OutputAnnoFile_' + l[5] + '.hg38_multianno.txt.dat'
        # Check if all 6 files are exist for a patient (that's because for some patients, their survival time not reported)
        all_files = [
            myth_file, diff_exp_norm_file, diff_myth_file,
            exp_norm_file, cnv_file, vcf_file]
        for fname in all_files:
            if not os.path.exists(fname):
                print('File ' + fname + ' does not exist!')
                sys.exit(1)
        clin.append(clinical_file)
        suv_time.append(surv_file)
        temp_myth=myth_data(myth_file, seen, d, dic)
        vec = np.array(
            get_data(
                exp_norm_file, diff_exp_norm_file, diff_myth_file,
                cnv_file, vcf_file, temp_myth, seen, dic), dtype=np.float32)
        vec = vec.flatten()
#         vec = np.concatenate([
#             vec, cancer_type_vector, cancer_subtype_vector,
#             anatomical_location_vector, cell_type_vector])
        feat_vecs[i, :] = vec
        i += 1
    print("Loading data ...... Done")
    min_max_scaler = MinMaxScaler()
    labels_days = []
    labels_surv = []
    for days, surv in zip(clin, suv_time):
        labels_days.append(float(days))
        labels_surv.append(float(surv))

    # Train by batch
    dataset = feat_vecs
    labels_days = np.array(labels_days)
    labels_surv = np.array(labels_surv)

    censored_index = []
    uncensored_index = []
    for i in range(len(dataset)):
        if labels_surv[i] == 1:
            censored_index.append(i)
        else:
            uncensored_index.append(i)
    model = CoxPH(MyNet(edge_index).to(device), tt.optim.Adam(0.0001))

    censored_index = np.array(censored_index)
    uncensored_index = np.array(uncensored_index)

    # names = ["TCGA-ACC","TCGA-BLCA","TCGA-BRCA","TCGA-CESC","TCGA-CHOL","TCGA-COAD","TCGA-DLBC","TCGA-ESCA","TCGA-GBM","TCGA-HNSC","TCGA-KICH","TCGA-KIRC","TCGA-KIRP","TCGA-LAML","TCGA-LGG","TCGA-LIHC","TCGA-LUAD","TCGA-LUSC","TCGA-MESO","TCGA-OV","TCGA-PAAD","TCGA-PCPG","TCGA-PRAD","TCGA-READ","TCGA-SARC","TCGA-SKCM","TCGA-STAD","TCGA-TGCT","TCGA-THCA","TCGA-THYM","TCGA-UCEC","TCGA-UCS","TCGA-UVM"]

    # for cancer_type in range(len(names)):
    ev_ = []
    splits = 5
    best_cindex = 0
    for fold in range(splits):
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

        train_data = dataset[train_idx]
        train_data = min_max_scaler.fit_transform(train_data)
        train_labels_days = labels_days[train_idx]
        train_labels_surv = labels_surv[train_idx]
        train_labels = (train_labels_days, train_labels_surv)

        val_data = dataset[valid_idx]
        val_data = min_max_scaler.transform(val_data)
        val_labels_days = labels_days[valid_idx]
        val_labels_surv = labels_surv[valid_idx]
        test_data = dataset[test_idx]
        test_data = min_max_scaler.transform(test_data)
        test_labels_days = labels_days[test_idx]
        test_labels_surv = labels_surv[test_idx]
        val_labels = (val_labels_days, val_labels_surv)


        callbacks = [tt.callbacks.EarlyStopping()]
        batch_size = 16
        epochs = 100
        val = (val_data, val_labels)
        log = model.fit(
            train_data, train_labels, batch_size, epochs, callbacks, False,
            val_data=val,
            val_batch_size=batch_size)
        train = train_data, train_labels
        # Compute the evaluation measurements
        _ = model.compute_baseline_hazards(*train)
        surv = model.predict_surv_df(test_data)
        ev = EvalSurv(surv, test_labels_days, test_labels_surv)
        ev_.append(ev.concordance_td())

        if ev.concordance_td() > best_cindex:
            best_cindex = ev.concordance_td()
            with open('test_'+str(fold+1)+'.pkl','wb') as f:
                pickle.dump(test_data, f)

            with open('test_labels_days_'+str(fold+1)+'.pkl','wb') as f:
                pickle.dump(test_labels_days, f)

            with open('test_labels_surv_'+str(fold+1)+'.pkl','wb') as f:
                pickle.dump(test_labels_surv, f)

    print(can_types[ii])            
    print(str(statistics.mean(ev_))+"["+str(min(ev_))+"-"+str(max(ev_))+"]")


# In[ ]:




