#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import pickle
import gzip
import os
import sys
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.data import Data
from rdflib import Graph

@ck.command()
@ck.option('--data-root', '-dr', default='data/', help='Data root folder', required=True)
@ck.option('--in-file', '-if', help='Input file', required=True)
@ck.option('--model-file', '-mf', default='model.h5', help='Pytorch model file')
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
    data = load_data(data_root, in_file)
    # Load and Run GCN model
    output = load_model(model_file, data, cancer_type_flag, anatomical_part_flag)
    # Write the results to a file
    print_results(data, output, out_file, in_file)

    

def load_data(data_root, in_file, rdf_graph = 'rdf_string.ttl', conv_prot = 'ens_dic.pkl'):
    """This function load input data and formats it
    """    
    # Import the RDF graph for PPI network
    g = Graph()
    g.parse(os.path.join(data_root, rdf_graph), format="turtle")
    ##############
    seen = {}
    done = {}
    ei = [[],[]]
    ii = 0
    def get_name(raw):
        index=raw.rfind('_')
        return raw[index+1:-1]

    for i,j,k in g:
        sbj = get_name(i.n3())
        obj = get_name(k.n3())
        if sbj not in seen:
            seen[sbj] = ii
            ii += 1
        if obj not in seen:
            seen[obj] = ii
            ii += 1
            ei[0].append((seen[sbj]))
            ei[1].append((seen[obj]))
    for i,j,k in g:
        sbj = get_name(i.n3())
        obj = get_name(k.n3())
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
    f = open(in_file)
    line = f.readlines()
    f.close()
    data = [[0,0,0,0,0,0] for j in range(ii+1)]
    for l in line:
        gene, exp, diffexp, methyl, diffmethyl, cnv, snv = l.split('\t')
        # Normalize the features
        exp = float(exp)
        diffexp = float(diffexp)
        methyl = float(methyl)
        diffmethyl = float(diffmethyl)
        cnv = float(cnv)
        snv = float(snv)
        if gene in dic:
            for p in dic[gene]:
                if p in seen:
                    data[seen[p]][0] = exp
                    data[seen[p]][1] = diffexp
                    data[seen[p]][2] = methyl
                    data[seen[p]][3] = diffmethyl
                    data[seen[p]][4] = cnv
                    data[seen[p]][5] = snv
    edge = torch.tensor(ei,dtype=torch.long)
    x = torch.tensor(data,dtype=torch.float)
    dataset = Data(x = x,edge_index = edge)
    return dataset

def load_model(model_file, data, cancer_type_flag, anatomical_part_flag):
    """The function for loading a pytorch model
    """
    #############
    cancer_type = [0] * 33
    cancer_subtype = [0] * 25
    anatomical_location = [0] * 52
    cell_type = [0] * 10
    if cancer_type_flag == '1':
        for i in [0,12,7,14,4,1,6,2,3]:
            cancer_subtype[i] = 1
        cell_type[0] = 1
    elif cancer_type_flag == '2':
        for i in [4]:
            cancer_subtype[i] = 1
        cell_type[0] = 1
    elif cancer_type_flag == '3':
        for i in [5,4,14,6]:
            cancer_subtype[i] = 1
        cell_type[0] = 1
    elif cancer_type_flag == '4':
        for i in [6,4,12,7]:
            cancer_subtype[i] = 1
        cell_type[0] = 1
    elif cancer_type_flag == '5':
        for i in [4]:
            cancer_subtype[i] = 1
        cell_type[0] = 1
    elif cancer_type_flag == '6':
        for i in [6,4,12,7]:
            cancer_subtype[i] = 1
        cell_type[0] = 1
    elif cancer_type_flag == '7':
        for i in [8]:
            cancer_subtype[i] = 1
        cell_type[1] = 1
    elif cancer_type_flag == '8':
        for i in [6,4,12]:
            cancer_subtype[i] = 1
        cell_type[0] = 1
    elif cancer_type_flag == '9':
        for i in [9]:
            cancer_subtype[i] = 1
        cell_type[2] = 1
    elif cancer_type_flag == '10':
        for i in [6]:
            cancer_subtype[i] = 1
        cell_type[0] = 1
    elif cancer_type_flag == '11':
        for i in [4]:
            cancer_subtype[i] = 1
        cell_type[3] = 1
    elif cancer_type_flag == '12':
        for i in [4]:
            cancer_subtype[i] = 1
        cell_type[0] = 1
    elif cancer_type_flag == '13':
        for i in [4]:
            cancer_subtype[i] = 1
        cell_type[0] = 1
    elif cancer_type_flag == '14':
        for i in [10]:
            cancer_subtype[i] = 1
        cell_type[4] = 1
    elif cancer_type_flag == '15':
        for i in [9]:
            cancer_subtype[i] = 1
        cell_type[2] = 1
    elif cancer_type_flag == '16':
        for i in [4]:
            cancer_subtype[i] = 1
        cell_type[0] = 1
    elif cancer_type_flag == '17':
        for i in [4,11,12]:
            cancer_subtype[i] = 1
        cell_type[0] = 1
    elif cancer_type_flag == '18':
        for i in [6]:
            cancer_subtype[i] = 1
        cell_type[0] = 1
    elif cancer_type_flag == '19':
        for i in [13]:
            cancer_subtype[i] = 1
        cell_type[5] = 1
    elif cancer_type_flag == '20':
        for i in [12]:
            cancer_subtype[i] = 1
        cell_type[0] = 1
    elif cancer_type_flag == '21':
        for i in [0,4,12,14]:
            cancer_subtype[i] = 1
        cell_type[0] = 1
    elif cancer_type_flag == '22':
        for i in [15]:
            cancer_subtype[i] = 1
        cell_type[6] = 1
    elif cancer_type_flag == '23':
        for i in [4,0,12]:
            cancer_subtype[i] = 1
        cell_type[0] = 1
    elif cancer_type_flag == '24':
        for i in [4,12]:
            cancer_subtype[i] = 1
        cell_type[0] = 1
    elif cancer_type_flag == '25':
        for i in [16,17,18,19,20]:
            cancer_subtype[i] = 1
        cell_type[7] = 1
    elif cancer_type_flag == '26':
        for i in [20]:
            cancer_subtype[i] = 1
        cell_type[8] = 1
    elif cancer_type_flag == '27':
        for i in [4,12]:
            cancer_subtype[i] = 1
        cell_type[0] = 1
    elif cancer_type_flag == '28':
        for i in [22]:
            cancer_subtype[i] = 1
        cell_type[9] = 1
    elif cancer_type_flag == '29':
        for i in [4,14]:
            cancer_subtype[i] = 1
        cell_type[0] = 1
    elif cancer_type_flag == '30':
        for i in [23]:
            cancer_subtype[i] = 1
        cell_type[0] = 1
    elif cancer_type_flag == '31':
        for i in [4,12,14]:
            cancer_subtype[i] = 1
        cell_type[0] = 1
    elif cancer_type_flag == '32':
        for i in [24]:
            cancer_subtype[i] = 1
        cell_type[0] = 1
    elif cancer_type_flag == '33':
        for i in [21]:
            cancer_subtype[i] = 1
        cell_type[8] = 1
    else:
        print('!!The Cancer Type You Entered is NOT Correct!!')
            
    t = int(cancer_type_flag)
    a = int(anatomical_part_flag)
    cancer_type[t-1] = 1
    anatomical_location [a-1] = 1

    pt_tensor_cancer_type = torch.FloatTensor(cancer_type)
    pt_tensor_cancer_subtype = torch.FloatTensor(cancer_subtype)
    pt_tensor_anatomical_location = torch.FloatTensor(anatomical_location)
    pt_tensor_cell_type = torch.FloatTensor(cell_type)
    
    # Define the model
    class MyNet(nn.Module):
        def __init__(self):
            super(MyNet, self).__init__()
            self.conv1 = GCNConv(6,64)
            self.pool1 = SAGPooling(64, ratio=0.70, GNN=GCNConv)
            self.conv2 = GCNConv(64,32)
            self.fc1 = nn.Linear(32,1)

            self.fc2 = nn.Linear(33,1)
            self.fc3 = nn.Linear(25,1)
            self.fc4 = nn.Linear(52,1)
            self.fc5 = nn.Linear(10,1)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = F.relu(self.conv1(x, edge_index))
            x, edge_index, _, batch, perm, score = self.pool1(x, edge_index, None, None, None)
            x = F.relu(self.conv2(x, edge_index))
            x = gmp(x,batch)
            features = x
            x=x.view(1,-1)
            ct = self.fc2(pt_tensor_cancer_type)
            cs = self.fc3(pt_tensor_cancer_subtype)
            al = self.fc4(pt_tensor_anatomical_location)
            cet = self.fc5(pt_tensor_cell_type)
            concat_tensors = torch.cat([ct, cs, al, cet], dim=0)
            x = self.fc1(x)
            concat_tensors = torch.unsqueeze(concat_tensors, 0)
            x = torch.matmul(x, concat_tensors)
            x = x.squeeze(1)
            x = torch.mean(x)
            x = torch.tensor([x])
            return x, features
        
    model = MyNet()
    model.load_state_dict(torch.load(model_file))
    model.eval()
    prediction, features = model(data)

    return prediction, features

def print_results(dataset, results, out_file, in_file):
    """Write results to a file
    """
    prediction, features = results
    file_name = os.path.splitext(in_file)[0] + '_' + out_file
    features = features.data.cpu().numpy()
    with open(file_name, 'w') as f:
        f.write(os.path.splitext(in_file)[0] + '\t')
        for item in prediction:
            f.write(str(item.item()) + '\t')
        f.write(str(features) + '\n')
            

    print(f'***DONE***\n***The results have been written to {file_name}***')
    

if __name__ == '__main__':
    main()
