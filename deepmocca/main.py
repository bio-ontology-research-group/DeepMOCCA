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
            in_file = os.path.join(data_root, in_file)
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
    data = load_data(in_file)
    # Load and Run GCN model
    output = load_model(model_file, data, cancer_type_flag, anatomical_part_flag)
    # Write the results to a file
    print_results(data, output, out_file)

    

def load_data(in_file, rdf_graph = 'rdf_string.ttl', conv_prot = 'ens_dic.pkl'):
    """This function load input data and formats it
    """    
    # Import the RDF graph for PPI network
    g = Graph()
    g.parse(rdf_graph, format="turtle")
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
    f = open(conv_prot,'rb')
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
        exp = np.log(float(exp))
        diffexp = np.log(float(diffexp))
        methyl = np.log(float(methyl))
        diffmethyl = np.log(float(diffmethyl))
        cnv = np.log(float(cnv))
        snv = np.log(float(snv))
        if gene in seen:
            data[seen[gene]][0] = exp
            data[seen[gene]][1] = diffexp
            data[seen[gene]][2] = methyl
            data[seen[gene]][3] = diffmethyl
            data[seen[gene]][4] = cnv
            data[seen[gene]][5] = snv
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
    t = int(cancer_type_flag)
    a = int(anatomical_part_flag)
    cancer_type[t-1] = 1
    anatomical_location [a-1] = 1
    for i in [4,7,4,22]:
        cancer_subtype[i] = 1
    for j in [0]:
        cell_type[j] = 1
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
            self.fc1 = nn.Linear(32,8)

            self.fc2 = nn.Linear(33,1)
            self.fc3 = nn.Linear(25,1)
            self.fc4 = nn.Linear(52,1)
            self.fc5 = nn.Linear(10,1)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = F.relu(self.conv1(x, edge_index))
            x, edge_index, _, batch, perm, score = self.pool1(x, edge_index, None, None, None)
            x = F.relu(self.conv2(x, edge_index))
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
            return x
        
    model = MyNet()
    model.load_state_dict(torch.load(model_file))
    model.eval()
    prediction = model(data)
    with open('results_rep.tsv', 'w') as f:
        for item in model.conv2.weight.data:
            f.write(str(item.float()) + '\n')
    return prediction

def print_results(dataset, results, out_file):
    """Write results to a file
    """
    with open(out_file, 'w') as f:
        for item in results:
            f.write(str(item.item()) + '\n')
            
    

if __name__ == '__main__':
    main()
