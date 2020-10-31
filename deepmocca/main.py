#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import os
import sys
import logging

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGPooling
from rdflib import Graph

@ck.command()
@ck.option('--data-root', '-dr', default='data/', help='Data root folder', required=True)
@ck.option('--in-file', '-if', help='Input file', required=True)
@ck.option('--model-file', '-mf', default='model.h5', help='Pytorch model file')
@ck.option('--out-file', '-of', default='results.tsv', help='Output result file')
def main(data_root, in_file, model_file, out_file):
    # Check data folder and required files
    try:
        if os.path.exists(data_root):
            in_file = os.path.join(data_root, in_file)
            model_file = os.path.join(data_root, model_file)
            if not os.path.exists(in_file):
                raise Exception(f'Input file ({go_file}) is missing!')
            if not os.path.exists(model_file):
                raise Exception(f'Model file ({model_file}) is missing!')
        else:
            raise Exception(f'Data folder {data_root} does not exist!')
    except Exception as e:
        logging.error(e)
        sys.exit(1)

    # Read input data
    data = load_data(in_file)
    # Load GCN model
    model = load_model(model_file)
    # Run model
    output = model(data)
    # Write the results to a file
    print_results(output, out_file)

    

def load_data(in_file):
    """This function load input data and formats it
    """    
    # Import the RDF graph for PPI network
    g = Graph()
    g.parse("rdf_string.ttl", format="turtle")
    ##############
    seen={}
    done={}
    ei=[[],[]]
    ii=0
    def get_name(raw):
        index=raw.rfind('_')
        return raw[index+1:-1]
    for i,j,k in g:
        sbj=get_name(i.n3())
        obj=get_name(k.n3())
        if sbj not in seen:
            seen[sbj]=ii
            ii+=1
        if obj not in seen:
            seen[obj]=ii
            ii+=1
        ei[0].append((seen[sbj]))
        ei[1].append((seen[obj]))
    for i,j,k in g:
    sbj=get_name(i.n3())
    obj=get_name(k.n3())
    #############
    # Import a dictionary that maps protiens to their coresponding genes from Ensembl database
    import pickle
    f=open('ens_dic.pkl','rb')
    dicty=pickle.load(f)
    f.close()
    dic={}
    for d in dicty:
        key=dicty[d]
        if key not in dic:
            dic[key]={}
        dic[key][d]=1
    #############
    f=open(in_file)
    line=f.readlines()
    f.close()
    data=[[0,0,0,0,0,0] for j in range(ii+1)]
    for l in line:
        gene,exp,diffexp,methyl,diffmethyl,cnv,snv=l.split('\t')
        exp=float(exp)
        diffexp=float(diffexp)
        methyl=float(methyl)
        diffmethyl=float(diffmethyl)
        cnv=float(cnv)
        snv=float(snv)
        if gene in seen:
            data[seen[gene]][0]=exp
            data[seen[gene]][1]=diffexp
            data[seen[gene]][2]=methyl
            data[seen[gene]][3]=diffmethyl
            data[seen[gene]][4]=cnv
            data[seen[gene]][5]=snv
    return data

def load_model(model_file):
    """The function for loading a pytorch model
    """
    # Define the model
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GCNConv(6,64)
            self.pool1 = SAGPooling(64, ratio=0.70, GNN=GCNConv)
            self.conv2 = GCNConv(64,32)
            self.fc1 = nn.Linear(32,8)

            self.fc2 = nn.Linear(33,1)
            self.fc3 = nn.Linear(25,1)
            self.fc4 = nn.Linear(52,1)
            self.fc5 = nn.Linear(10,1)

        def forward(self, dataA):
            x, edge_index, batch = dataA.x, dataA.edge_index, dataA.batch
            x = F.relu(self.conv1(x, edge_index))
            x, edge_index, _, batch, perm, score = self.pool1(x, edge_index, None, batch)
            x = F.relu(self.conv2(x, edge_index))
            b=data.y.shape[0]
            x=x.view(b,-1)
#             ct = self.fc2(pt_tensor_cancer_type.to(device))
#             cs = self.fc3(pt_tensor_cancer_subtype.to(device))
#             al = self.fc4(pt_tensor_anatomical_location.to(device))
#             cet = self.fc5(pt_tensor_cell_type.to(device))
#             concat_tensors = torch.cat([ct, cs, al, cet], dim=0)
            x = self.fc1(x)
#             x = torch.matmul(x, concat_tensors)
            return x
        
    model = Net(*args, **kwargs)
    model.load_state_dict(torch.load(model_file))
    model.eval()

def print_results(results, out_file):
    """Write results to a file
    """
    with open(out_file, 'w') as f:
        for item in results:
            f.write(item + '\n')
    

if __name__ == '__main__':
    main()
