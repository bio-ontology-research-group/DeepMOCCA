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
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, SAGPooling
import click as ck
import gzip
import pickle
import sys

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
    def __init__(self, pt_tensor_cancer_type, pt_tensor_cancer_subtype, pt_tensor_anatomical_location, pt_tensor_cell_type):
        super(MyNet, self).__init__()
        self.pt_tensor_cancer_type = pt_tensor_cancer_type
        self.pt_tensor_cancer_subtype = pt_tensor_cancer_subtype
        self.pt_tensor_anatomical_location = pt_tensor_anatomical_location
        self.pt_tensor_cell_type = pt_tensor_cell_type
        self.conv1 = GCNConv(6,64)
        self.pool1 = SAGPooling(64, ratio=0.70, GNN=GCNConv)
        self.conv2 = GCNConv(64,32)
        self.fc1 = nn.Linear(32,1)

        self.fc2 = nn.Linear(33,1)
        self.fc3 = nn.Linear(25,1)
        self.fc4 = nn.Linear(52,1)
        self.fc5 = nn.Linear(10,1)
        

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, perm, score = self.pool1(x, edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index))
        b=data.y.shape[0]
        x=x.view(b,-1)
        ct = self.fc2(self.pt_tensor_cancer_type.to(device))
        cs = self.fc3(self.pt_tensor_cancer_subtype.to(device))
        al = self.fc4(self.pt_tensor_anatomical_location.to(device))
        cet = self.fc5(self.pt_tensor_cell_type.to(device))
        concat_tensors = torch.cat([ct, cs, al, cet], dim=0)
        x = self.fc1(x)
        concat_tensors = torch.unsqueeze(concat_tensors, 0)
        x = torch.matmul(x, concat_tensors)
        x = x.squeeze(1)
        x = torch.mean(x)
        x = torch.tensor([x])
        return x


@ck.command()
@ck.option('--cancer-type', '-ct', default=0, help='Cancer type index (0-32)')
@ck.option('--anatomical-location', '-al', default=0, help='Anatomical location index (0-51)')
def main(cancer_type, anatomical_location):

    # Import the RDF graph for PPI network
    f = open('seen.pkl','rb')
    seen = pickle.load(f)
    f.close()
    #####################

    f = open('ei.pkl','rb')
    ei = pickle.load(f)
    f.close()

    cancer_type_vector = [0] * 33
    cancer_type_vector[cancer_type] = 1
    
    cancer_subtype_vector = [0] * 25
    for i in CANCER_SUBTYPES[cancer_type]:
        cancer_subtype_vector[i] = 1
    
    anatomical_location_vector = [0] * 52
    anatomical_location_vector[anatomical_location] = 1
    cell_type_vector = [0] * 10
    cell_type_vector[CELL_TYPES[cancer_type]] = 1

    pt_tensor_cancer_type = torch.FloatTensor(cancer_type_vector)
    pt_tensor_cancer_subtype = torch.FloatTensor(cancer_subtype_vector)
    pt_tensor_anatomical_location = torch.FloatTensor(anatomical_location_vector)
    pt_tensor_cell_type = torch.FloatTensor(cell_type_vector)
    
    # Import a dictionary that maps protiens to their coresponding genes by Ensembl database
    f = open('ens_dic.pkl','rb')
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
    with open('data1/prot_names1.txt') as f:
        for line in f:
            tok = line.split()
            d[tok[1]] = tok[0]

    clin = [] # for clinical data (i.e. number of days to survive, days to death for dead patients and days to last followup for alive patients)
    feat_vecs = [] # list of lists ([[patient1],[patient2],.....[patientN]]) -- [patientX] = [gene_expression_value, diff_gene_expression_value, methylation_value, diff_methylation_value, VCF_value, CNV_value]
    suv_time = [] # list that include wheather a patient is alive or dead (i.e. 0 for dead and 1 for alive)
    can_types = ["BRCA"]
    for i in range(len(can_types)):
        # file that contain patients ID with their coressponding 6 differnt files names (i.e. files names for gene_expression, diff_gene_expression, methylation, diff_methylation, VCF and CNV)
        f = open(can_types[i] + '.tsv')
        lines = f.read().splitlines()
        f.close()
        lines = lines[1:]
        count = 0
        for l in tqdm(lines):
            try:
                l = l.split('\t')
                clinical_file = 'cancer_types/TCGA-' + can_types[i] + '/clinical/' + l[6]
                surv_file = 'data1/file/surv/' + l[2]
                myth_file = 'cancer_types/TCGA-' + can_types[i] + '/myth/' + l[3]
                diff_myth_file = 'data1/file/diff_myth/' + l[1]
                exp_norm_file = 'cancer_types/TCGA-' + can_types[i] + '/exp_count/col/' + l[-1]
                diff_exp_norm_file = 'data1/file/diff_exp/' + l[0]
                cnv_file = 'cancer_types/TCGA-' + can_types[i] + '/cnv/' + l[4] + '.txt'
                vcf_file = 'cancer_types/TCGA-' + can_types[i] + '/vcf/output/' + 'OutputAnnoFile_' + l[5] + '.hg38_multianno.txt.dat'
                # Check if all 6 files are exist for a patient (that's because for some patients, their survival time not reported)
                all_files = [
                    clinical_file, surv_file, myth_file, diff_myth_file,
                    exp_norm_file, diff_exp_norm_file, cnv_file, vcf_file]
                for fname in all_files:
                    if not os.path.exists(fname):
                        print('File ' + fname + ' does not exist!')
                        sys.exit(1)
                f = open(clinical_file)
                content = f.read().strip()
                f.close()
                clin.append(content)
                f = open(surv_file)
                content = f.read().strip()
                f.close()
                suv_time.append(content)
                temp_myth=myth_data(myth_file, seen, d, dic)
                feat_vecs.append(
                    get_data(
                        exp_norm_file, diff_exp_norm_file, diff_myth_file,
                        cnv_file, vcf_file, temp_myth, seen, dic))
            except Exception as e:
                print(e)
                sys.exit(1)
    clinn = []
    for i in clin:
        clinn.append(i.replace("-",""))
    clinnn = []
    for i in clinn:
        if i != "":
            clinnn.append(i)
        else:
            i = "0"
            clinnn.append(i)
                                     
    label = [float(i) for i in clinnn]

    # Train by batch
    dataset=[]
    edge=torch.tensor(ei,dtype=torch.long)
    i=0
    for e in range(len(feat_vecs)):
        x=torch.tensor(feat_vecs[e],dtype=torch.float)
        labell = label[e]
        # event=torch.tensor(suv_time[e])
        dataset.append(Data(x=x,edge_index=edge,y=torch.tensor([labell])))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CoxPH(MyNet(pt_tensor_cancer_type, pt_tensor_cancer_subtype, pt_tensor_anatomical_location, pt_tensor_cell_type).to(device), tt.optim.Adam(0.001))

    # Each time test on a specific cancer type
    total_cancers = ["TCGA-BRCA"]
    for i in range(len(total_cancers)):
        test_set = [d for t, d in zip(total_cancers, dataset) if t == total_cancers[i]]
        train_set = [d for t, d in zip(total_cancers, dataset) if t != total_cancers[i]]

        # Split 70% from all 32 cancers and test on 15% of a specific one
        train_size = len(train_set)
        train_indices = list(range(train_size))
        np.random.shuffle(train_indices)
        
        test_size = len(test_set)
        test_indices = list(range(test_size))
        np.random.shuffle(test_indices)
        test_split_index = int(np.floor(0.5 * test_size))

        train_idx = train_indices 
        val_idx = test_indices[:test_split_index]
        test_idx = test_indices[test_split_index:]

        train_dataset = SubsetRandomSampler(train_idx)
        val_dataset = SubsetRandomSampler(val_idx)
        test_dataset = SubsetRandomSampler(test_idx)

        callbacks = [tt.callbacks.EarlyStopping()]

        trained_model = model.fit(
            train_dataset, train_dataset.y, batch_size=3, epochs=100,
            callbacks=callbacks, val_data=val_dataset, val_batch_size=3)

        # Compute the evaluation measurements
        for t in test_dataset:
            predicted = model.predict_surv_df(t[x])
            c_index = EvalSurv(predicted, t[y],t[event]).concordance_td()
            mse = mean_squared_error(t[y], predicted)
            rmse = math.sqrt(mse)

# Import and pre-process methylation data
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

# Import gene expression files and Pre-process them

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

    # Import differential gene expression files and Pre-process them             
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
        if gene in dic:
            for p in dic[gene]:
                if p in seen:
                    output[seen[p]][2]=diffexp

    # Import differential methylation files and Pre-process them             
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
        if gene in dic:
            for p in dic[gene]:
                if p in seen:
                    output[seen[p]][3]=diffmethy
    # Import CNV files and Pre-process them
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
    # Import VCF files and Pre-process them            
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


if __name__ == '__main__':
    main()
