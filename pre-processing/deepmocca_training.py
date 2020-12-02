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
import matplotlib.pyplot as plt
from pycox.models import CoxPH, LogisticHazard
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, SAGPooling
from torch_geometric.nn import global_max_pool as gmp

from rdflib import Graph
# Import the RDF graph for PPI network
g = Graph()
g.parse("rdf_string.ttl", format="turtle")

cancer_type = [0] * 33
cancer_subtype = [0] * 25
anatomical_location = [0] * 52
cell_type = [0] * 10

s = 4
r = 6

cancer_type[s-1] = 1
anatomical_location [r-1] = 1

if cancer_type == '1':
    for i in [0,12,7,14,4,1,6,2,3]:
        cancer_subtype[i] = 1
    cell_type[0] = 1
elif cancer_type == '2':
    for i in [4]:
        cancer_subtype[i] = 1
    cell_type[0] = 1
elif cancer_type == '3':
    for i in [5,4,14,6]:
        cancer_subtype[i] = 1
    cell_type[0] = 1
elif cancer_type == '4':
    for i in [6,4,12,7]:
        cancer_subtype[i] = 1
    cell_type[0] = 1
elif cancer_type == '5':
    for i in [4]:
        cancer_subtype[i] = 1
    cell_type[0] = 1
elif cancer_type == '6':
    for i in [6,4,12,7]:
        cancer_subtype[i] = 1
    cell_type[0] = 1
elif cancer_type == '7':
    for i in [8]:
        cancer_subtype[i] = 1
    cell_type[1] = 1
elif cancer_type == '8':
    for i in [6,4,12]:
        cancer_subtype[i] = 1
    cell_type[0] = 1
elif cancer_type == '9':
    for i in [9]:
        cancer_subtype[i] = 1
    cell_type[2] = 1
elif cancer_type == '10':
    for i in [6]:
        cancer_subtype[i] = 1
    cell_type[0] = 1
elif cancer_type == '11':
    for i in [4]:
        cancer_subtype[i] = 1
    cell_type[3] = 1
elif cancer_type == '12':
    for i in [4]:
        cancer_subtype[i] = 1
    cell_type[0] = 1
elif cancer_type == '13':
    for i in [4]:
        cancer_subtype[i] = 1
    cell_type[0] = 1
elif cancer_type == '14':
    for i in [10]:
        cancer_subtype[i] = 1
    cell_type[4] = 1
elif cancer_type == '15':
    for i in [9]:
        cancer_subtype[i] = 1
    cell_type[2] = 1
elif cancer_type == '16':
    for i in [4]:
        cancer_subtype[i] = 1
    cell_type[0] = 1
elif cancer_type == '17':
    for i in [4,11,12]:
        cancer_subtype[i] = 1
    cell_type[0] = 1
elif cancer_type == '18':
    for i in [6]:
        cancer_subtype[i] = 1
    cell_type[0] = 1
elif cancer_type == '19':
    for i in [13]:
        cancer_subtype[i] = 1
    cell_type[5] = 1
elif cancer_type == '20':
    for i in [12]:
        cancer_subtype[i] = 1
    cell_type[0] = 1
elif cancer_type == '21':
    for i in [0,4,12,14]:
        cancer_subtype[i] = 1
    cell_type[0] = 1
elif cancer_type == '22':
    for i in [15]:
        cancer_subtype[i] = 1
    cell_type[6] = 1
elif cancer_type == '23':
    for i in [4,0,12]:
        cancer_subtype[i] = 1
    cell_type[0] = 1
elif cancer_type == '24':
    for i in [4,12]:
        cancer_subtype[i] = 1
    cell_type[0] = 1
elif cancer_type == '25':
    for i in [16,17,18,19,20]:
        cancer_subtype[i] = 1
    cell_type[7] = 1
elif cancer_type == '26':
    for i in [20]:
        cancer_subtype[i] = 1
    cell_type[8] = 1
elif cancer_type == '27':
    for i in [4,12]:
        cancer_subtype[i] = 1
    cell_type[0] = 1
elif cancer_type == '28':
    for i in [22]:
        cancer_subtype[i] = 1
    cell_type[9] = 1
elif cancer_type == '29':
    for i in [4,14]:
        cancer_subtype[i] = 1
    cell_type[0] = 1
elif cancer_type == '30':
    for i in [23]:
        cancer_subtype[i] = 1
    cell_type[0] = 1
elif cancer_type == '31':
    for i in [4,12,14]:
        cancer_subtype[i] = 1
    cell_type[0] = 1
elif cancer_type == '32':
    for i in [24]:
        cancer_subtype[i] = 1
    cell_type[0] = 1
elif cancer_type == '33':
    for i in [21]:
        cancer_subtype[i] = 1
    cell_type[8] = 1
    
    
pt_tensor_cancer_type = torch.FloatTensor(cancer_type)
pt_tensor_cancer_subtype = torch.FloatTensor(cancer_subtype)
pt_tensor_anatomical_location = torch.FloatTensor(anatomical_location)
pt_tensor_cell_type = torch.FloatTensor(cell_type)

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
for s in seen:
    print(s,seen[s])
    break
    
for i,j,k in g:
    sbj=get_name(i.n3())
    obj=get_name(k.n3())
    print(sbj,obj)
    break
    
# Import a dictionary that maps protiens to their coresponding genes by Ensembl database
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
print(len(seen))

# Build a dictionary from ENSG -- ENST
d = {}
with open('prot_names1.txt') as f:
    for line in f:
        tok = line.split()
        d[tok[1]] = tok[0]
        
# Import and pre-process methylation data
def myth_data(fname):
    f=open(fname)
    line=f.readlines()
    f.close()
    output=[[0,0,0,0,0,0] for j in range(ii+1)]
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
            if gen in seen:
                output[seen[gen]][0]=myth
    return output

# Import gene expression files and Pre-process them
with open('samples.txt', 'w') as f:
    import gzip
    def get_data(expname,diffexpname,diffmethyname,cnvname,vcfname,output):
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
            if gene in seen:
                output[seen[gene]][1]=exp
                
    # Import differential gene expression files and Pre-process them             
        f=open(diffexpname)
        line=f.readlines()
        f.close()    
        for l in line:
            gene,diffexp=l.split('\t')
            prev=gene
            index=gene.find('.')
            if index<1:
                index=len(gene)
            gene=gene[:index]
            diffexp=int(diffexp)
            if gene in seen:
                output[seen[gene]][2]=diffexp
        
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
            diffmethy=int(diffmethy)
            if gene in seen:
                output[seen[gene]][3]=diffmethy
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
            cnv=int(cnv)
            if gene in seen:
                output[seen[gene]][4]=cnv                        
    # Import VCF files and Pre-process them            
        f=open(vcfname)
        line=f.readlines()
        f.close()    
        for l in line:
            gene,score=l.split('\t')
            score=float(score)
            if gene in seen:
                print(gene + '\t' + output, file=f)
                print('\n', file=f)
                output[seen[gene]][5]=score

        return output
    
clin = [] # for clinical data (i.e. number of days to survive)
feat_vecs=[] # list of lists ([[patient1],[patient2],.....[patientN]]) -- [patientX] = [gene_expression_value, methylation_value, VCF_value, CNV_value]
can_types = ["BRCA","GBM","OV","LUAD","UCEC","KIRC","HNSC","LGG","THCA","LUSC","PRAD","SKCM","COAD","STAD","BLCA","LIHC","CESC","KIRP","SARC","LAML","PAAD","ESCA","PCPG","READ","TGCT","THYM","KICH","ACC","MESO","UVM","DLBC","UCS","CHOL"]
for i in range(len(can_types)):
    # file that contain patients ID with their coressponding 4 differnt files names (i.e. files names for gene_expression, methylation, VCF and CNV)
    f=open('intersect_five_'+can_types[i]+'.tsv')
    lines=f.readlines()
    f.close()
    lines=lines[1:]
    count=0
    for l in tqdm(lines):
        try:
            l=l.split('\t')
            # Check if all 4 files are exist for a patient (that's because for some patients, their survival time not reported)
            if os.path.isfile('cancer_types/TCGA-'+can_types[i]+'/clinical/'+l[2]) and os.path.isfile('cancer_types/TCGA-'+can_types[i]+'/myth/'+l[3]) and os.path.isfile('cancer_types/TCGA-'+can_types[i]+'/exp_norm/col/'+l[len(l)-1].rstrip()) and os.path.isfile('cancer_types/TCGA-'+can_types[i]+'/cnv/'+l[4]+'.txt') and os.path.isfile('cancer_types/TCGA-'+can_types[i]+'/vcf/output/'+'OutputAnnoFile_'+l[5]+'.hg38_multianno.txt.dat'):
                temp=l[2]
                f=open('cancer_types/TCGA-'+can_types[i]+'/clinical/'+temp)
                content=f.read().strip()
                f.close()
                clin.append(content)
                temp_myth=myth_data('cancer_types/TCGA-'+can_types[i]+'/myth/'+l[3])
                feat_vecs.append(exp_data('cancer_types/TCGA-'+can_types[i]+'/exp_norm/col/'+l[len(l)-1].rstrip(),'cancer_types/TCGA-'+can_types[i]'+/diffexp/'+l[0]+'.txt','cancer_types/TCGA-'+can_types[i]'+/diffmethy/'+l[1]+'.txt','cancer_types/TCGA-'+can_types[i]'+/cnv/'+l[4]+'.txt','cancer_types/TCGA-'+can_types[i]+'/vcf/output/'+'OutputAnnoFile_'+l[5]+'.hg38_multianno.txt.dat',temp_myth))
            else:
                print('Not exist!')

        except:
            count+=1
            
print(len(feat_vecs))
print(len(clin))

labels = [float(i) for i in clin]
label = []
for i in range(len(labels)):
    label.append(labels[i]/max(labels))
    
from torch_geometric.data import Data
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
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, perm, score = self.pool1(x, edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index))
        x = gmp(x, batch)
        b=data.y.shape[0]
        x=x.view(b,-1)
        ct = self.fc2(pt_tensor_cancer_type.to(device))
        cs = self.fc3(pt_tensor_cancer_subtype.to(device))
        al = self.fc4(pt_tensor_anatomical_location.to(device))
        cet = self.fc5(pt_tensor_cell_type.to(device))
        concat_tensors = torch.cat([ct, cs, al, cet], dim=0)
        x = self.fc1(x)
        concat_tensors = torch.unsqueeze(concat_tensors, 0)
        x = torch.matmul(x, concat_tensors)
        x = x.squeeze(1)
        x = torch.mean(x)
        x = torch.tensor([x])
        return x
    
# Train by batch
dataset=[]
edge=torch.tensor(ei,dtype=torch.long)
i=0
for e in range(len(feat_vecs)):
    x=torch.tensor(feat_vecs[e],dtype=torch.float)
    labell = label[e]
    dataset.append(Data(x=x,edge_index=edge,y=torch.tensor([labell])))
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CoxPH(MyNet().to(device))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
def train(epoch):
    model.train()
    loss_all = 0
    loss_=torch.nn.MSELoss()
    for data in train_loader:
        label=data.y
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output.type(torch.FloatTensor)
        loss = loss_(output, (data.y.view(output.shape[0],1)).type(torch.FloatTensor))
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)

def test(loader,dataset,thresh):
    with torch.no_grad():
        model.eval()
        correct = 0
        for data in (loader):
            data = data.to(device)
            output = model(data)
            b,_=output.shape
            data.y=data.y.reshape(output.shape[0],1)
            correct+=sum((output>=thresh) & (data.y>0.0))+sum((data.y<1.0) & (output<thresh))
        return float(correct) / float(len(dataset))

# Each time test on a specific cancer type
total_cancers = ["TCGA-BRCA","TCGA-GBM","TCGA-OV","TCGA-LUAD","TCGA-UCEC","TCGA-KIRC","TCGA-HNSC","TCGA-LGG","TCGA-THCA","TCGA-LUSC","TCGA-PRAD","TCGA-SKCM","TCGA-COAD","TCGA-STAD","TCGA-BLCA","TCGA-LIHC","TCGA-CESC","TCGA-KIRP","TCGA-SARC","TCGA-LAML","TCGA-PAAD","TCGA-ESCA","TCGA-PCPG","TCGA-READ","TCGA-TGCT","TCGA-THYM","TCGA-KICH","TCGA-ACC","TCGA-MESO","TCGA-UVM","TCGA-DLBC","TCGA-UCS","TCGA-CHOL"]
for i in range(len(total_cancers)):
    preds=[]
    trues=[]
    test_set = [d for t, d in zip(total_cancers, dataset) if t == total_cancers[i]]
    train_set = [d for t, d in zip(total_cancers, dataset) if t != total_cancers[i]]
    
    # Split 70% from all 32 cancers and test on 15% of a specific one
    train_size = len(train_set)
    train_indices = list(range(train_size))
    np.random.shuffle(train_indices)
    train_split_index = int(np.floor(0.7 * train_size))
    
    test_size = len(test_set)
    test_indices = list(range(test_size))
    np.random.shuffle(test_indices)
    test_split_index = int(np.floor(0.15 * test_size))
    
    train_idx = train_indices[train_split_index:] 
    test_idx = test_indices[:test_split_index]
    
    train_dataset = SubsetRandomSampler(train_idx)
    test_dataset = SubsetRandomSampler(test_idx)
    
    train_loader=DataLoader(train_dataset,batch_size=3)
    test_loader=DataLoader(test_dataset,batch_size=3)
    
    for epoch in range(1, 101):
        loss = train(epoch)
        train_acc = test(train_loader,train_dataset,0.5)
        test_acc = test(test_loader,test_dataset,0.5)
        print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
              format(epoch, loss, train_acc, test_acc))
    # Compute the evaluation measurements
    for t in test_loader:
        preds+=(model(t.to(device)).detach()).tolist()
        trues+=t.y.tolist()
    preds=[x for x in preds]
    c_index = EvalSurv(trues, preds).concordance_td()
    mse = mean_squared_error(trues, preds)
    rmse = math.sqrt(mse)
    
    torch.save(MyNet().state_dict(), 'model.h5')
