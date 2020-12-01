import pandas as pd
import csv
import numpy as np
import os
import glob

names = ["TCGA-GBM","TCGA-OV","TCGA-LUAD","TCGA-UCEC","TCGA-KIRC","TCGA-HNSC","TCGA-LGG","TCGA-THCA","TCGA-LUSC","TCGA-PRAD","TCGA-SKCM","TCGA-COAD","TCGA-STAD","TCGA-BLCA","TCGA-LIHC","TCGA-CESC","TCGA-KIRP","TCGA-SARC","TCGA-LAML","TCGA-PAAD","TCGA-ESCA","TCGA-PCPG","TCGA-READ","TCGA-TGCT","TCGA-THYM","TCGA-KICH","TCGA-ACC","TCGA-MESO","TCGA-UVM","TCGA-DLBC","TCGA-UCS","TCGA-CHOL"]
dirFile = 'cancer_types/'

for i in range(len(names)):
    print(names[i])
    search_path = dirFile + names[i] + "/cnv/*.focal_score_by_genes.txt"
    allFile = glob.glob(search_path)
    for file_ in allFile:
        print(file_)
        df1 = pd.read_csv(file_, header=0, sep='\t')
        del df1['Gene ID']
        del df1['Cytoband']
        for j in range(df1.shape[1]-1):
            df = df1[[df1.columns[0],df1.columns[j+1]]]
            df.to_csv(dirFile + names[i] +'/cnv/'+df1.columns[j+1]+'.txt', sep='\t', index=False, header=False)
