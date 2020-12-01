import pandas as pd
import csv
import numpy as np
import os
import glob

names = ["TCGA-BRCA","TCGA-GBM","TCGA-OV","TCGA-LUAD","TCGA-UCEC","TCGA-KIRC","TCGA-HNSC","TCGA-LGG","TCGA-THCA","TCGA-LUSC","TCGA-PRAD","TCGA-SKCM","TCGA-COAD","TCGA-STAD","TCGA-BLCA","TCGA-LIHC","TCGA-CESC","TCGA-KIRP","TCGA-SARC","TCGA-LAML","TCGA-PAAD","TCGA-ESCA","TCGA-PCPG","TCGA-READ","TCGA-TGCT","TCGA-THYM","TCGA-KICH","TCGA-ACC","TCGA-MESO","TCGA-UVM","TCGA-DLBC","TCGA-UCS","TCGA-CHOL"]
dirFile = 'cancer_types/'

for file_ in allFile:
    df1 = pd.read_csv(file_, header=0, sep='\t')
    print(file_)
    dff = df1[['Transcript_ID','Beta_value']]
    dff = dff[dff['Beta_value'].notna()]
    dff = dff[dff['Transcript_ID'] != '.']
    dff.to_csv(file_, sep='\t', index=False, header=False)
