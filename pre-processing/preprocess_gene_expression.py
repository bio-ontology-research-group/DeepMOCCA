import pandas as pd
import csv
import numpy as np
import os
import glob

names = ["TCGA-BRCA","TCGA-GBM","TCGA-OV","TCGA-LUAD","TCGA-UCEC","TCGA-KIRC","TCGA-HNSC","TCGA-LGG","TCGA-THCA","TCGA-LUSC","TCGA-PRAD","TCGA-SKCM","TCGA-COAD","TCGA-STAD","TCGA-BLCA","TCGA-LIHC","TCGA-CESC","TCGA-KIRP","TCGA-SARC","TCGA-LAML","TCGA-PAAD","TCGA-ESCA","TCGA-PCPG","TCGA-READ","TCGA-TGCT","TCGA-THYM","TCGA-KICH","TCGA-ACC","TCGA-MESO","TCGA-UVM","TCGA-DLBC","TCGA-UCS","TCGA-CHOL"]
dirFile = 'cancer_types/'

for i in range(len(names)):
    print(names[i])
    new_df = pd.DataFrame()
    # my_df -- is a dataframe contains two columns of (patiant_gene_expression_files_names, cancer_type)
    temp = my_df.loc[my_df['Cancer Type'] == names[i]]
    for index, row in temp.iterrows():
        df = pd.read_csv(dirFile+names[i]+"/exp_count/"+row['File Name'], compression='gzip', error_bad_lines=False, index_col=None, header=None, sep='\t')
        df.columns = ['gene', str(row['File Name'])]
        new_df = pd.concat([new_df, df[str(row['File Name'])]], axis=1)
    # row normlization
    new_df = new_df.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=1)
    for column in new_df:
        print(column)
        df2 = pd.concat([df[['gene']], new_df[[column]]], axis=1)
        df2 = df2.head(-5)
        df2.to_csv(dirFile+names[i]+"/exp_data/"+"row/"+column, sep='\t', index=False, header=False, compression='gzip')
        
        
        
for i in range(len(names)):
    print(names[i])
    new_df = pd.DataFrame()
    # my_df -- is a dataframe contains two columns of (patiant_gene_expression_files_names, cancer_type)
    temp = my_df.loc[my_df['Cancer Type'] == names[i]]
    for index, row in temp.iterrows():
        df = pd.read_csv(dirFile+names[i]+"/exp_count/"+row['File Name'], compression='gzip', error_bad_lines=False, index_col=None, header=None, sep='\t')
        df.columns = ['gene', str(row['File Name'])]
        new_df = pd.concat([new_df, df[str(row['File Name'])]], axis=1)
    # matrix normlization
    column_maxes = new_df.max()
    df_max = column_maxes.max()
    column_mins = new_df.min()
    df_min = column_mins.min()
    normalized_df = (new_df - df_min) / (df_max - df_min)
    for column in normalized_df:
        print(column)
        df2 = pd.concat([df[['gene']], normalized_df[[column]]], axis=1)
        df2 = df2.head(-5)
        df2.to_csv(dirFile+names[i]+"/exp_data/"+"matrix/"+column, sep='\t', index=False, header=False, compression='gzip')
        


for i in range(len(names)):
    print(names[i])
    new_df = pd.DataFrame()
    # my_df -- is a dataframe contains two columns of (patiant_gene_expression_files_names, cancer_type)
    temp = my_df.loc[my_df['Cancer Type'] == names[i]]
    for index, row in temp.iterrows():
        df = pd.read_csv(dirFile+names[i]+"/exp_count/"+row['File Name'], compression='gzip', error_bad_lines=False, index_col=None, header=None, sep='\t')
        df.columns = ['gene', str(row['File Name'])]
        new_df = pd.concat([new_df, df[str(row['File Name'])]], axis=1)
    # column normlization
    for column in new_df:
        print(column)
        new_df[column] = (new_df[column] - new_df[column].min()) / (new_df[column].max() - new_df[column].min())
        df2 = pd.concat([df[['gene']], new_df[[column]]], axis=1)
        df2 = df2.head(-5)
        df2.to_csv(dirFile+names[i]+"/exp_data/"+"col/"+column, sep='\t', index=False, header=False, compression='gzip')
