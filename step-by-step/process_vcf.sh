#!/bin/bash 

declare -a CancerTypes=("TCGA-GBM" "TCGA-OV" "TCGA-LUAD" "TCGA-UCEC" "TCGA-KIRC" "TCGA-HNSC" "TCGA-LGG" "TCGA-THCA" "TCGA-LUSC" "TCGA-PRAD" "TCGA-SKCM" "TCGA-COAD" "TCGA-STAD" "TCGA-BLCA" "TCGA-LIHC" "TCGA-CESC" "TCGA-KIRP" "TCGA-SARC" "TCGA-LAML" "TCGA-PAAD" "TCGA-ESCA" "TCGA-PCPG" "TCGA-READ" "TCGA-TGCT" "TCGA-BRCA" "TCGA-THYM" "TCGA-KICH" "TCGA-ACC" "TCGA-MESO" "TCGA-UVM" "TCGA-DLBC" "TCGA-UCS" "TCGA-CHOL")

# Iterate over different cancer types
for val in ${CancerTypes[@]}
do

# the path to input/samples folder
path_to_vcf=/cancer_types/$val/vcf/output

for entry in $path_to_vcf/*.vcf
do

fbname=$(basename "$entry" .vcf)

# Select the two columns that include ENSG and pathogenicity score

awk -F'\t' '{ print $56,$24 }' "$path_to_vcf/$fbname".vcf > "$path_to_vcf/$fbname".ta
python process.py "$path_to_vcf/$fbname".ta

sed 's/ /\t/g' "$path_to_vcf/$fbname".ta.out > "$path_to_vcf/$fbname".re

awk -F'\t' '{ print $1,$3 }' "$path_to_vcf/$fbname".re > "$path_to_vcf/$fbname".dat

done
done