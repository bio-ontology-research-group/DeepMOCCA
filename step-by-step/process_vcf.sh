#!/bin/bash
#SBATCH --mem 30Gb # memory pool for all cores
#SBATCH --time 00:20:00 # time, specify max time allocation
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=sara.althubaiti@kaust.edu.sa ##specify your e-mail address 
#SBATCH --job-name=process
#SBATCH --array=0-560 ## 560 is the last index in the input file (e.g. list.txt)

# This file should be used for every cancer type (for each folder -- 33 times) -- here for example for TCGA-HNSC/

echo "Job ID=$SLURM_JOB_ID,  Running task:$SLURM_ARRAY_TASK_ID"

# Create a list.txt that include the names for the annotated vcf files as:
# 0:file_name0.vcf
# 1:file_name1.vcf
# ......
# 560:file_name560.vcf

values=$(grep "^${SLURM_ARRAY_TASK_ID}:" /cancer_types/TCGA-HNSC/vcf/output/list.txt)
echo $values
filename=$(echo $values | cut -f 2 -d:)
echo $filename

# Select the two columns that include ENSG and pathogenicity score

awk -F'\t' '{ print $56,$24 }' /cancer_types/TCGA-HNSC/vcf/output/$filename > /cancer_types/TCGA-HNSC/vcf/output/"$filename".ta
python process.py /cancer_types/TCGA-HNSC/vcf/output/"$filename".ta

sed 's/ /\t/g' /cancer_types/TCGA-HNSC/vcf/output/"$filename".ta.out > /cancer_types/TCGA-HNSC/vcf/output/"$filename".re

awk -F'\t' '{ print $1,$3 }' /cancer_types/TCGA-HNSC/vcf/output/"$filename".re > /cancer_types/TCGA-HNSC/vcf/output/"$filename".dat
