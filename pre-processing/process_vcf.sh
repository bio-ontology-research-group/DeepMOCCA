#!/bin/bash
#SBATCH --mem 50Gb # memory pool for all cores
#SBATCH --time 00:20:00 # time, specify max time allocation
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=sara.althubaiti@kaust.edu.sa ##specify your e-mail address 
#SBATCH --job-name=process
#SBATCH --array=0-560 ## 1 is the last index in the input file (e.g. OA.files.txt)
echo "Job ID=$SLURM_JOB_ID,  Running task:$SLURM_ARRAY_TASK_ID"

values=$(grep "^${SLURM_ARRAY_TASK_ID}:" /encrypted/e3008/Sara/cancer_types/TCGA-LUSC/vcf/output/list.txt)
echo $values
filename=$(echo $values | cut -f 2 -d:)
echo $filename

awk -F'\t' '{ print $56,$24 }' /encrypted/e3008/Sara/cancer_types/TCGA-LUSC/vcf/output/$filename > /encrypted/e3008/Sara/cancer_types/TCGA-LUSC/vcf/output/"$filename".ta
python3 process.py /encrypted/e3008/Sara/cancer_types/TCGA-LUSC/vcf/output/"$filename".ta

sed 's/ /\t/g' /encrypted/e3008/Sara/cancer_types/TCGA-LUSC/vcf/output/"$filename".ta.out > /encrypted/e3008/Sara/cancer_types/TCGA-LUSC/vcf/output/"$filename".re

awk -F'\t' '{ print $1,$3 }' /encrypted/e3008/Sara/cancer_types/TCGA-LUSC/vcf/output/"$filename".re > /encrypted/e3008/Sara/cancer_types/TCGA-LUSC/vcf/output/"$filename".dat
