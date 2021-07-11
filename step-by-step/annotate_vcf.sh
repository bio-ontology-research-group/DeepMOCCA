#!/bin/bash 

#SBATCH --mem 80Gb # memory pool for all cores
#SBATCH --time 24:00:00 # time, specify max time allocation
#SBATCH --gres=gpu
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=sara.althubaiti@kaust.edu.sa
#SBATCH --job-name=Annotategene
#SBATCH --output=HNSC.log

# the path to input/samples folder
path_to_vcf='/cancer_types/TCGA-HNSC/vcf'
mkdir /encrypted/e3008/Sara/cancer_types/TCGA-HNSC/vcf/output
# the path to Results folder
path_to_output='/cancer_types/TCGA-HNSC/vcf/output'

# the path to annovar folder
path_to_annovar='/encrypted/e3000/gatkwork/annovar'

for entry in $path_to_vcf/*.vcf.gz
do

fbname=$(basename "$entry" .vcf.gz)

# start annotate the vcf file
perl "$path_to_annovar"/table_annovar.pl "$path_to_vcf/$fbname".vcf.gz humandb/ -buildver hg38 -out "$path_to_output/OutputAnnoFile_$fbname" -remove -protocol ensGene,avsnp147,dbnsfp30a -operation g,f,f -nastring . -vcfinput

done
