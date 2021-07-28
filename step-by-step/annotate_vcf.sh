#!/bin/bash 

declare -a CancerTypes=("TCGA-GBM" "TCGA-OV" "TCGA-LUAD" "TCGA-UCEC" "TCGA-KIRC" "TCGA-HNSC" "TCGA-LGG" "TCGA-THCA" "TCGA-LUSC" "TCGA-PRAD" "TCGA-SKCM" "TCGA-COAD" "TCGA-STAD" "TCGA-BLCA" "TCGA-LIHC" "TCGA-CESC" "TCGA-KIRP" "TCGA-SARC" "TCGA-LAML" "TCGA-PAAD" "TCGA-ESCA" "TCGA-PCPG" "TCGA-READ" "TCGA-TGCT" "TCGA-BRCA" "TCGA-THYM" "TCGA-KICH" "TCGA-ACC" "TCGA-MESO" "TCGA-UVM" "TCGA-DLBC" "TCGA-UCS" "TCGA-CHOL")
 
# Iterate over different cancer types
for val in ${CancerTypes[@]}
do

# the path to input/samples folder
path_to_vcf=/cancer_types/$val/vcf
mkdir /cancer_types/$val/vcf/output
# the path to Results folder
path_to_output=/cancer_types/$val/vcf/output

# the path to annovar folder
path_to_annovar='/annovar'

for entry in $path_to_vcf/*.vcf.gz
do

fbname=$(basename "$entry" .vcf.gz)

# start annotate the vcf file
perl "$path_to_annovar"/table_annovar.pl "$path_to_vcf/$fbname".vcf.gz humandb/ -buildver hg38 -out "$path_to_output/OutputAnnoFile_$fbname" -remove -protocol ensGene,avsnp147,dbnsfp30a -operation g,f,f -nastring . -vcfinput

done
done