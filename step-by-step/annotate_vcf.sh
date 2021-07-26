#!/bin/bash 

# the path to input/samples folder
path_to_vcf='/cancer_types/TCGA-HNSC/vcf'
mkdir /cancer_types/TCGA-HNSC/vcf/output
# the path to Results folder
path_to_output='/cancer_types/TCGA-HNSC/vcf/output'

# the path to annovar folder
path_to_annovar='/annovar'

for entry in $path_to_vcf/*.vcf.gz
do

fbname=$(basename "$entry" .vcf.gz)

# start annotate the vcf file
perl "$path_to_annovar"/table_annovar.pl "$path_to_vcf/$fbname".vcf.gz humandb/ -buildver hg38 -out "$path_to_output/OutputAnnoFile_$fbname" -remove -protocol ensGene,avsnp147,dbnsfp30a -operation g,f,f -nastring . -vcfinput

done
