
# the path to input/samples folder
path_to_vcf='/cancer_types/TCGA-HNSC/vcf/output'

for entry in $path_to_vcf/*.vcf
do

fbname=$(basename "$entry" .vcf)

# Select the two columns that include ENSG and pathogenicity score

awk -F'\t' '{ print $56,$24 }' "$path_to_vcf/$fbname".vcf > "$path_to_vcf/$fbname".ta
python process.py "$path_to_vcf/$fbname".ta

sed 's/ /\t/g' "$path_to_vcf/$fbname".ta.out > "$path_to_vcf/$fbname".re

awk -F'\t' '{ print $1,$3 }' "$path_to_vcf/$fbname".re > "$path_to_vcf/$fbname".dat
