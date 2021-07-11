# Deep Multi Omics CanCer Analysis (DeepMOCCA) - User Guide on Training the Model

## Installation requirements

- Python 3.7 and install the dependencies (for Python 3.7) with:
 ```
	pip install -r requirements.txt
 ```
 - `torch` and `torch geometric cuda` need to have the same version, we suggest to follow the instructions [Here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
 
## Prepare the multi-omics data

* Download the folder `data_folder/` which includes all intermediate needed files for mappings.

* All multi-omics data for the 33 cancer types (i.e. gene expression, DNA methylation, copy number variation (CNV), single nucleotide variation (SNV) and clinical data) except for differentional gene expression and differentional DNA methylation have been downloded from [The Cancer Genome Atlas (TCGA)](http://cancergenome.nih.gov) via their [Data Transfer Tool Client](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool)

* Create a folder `cancer_types/` and sub-folders for each cancer type `TCGA-CANCERTYPE/` that includes all related sub-sub-folders omics data.

### Gene Expression

- Run `preprocess_gene_expression.ipynb` to normlize gene expression data (i.e column, row and matrix -based normlization).

### Differential Gene Expression

- We used `TCGAanalyze_DEA` function in [TCGAbiolinks](/bioc/vignettes/TCGAbiolinks/inst/doc/analysis.html#TCGAanalyze_DEA__TCGAanalyze_LevelTab:_Differential_expression_analysis_(DEA)) to calculate the differential expression value of each gene for all patients as shown in `process_diffexp.r`

###  DNA Methylation

- Run `preprocess_methylation.ipynb` for filter out `null` beta values and averaging the methylation values for these transcripts and assigned the resulted value to their corresponding gene.

### Differential DNA Methylation

- Run `process_diffmethyl.ipynb` that uses Wilcoxon rank-sum test adjusted by Benjamini-Hochberg method and then we assign the calculated p-value for each gene.

###  Copy Number Variation

- Run `Preprocess_cnv.ipynb` which process the `*.focal_score_by_genes.txt` that contain the CNV values for each gene per patient to generate separate file for each patient.

###  Single Nucleotide Variation

- Run `./process_vcf.sh` on the annotated VCF files (after run Annovar with `./annotate_vcf.sh` script) by the FATHMM tool which would filter and assign each gene a set of pathogenicity scores for its variants.

## Usage:

To run the training model `deepmocca_training.ipynb`, the user need to provide (as already provided in `data_folder/`):

- `seen.pkl`: The protein nodes.

- `ei.pkl`: The PPI edges.

- `ens_dic.pkl`: A dictionary that maps protiens to their coresponding genes by Ensembl database.

- `prot_names1.txt`: A mapping file used to build a dictionary from ENSG -- ENST (for DNA Methylation data).

