# Deep Multi Omics CanCer Analysis (DeepMOCCA) - User Guide on Training the Model

DeepMOCCA is a tool for integrating and analyzing omics data with respect to background knowledge in the form of a graph for cancer survival prediction.

## Installation requirements

- Python 3.7 and install the dependencies (for Python 3.7) with:
 ```
	pip install -r requirements.txt
 ```
 - `torch` and `torch geometric cuda` need to have the same version, we suggest to follow the instructions [Here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
 
## Prepare the multi-omics data

* All multi-omics data for the 33 cancer types (i.e. gene expression, DNA methylation, copy number variation (CNV), single nucleotide variation (SNV) and clinical data) except for differentional gene expression and differentional DNA methylation have been downloded from [The Cancer Genome Atlas (TCGA)](http://cancergenome.nih.gov) via their [Data Transfer Tool Client](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool)

### Gene Expression

- Run `preprocess_gene_expression.py` to normlize gene expression data (i.e column, row and matrix -based normlization) with specifying the folder that includes the cancer types subfolders which have gene expression data files for each patient.

### Differential Gene Expression

- We used `TCGAanalyze_DEA` function in [TCGAbiolinks](/bioc/vignettes/TCGAbiolinks/inst/doc/analysis.html#TCGAanalyze_DEA__TCGAanalyze_LevelTab:_Differential_expression_analysis_(DEA)) to calculate the differential expression value of each gene for all patients.

###  DNA Methylation

- Run `preprocess_methylation.py` with specifying the folder that includes the cancer types subfolders which have dna methylation data files for each patient. It maps the transcripts to their corresponding genes, filter out `null` beta values and averaging the methylation values forthese transcripts and assigned the resulted value to their corresponding gene.

### Differential DNA Methylation

- Run `preprocess_diff_methylation.py` with specifying the folder that includes the cancer types subfolders which have dna methylation data files for each patient and the sample sheet specifying wheather a file derived from tumor and normal tissue. It use Wilcoxon rank-sum test adjusted by Benjamini-Hochberg method with p < 0.05, and then we assign the calculated p-value for each gene.

###  Copy Number Variation

- Run `preprocess_CNV.py` which process the `*.focal_score_by_genes.txt` which contain the CNV values for each gene per patient to generate separate file for each patient.

###  Single Nucleotide Variation

- Run `process_vcf.sh` on the annotated VCF files by the FATHMM tool which would filter and assign each gene a set of pathogenicity scores for its variants.

## Usage:

To run the training model `deepmocca_training.py`, the user need to provide:

- `rdf_string.ttl`: The RDF graph for PPI network.

- `ens_dic.pkl`: A dictionary that maps protiens to their coresponding genes by Ensembl database.

- `prot_names.txt`: A mapping file used to build a dictionary from ENSG -- ENST (for DNA Methylation data).

- `samples_'+can_types[i]*+'.tsv`: A file that contain patients ID with their coressponding 7 differnt files names (i.e. files names for gene expression, differential gene expression, dna methylation, differential dna methylation, VCF, CNV and the clinical data).

- Specifiy the cancer type and the anatomical location.

- The path to the folders where omics data located.
