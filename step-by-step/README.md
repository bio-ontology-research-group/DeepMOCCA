# Instructions on training DeepMOCCA using TCGA data

## Installation requirements

- Python 3.7 and install the dependencies (for Python 3.7) with:
 ```
	pip install -r requirements.txt
 ```
 - `torch` and `torch geometric cuda` need to have the same version, we suggest to follow the instructions [Here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
 
## Prepare the multi-omics data

* Download the folder `data_folder/` which includes all intermediate needed files for mappings.

* All multi-omics data for the 33 cancer types (i.e. gene expression, DNA methylation, copy number variation (CNV), single nucleotide variation (SNV) and clinical data) except for differentional gene expression and differentional DNA methylation have been downloded from [The Cancer Genome Atlas (TCGA)](http://cancergenome.nih.gov) via their [Data Transfer Tool Client](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool). Approved access to individual-level data is required to train the full DeepMOCCA model.

* Create a folder `cancer_types/` and sub-folders for each cancer type `TCGA-CANCERTYPE/` that includes all related sub-sub-folders omics data:

```
cancer_types  
│
└───TCGA-ACC
│   │
│   └───exp
│       │   686fb7ee-bdd2-4c41-ab25-c0d8b0ccc049.htseq.gz
│       │   dcf26935-b9a9-40fc-a6f0-7e0ef4b7228a.htseq.gz
│       │   ...
│   │
│   └───exp_diff
│       │   da33344b-b0d8-4504-bf2c-95aed9bf607a.txt.gz
│       │   df34297d-30aa-484d-8d21-4c1223716469.txt.gz
│       │   ...
│   │
│   └─── methyl
│       │   jhu-usc.edu_ACC.HumanMethylation450.1.lvl-3.TCGA-PK-A5HB-01A-11D-A29J-05.gdc_hg38.txt
│       │   jhu-usc.edu_ACC.HumanMethylation450.1.lvl-3.TCGA-OR-A5LM-01A-11D-A29J-05.gdc_hg38.txt
│       │   ...
│   │
│   └─── methyl_diff
│       │   jhu-usc.edu_ACC.HumanMethylation450.1.lvl-3.TCGA-OR-A5J3-01A-11D-A29J-05.gdc_hg38.txt
│       │   jhu-usc.edu_ACC.HumanMethylation450.1.lvl-3.TCGA-OR-A5J7-01A-11D-A29J-05.gdc_hg38.txt
│       │   ...
│   │
│   └─── vcf
│       │   88d42bb6-b07d-4f41-b512-cdf2e7650776.vep.vcf.gz
│       │   63dece65-5f99-48db-bbf8-84eb652f3167.vep.vcf.gz
│       │   ...
│   │
│   └─── cnv
│       │   ACC.focal_score_by_genes.txt
│   │
│   └─── clinical
│       │   nationwidechildrens.org_clinical.TCGA-OR-A5KP.xml
│       │   nationwidechildrens.org_clinical.TCGA-OR-A5JT.xml
│       │   ...
│
└───TCGA-...
```

### Gene Expression

- Run `preprocess_gene_expression.ipynb` to normlize gene expression data (i.e., column, row and matrix-based normalization).

### Differential Gene Expression

- We used the `TCGAanalyze_DEA` function in [TCGAbiolinks](/bioc/vignettes/TCGAbiolinks/inst/doc/analysis.html#TCGAanalyze_DEA__TCGAanalyze_LevelTab:_Differential_expression_analysis_(DEA)) to calculate the differential expression value of each gene for all patients as shown in `process_diffexp.r`

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

