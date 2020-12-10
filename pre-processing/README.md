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
