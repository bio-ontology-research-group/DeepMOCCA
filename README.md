# Multi-omics based survival analysis for cancer

## Datasets

* All multi-omics data for the 33 cancer types (i.e. gene expression, DNA methylation, copy number variation (CNV), single nucleotide variation (SNV) and clinical data) have been downloded from [The Cancer Genome Atlas (TCGA)](http://cancergenome.nih.gov) via their [Data Transfer Tool Client](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool)

* The protein-protein interactions network for human from [STRING](https://string-db.org/cgi/download.pl?sessionId=VKCYtvc7YJch&species_text=Homo+sapiens)

* All processed multi-omics data for the 33 cancer types can be found on [Zenodo](https://zenodo.org/record/3981464#.X1KBES2B3EY)

## Dependencies

To install python dependencies run: `pip install -r requirements.txt`

## Notebooks

The provided Jupyter notebooks includes:

* [pre-processing](https://github.com/bio-ontology-research-group/Cancer_SurvivalPrediction/tree/master/pre-processing) folder -- includes all pre-processing notebooks for gene expression, DNA methylation, copy number variation (CNV), single nucleotide variation (SNV) and clinical data

* single_model.ipynb -- single cancer type-based survival prediction

* multi_cancer_model.ipynb -- combined model for all 33 cancer types and predict on a specific cancer type iteratively

* atten_rank_genes.ipynb -- attention-based rank of driver genes for each of the 33 cancer types separately


## Final notes

For any comments or help needed with how to run the scripts, please send an email to: sara.althubaiti@kaust.edu.sa
