# DeepMOCCA - Deep Multi Omics CanCer Analysis

## Datasets

* All multi-omics data for the 33 cancer types (i.e. gene expression, DNA methylation, copy number variation (CNV), single nucleotide variation (SNV) and clinical data) have been downloded from [The Cancer Genome Atlas (TCGA)](http://cancergenome.nih.gov) via their [Data Transfer Tool Client](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool)

* The protein-protein interactions network for human from [STRING](https://string-db.org/cgi/download.pl?sessionId=VKCYtvc7YJch&species_text=Homo+sapiens)

## Dependencies

To install python dependencies run: `pip install -r requirements.txt`

## Prediction of cancer patients survival time workflow

- Download all the provided files from this repository
```
git clone https://github.com/bio-ontology-research-group/Cancer_SurvivalPrediction.git
```
## Graph neural networks predict survival time from personal omics data across subtypes
![alt text](img/combined_model_workflow.png)

We developed a subtypes-based model that integrates information from encoded subgroups of cancer subtypes, anatomical parts, and the cell types of origin besides the cancer types and given different multi-omics data for cancer patients fall within these subgroups.

### Output
The model will output:
- A file contains average attention-based ranks for the genes over all samples for the selected cancer type.
- A file contains the time to live for all samples for the selected cancer type.
- A file contains a vector representation for each patient retrieved through the model learning process.

## Final notes

For any comments or help needed with how to run the scripts, please send an email to: sara.althubaiti@kaust.edu.sa
