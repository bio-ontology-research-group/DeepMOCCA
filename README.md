# DeepMOCCA - Deep Multi Omics CanCer Analysis

## Datasets

* All multi-omics data for the 33 cancer types (i.e. gene expression, DNA methylation, copy number variation (CNV), single nucleotide variation (SNV) and clinical data) have been downloded from [The Cancer Genome Atlas (TCGA)](http://cancergenome.nih.gov) via their [Data Transfer Tool Client](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool)

* The protein-protein interactions network for human from [STRING](https://string-db.org/cgi/download.pl?sessionId=VKCYtvc7YJch&species_text=Homo+sapiens)

## Dependencies

To install python dependencies run: `pip install -r requirements.txt`

## Installation

`pip install deepmocca`

## Running

* Download all the files in `data.tar.gz` and place them into data folder
* `deepmocca -dr <path_to_data_folder> -if <input_fasta_filename> -ct <cancer_type> -ap <anatomical_location>`

### Output
The model will output:
- A file contains the time to live for all samples for the selected cancer type.
- A file contains a vector representation for each patient retrieved through the model learning process.

## Final notes

For any comments or help needed with how to run the scripts, please send an email to: sara.althubaiti@kaust.edu.sa
