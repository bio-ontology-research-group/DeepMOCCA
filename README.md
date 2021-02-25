# DeepMOCCA - Deep Multi Omics CanCer Analysis

DeepMOCCA is a method to predict survival time in individual cancer samples. DeepMOCCA also learns representations of the multi-scale activities and interactions within a cell from multi-omics data associated with these samples.

Our tool takes as input data derived from individual sample, in particular the absolute gene expression, differential expression, absolute methylation, differential methylation, type of the copy number variants, and pathogenicity scores for the set of germline and somatic variants; it also has as input the cancer type and anatomical location of the tumor. We use this information to determine the cell type of origin.

## Datasets

* All multi-omics data for the 33 cancer types (i.e. gene expression, DNA methylation, copy number variation (CNV), single nucleotide variation (SNV) and clinical data) have been downloaded from the [The Cancer Genome Atlas (TCGA)](http://cancergenome.nih.gov) via their [Data Transfer Tool Client](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool). Approval is required to access individual level data.

* The protein-protein interactions network is from [STRING](https://string-db.org/cgi/download.pl?sessionId=VKCYtvc7YJch&species_text=Homo+sapiens).

## Dependencies

* To install python dependencies run: `pip install -r requirements.txt`

* Note: the `torch` and `torch geometric cuda` need to have the same version, we suggest to follow the instructions [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

## Installation

`pip install deepmocca` should work in most cases.

## Running

* Download all the files in [data.tar.gz](https://bio2vec.cbrc.kaust.edu.sa/data/deepmocca/data.tar.gz) and place them into data folder
* `deepmocca -dr <path_to_data_folder> -if <input_fasta_filename> -ct <cancer_type> -ap <anatomical_location>`
* Run `deepmocca --help` to display a description for the parameters:
```
Usage: deepmocca [OPTIONS]

Options:
  -dr, --data-root TEXT           Data root folder  [required]
  -if, --in-file TEXT             Input file  [required]
  -mf, --model-file TEXT          Pytorch model file
  -ct, --cancer-type-flag TEXT    Cancer type flag  [required]
  -ap, --anatomical-part-flag TEXT
                                  Anatomical part flag  [required]
  -of, --out-file TEXT            Output result file
  --help                          Show this message and exit.
  ```
  
#### Cancer types and Anatomical locations parameters
```
-ct, --cancer-type-flag <number_correspoing_to_cancer_type>
 
1	Breast Invasive Carcinoma (TCGA-BRCA)
2	Adrenocortical Carcinoma (TCGA-ACC)
3	Bladder Urothelial Carcinoma (TCGA-BLCA)
4	Cervical Squamous Cell Carcinoma and Endocervical Adenocarcinoma (TCGA-CESC)
5	Cholangiocarcinoma (TCGA-CHOL)
6	Colon Adenocarcinoma (TCGA-COAD)
7	Lymphoid Neoplasm Diffuse Large B-cell Lymphoma (TCGA-DLBC)
8	Esophageal Carcinoma (TCGA-ESCA)
9	Glioblastoma Multiforme (TCGA-GBM)
10	Head and Neck Squamous Cell Carcinoma (TCGA-HNSC)
11	Kidney Chromophobe (TCGA-KICH)
12	Kidney Renal Clear Cell Carcinoma (TCGA-KIRC)
13	Kidney Renal Papillary Cell Carcinoma (TCGA-KIRP)
14	Acute Myeloid Leukemia (TCGA-LAML)
15	Brain Lower Grade Glioma (TCGA-LGG)
16	Liver Hepatocellular Carcinoma (TCGA-LIHC)
17	Lung Adenocarcinoma (TCGA-LUAD)
18	Lung Squamous Cell Carcinoma (TCGA-LUSC)
19	Mesothelioma (TCGA-MESO)
20	Ovarian Serous Cystadenocarcinoma (TCGA-OV)
21	Pancreatic Adenocarcinoma (TCGA-PAAD)
22	Pheochromocytoma and Paraganglioma (TCGA-PCPG)
23	Prostate Adenocarcinoma (TCGA-PRAD)
24	Rectum Adenocarcinoma (TCGA-READ)
25	Sarcoma (TCGA-SARC)
26	Skin Cutaneous Melanoma (TCGA-SKCM)
27	Stomach Adenocarcinoma (TCGA-STAD)
28	Testicular Germ Cell Tumors (TCGA-TGCT)
29	Thyroid Carcinoma (TCGA-THCA)
30	Thymoma (TCGA-THYM)
31	Uterine Corpus Endometrial Carcinoma (TCGA-UCEC)
32	Uterine Carcinosarcoma (TCGA-UCS)
33	Uveal Melanoma (TCGA-UVM)
 
-ap, --anatomical-part-flag <number_correspoing_to_anatomical_part> 
 
1	Breast
2	Adrenal gland
3	Bladder
4	Cervix uteri
5	Gallbladder
6	Liver and intrahepatic bile ducts
7	Colon
8	Rectosigmoid junction
9	Bones, joints and articular cartilage of other and unspecified sites
10	Brain
11	Connective, subcutaneous and other soft tissues
12	Heart, mediastinum, and pleura
13	Hematopoietic and reticuloendothelial systems
14	Lymph nodes
15	Other and unspecified major salivary glands
16	Retroperitoneum and peritoneum
17	Small intestine
18	Stomach
19	Testis
20	Thyroid gland
21	Esophagus
22	Base of tongue
23	Floor of mouth
24	Gum
25	Hypopharynx
26	Larynx
27	Lip
28	Oropharynx
29	Other and ill-defined sites in lip, oral cavity and pharynx
30	Other and unspecified parts of mouth
31	Other and unspecified parts of tongue
32	Palate
33	Tonsil
34	Kidney
35	Bronchus and lung
36	Ovary
37	Pancreas
38	Other and ill-defined sites
39	Other endocrine glands and related structures
40	Spinal cord, cranial nerves, and other parts of central nervous system
41	Prostate gland
42	Rectum
43	Bones, joints and articular cartilage of limbs
44	Corpus uteri
45	Meninges
46	Other and unspecified male genital organs
47	Peripheral nerves and autonomic nervous system
48	Uterus, NOS
49	Skin
50	Thymus
51	Eye and adnexa
```
 
#### Output
The model will output:
- A tab separated file has the same name as input file with **_results** extension which contains:
    * Sample_ID
    * Predicted survival time for a sample, for the selected cancer type and anatomical part
    * A vector representation for the sample based on the internal representation of the model
 
## Scripts

* `preprocess_gene_expression.py` - This script is used to preprocess and normalize gene expression data.
* `preprocess_methylation.py` - This script is used to preprocess the DNA methylation data.
* `preprocess_CNV.py` - This script is used to preprocess the copy number variation (CNV) data.
* `process_vcf.sh` and `process_vcf.py` - These scripts are used to preprocess the single-nucleotide variation (SNV) data.
* `deepmocca_training.py` - This script is used to train and save the trained model.

## Results

We have some [results](https://github.com/bio-ontology-research-group/DeepMOCCA/tree/master/results) pre-generated:

- `patients_representations.txt` - The representation of features for each patient generated after the 2nd `Conv2` layer.
- `Top_10_ranked_genes_all_samples.tar.gz` - The top 10 ranked genes for each patient genrated from the attention mechanism which specifiy wheather a gene is:
  * `Driver in the same cancer` -> `0` if yes and `1` if no.
  * `Driver in other cancer` -> `0` if yes and `1` if no.
  * `Prognostic in the same cancer` -> `0` if yes and `1` if no.
  * `Prognostic in other cancer` -> `0` if yes and `1` if no.
      
## Final notes

For any comments or help, please use the issue tracker or send an email to sara.althubaiti@kaust.edu.sa.
