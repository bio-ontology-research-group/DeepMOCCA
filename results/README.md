## Results

- `patients_representations.txt` - The representation of features for each patient generated after the 2nd `Conv2` layer.

- `Top_10_ranked_genes_over_all_samples.tgz` - The top 10 ranked genes for each patient genrated from attention-based mechanism which specifiy wheather a gene is:
  * `Driver in the same cancer` -> `0` if yes and `1` if no.
  * `Driver in other cancer` -> `0` if yes and `1` if no.
  * `Prognostic in the same cancer` -> `0` if yes and `1` if no.
  * `Prognostic in other cancer` -> `0` if yes and `1` if no.

- `Top_1_rank.tsv` - The top 1 ranked genes over all the patients across all the cancer types, last column specify the number of patients having this specific gene within the top 1 and for that specific cancer type.

- `Top_10_rank.tsv` - The top 10 ranked genes over all the patients across all the cancer types combined from `ranked_genes_top10.tar.gz` and last column specifying the number of patients having this specific gene within the top 10 and for that specific cancer type.

- `top-novel.tsv` - The top 1 ranked genes over all the patients across all the cancer types which are not known driver genes or prognostic markers.

- `significant_genes_per_cohort.tgz` - Genes that are ranked significantly higher by our model's graph attention mechanism across all samples within each cohort/cancer type than expected under a uniform random distribution; reported their p-value, effect size, Holm–Bonferroni correction and Benjamini-Hochberg correction sorted by the effect size.

- `significant_genes_per_cohort_filtered.tgz` - Same list of genes in `significant_genes_per_cohort.tgz` filtered by p<0.2 (Benjamini-Hochberg correction) and sorted by the effect size.

## Comparion to [Cheerla et al.](https://academic.oup.com/bioinformatics/article/35/14/i446/5529139)

- The test set (patients IDs) for each cancer type are presented in `test_set.txt`

| Cancer type | DeepMOCCA (individual model) |  DeepMOCCA (joint model)  | Pancancer model (single cancer) | Pancancer model (pancancer) |
|:-----------:|:----------------------------:|:-------------------------:|:------------------------------------------------------:|:--------------------------------------------------:|
|     BRCA    |       0.86 [0.85-0.87]       | 0.87 [0.86-0.88] |                          0.65                          |                        0.82                        |
|     KIRC    |       0.86 [0.85-0.87]       | 0.87 [0.85-0.88] |                          0.80                          |                        0.79                        |
|     LIHC    |   0.78 [0.77-0.79]  |      0.75 [0.74-0.76]     |                      0.81                     |                        0.78                        |
|     BLCA    |       0.63 [0.62-0.64]       | 0.77 [0.75-0.79] |                          0.65                          |                        0.74                        |
|     CESC    |       0.61 [0.60-0.62]       | 0.84 [0.83-0.85] |                          0.56                          |                        0.79                        |
|     COAD    |       0.69 [0.68-0.70]       | 0.87 [0.85-0.88] |                          0.60                          |                        0.75                        |
|     READ    |       0.64 [0.63-0.65]       | 0.74 [0.73-0.75] |                          0.59                          |                    0.76                   |
|     HNSC    |       0.67 [0.66-0.68]       | 0.88 [0.86-0.89] |                          0.66                          |                        0.71                        |
|     KICH    |       0.75 [0.74-0.76]       |      0.77 [0.75-0.78]     |                          0.68                          |                    0.94                   |
|     KIRP    |       0.56 [0.55-0.57]       | 0.81 [0.80-0.82] |                          0.52                          |                        0.80                        |
|     LAML    |       0.68 [0.67-0.69]       | 0.72 [0.70-0.73] |                          0.67                          |                        0.69                        |
|     LGG     |       0.77 [0.76-0.78]       |      0.76 [0.75-0.77]     |                          0.74                          |                    0.88                   |
|     LUAD    |       0.86 [0.85-0.87]       | 0.88 [0.86-0.89] |                          0.70                          |                        0.74                        |
|     LUSC    |       0.68 [0.67-0.69]       | 0.87 [0.85-0.88] |                          0.64                          |                        0.75                        |
|      OV     |       0.59 [0.58-0.60]       | 0.86 [0.85-0.87] |                          0.57                          |                        0.73                        |
|     PAAD    |       0.64 [0.63-0.65]       |      0.73 [0.72-0.74]     |                          0.58                          |                    0.75                   |
|     PRAD    |       0.80 [0.79-0.81]       |      0.73 [0.72-0.74]     |                          0.78                          |                    0.84                   |
|     SKCM    |       0.62 [0.61-0.63]       |      0.74 [0.73-0.75]     |                          0.55                          |                    0.75                   |
|     STAD    |       0.70 [0.69-0.71]       | 0.85 [0.84-0.86] |                          0.61                          |                        0.80                        |
|     THCA    |       0.58 [0.57-0.59]       |      0.85 [0.84-0.87]     |                          0.55                          |                    0.92                   |
|     UCEC    |       0.71 [0.70-0.72]       |      0.79 [0.77-0.81]     |                          0.68                          |                    0.86                   |
