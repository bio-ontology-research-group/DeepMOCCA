## Results

- `patients_representations.txt` - The representation of features for each patient generated after the 2nd `Conv2` layer.

- `Top_10_ranked_genes_all_samples.tar.gz` - The top 10 ranked genes for each patient genrated from attention-based mechanism which specifiy wheather a gene is:
  * `Driver in the same cancer` -> `0` if yes and `1` if no.
  * `Driver in other cancer` -> `0` if yes and `1` if no.
  * `Prognostic in the same cancer` -> `0` if yes and `1` if no.
  * `Prognostic in other cancer` -> `0` if yes and `1` if no.

- `rank1.tsv` - The top 1 ranked genes over all the patients across all the cancer types, last column specify the number of patients having this specific gene within the top 1 and for that specific cancer type.

- `rank10.tsv` - The top 10 ranked genes over all the patients across all the cancer types combined from `ranked_genes_top10.tar.gz` and last column specifying the number of patients having this specific gene within the top 10 and for that specific cancer type.

- `top-novel.tsv` - The top 1 ranked genes over all the patients across all the cancer types which are not known driver genes or prognostic markers.

- `sorted/` - Genes that are ranked significantly higher by our model's graph attention mechanism across all samples within each cohort/cancer type than expected under a uniform random distribution; reported their p-value, effect size, Holm–Bonferroni correction and Benjamini-Hochberg correction sorted by the effect size.

- `filtered/` - Same list of genes in `sorted/` filtered by p<0.2 (Benjamini-Hochberg correction) and sorted by the effect size.
