# Prompt Performance Analysis Summary

Input CSV: `C:\Users\anujq\OneDrive\Documents\Amrita\curated_dataset_diffusion\diffusion_evaluation\manual_eval_with_com_columns.csv`
Rows: 2,199
Models: 11
Datasets: 4
Metric groups: com, d2, d3, d4

## Top Models By Common Composite

| Rank | Model | Mean | 95% CI | N |
|---:|---|---:|---:|---:|
| 1 | openai_gpt_image_1 | 3.984 | 3.802-4.158 | 198 |
| 2 | SD_api_free | 3.110 | 2.913-3.308 | 86 |
| 3 | openai_dalle_3 | 3.040 | 2.822-3.261 | 163 |
| 4 | SD_api_iteration_2_stable-diffusion-xl-1024-v1-0 | 2.901 | 2.618-3.207 | 157 |
| 5 | SD_api_iteration_1_stable-diffusion-xl-1024-v1-0 | 2.761 | 2.478-3.041 | 157 |
| 6 | Stable Diffusion 3.5 Large | 2.421 | 2.183-2.654 | 120 |
| 7 | flux1_1_pro | 2.190 | 1.944-2.458 | 108 |
| 8 | flux1-dev | 2.184 | 2.013-2.364 | 198 |

## Notes

- `com_mean_score` is the row-wise mean of all `*_com_*` metric columns.
- `dataset_specific_mean_score` is the row-wise mean of all `*_d*_*` metric columns.
- `overall_mean_score` averages common and dataset-specific composites when available.
- Confidence intervals are non-parametric bootstrap intervals over row-level scores.
- Pairwise model comparisons use Mann-Whitney U tests and Cliff's delta.
