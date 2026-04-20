# Prompt Performance Analysis

Editable Python project for analyzing prompt performance across diffusion models,
datasets, categories, and manual evaluation metric families.

The script expects columns like:

- `dataset_name`, `model_name`, `file_name`
- `r1_com_*` for common metrics across datasets
- `r1_d2_*`, `r1_d3_*`, `r1_d4_*` for dataset-specific metrics
- optional `category`, `super_category`, `manual_eval_negative`, `manual_eval_positive.`

## Setup

From VS Code, open this folder and select a Python environment that has the
packages in `requirements.txt`.

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python src/analyze_prompt_performance.py `
  --csv "C:/Users/anujq/OneDrive/Documents/Amrita/curated_dataset_diffusion/diffusion_evaluation/manual_eval_with_com_columns.csv" `
  --out outputs/research_grade_analysis
```

## Outputs

The script writes:

- `outputs/research_grade_analysis/tables/*.csv`
- `outputs/research_grade_analysis/figures/*.png`
- `outputs/research_grade_analysis/figures/*.pdf`
- `outputs/research_grade_analysis/analysis_summary.md`

Figures are saved at high DPI and also as PDF for paper/thesis use.

