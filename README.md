# Claims frequency forecasting and price-rule checks

My solution to two take-home exercises from Ominimo, an insurance company: predicting how many claims a car policy will have per year, and checking that an insurance price list respects its coverage order (MTPL < Limited Casco < Casco).

## Results

Claims forecasting on 678,013 French motor policies (freMTPL2), evaluated on a stratified 20% holdout:

| Model | Poisson deviance vs. baseline | Predicted / actual claims | Normalised Gini |
|---|---:|---:|---:|
| Portfolio claim rate (baseline) | 0% | 1.000 | 0 |
| Poisson GLM (splines) | −5.2% | 0.998 | 0.302 |
| LightGBM (Poisson) | **−8.8%** | 0.999 | 0.354 |

Observed claim frequency goes from 0.038 in the lowest predicted decile to 0.316 in the highest, an 8.4× spread; LightGBM predicts 8.0×.

Price-rule check: in the example price list, 11 of 24 comparisons break the coverage order, and all 11 are in the Limited vs. Casco block. Because 91.7% of that block fails, the correction raises the Casco block instead of patching single prices.

## Decisions worth noting

- **No rows dropped.** Instead of removing outliers, I measured each suspicious group's share of the baseline deviance (policies with more than four claims: 0.372%; exposure below 0.02 years: 2.031%) and kept every row. Notebook 04 then checks that excluding short exposures does not change which model wins.
- **Exposure is a weight, not a feature.** The brief says exposure is one or two years, but the data has a continuous 0.0027 to 2.01 years, so I model annual frequency as `ClaimNb / Exposure` weighted by exposure.
- **The holdout was opened once.** Grids are small and fixed before evaluation; the LightGBM choice sits at the edge of its grid, and I did not expand the search after seeing test results.

## Layout

```text
claims-forecasting/
  01_data_preparation.ipynb   download, exposure analysis, cleaning decisions
  02_exploration.ipynb        frequency by feature
  03_modelling.ipynb          baseline, GLM, LightGBM, holdout metrics
  04_error_analysis.ipynb     residuals, permutation importance, deciles
business-assignment/
  business_assignment.ipynb   price-order rules and the correction mechanism
```

## Running

Use Python 3.11 or 3.12:

```bash
python -m venv .venv
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm pyarrow jupyter
jupyter notebook
```

Run the claims notebooks in order. Notebook 01 downloads the source data and writes the git-ignored `data/` files the later notebooks need.
