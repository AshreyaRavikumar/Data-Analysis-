# Sales Transaction Analysis

Data science project — clean, analyze, visualize, and predict sales data.

## What's Inside

```
├── generate_dataset.py          # creates the raw dataset
├── analysis.py                  # full pipeline: clean → EDA → charts → model
├── data/
│   ├── sales_transactions_raw.csv   # raw (dirty) data
│   ├── 02_cleaned.csv               # cleaned data
│   └── 03_final_analysis.csv        # final analysis snapshot
└── outputs/charts/              # 11 saved charts
```

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python generate_dataset.py   # skip if you already have the CSV
python analysis.py
```

## What It Does

**Phase 1 — Data Foundations**
- Loads 2000-row sales dataset (10 products, 5 regions, 2 years)
- Profiles data types, sizes, feature descriptions
- Cleans: removes duplicates, fixes bad dates/negatives, imputes missing values

**Phase 2 — Analysis & Visualization**
- Descriptive stats, correlation matrix, outlier detection (IQR)
- 11 charts: line, bar, pie, heatmap, histogram, scatter
- Each chart has a one-line insight

**Phase 3 — Modeling**
- Linear Regression vs Random Forest for revenue prediction
- 80/20 train-test split, evaluated on R2, MAE, RMSE
- Feature importance ranking

## Key Results

| Metric | Linear Regression | Random Forest |
|--------|------------------|---------------|
| R2     | ~0.83            | ~0.99         |
| MAE    | ~$390            | ~$52          |
| RMSE   | ~$589            | ~$122         |

**Top predictors:** UnitPrice and Quantity dominate revenue.

## Dependencies

pandas, numpy, matplotlib, seaborn, scikit-learn
