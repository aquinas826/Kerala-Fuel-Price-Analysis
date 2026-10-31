# Kerala Fuel Price Analysis and Prediction Using Python

A Python data-analysis and machine-learning project exploring historical petrol and diesel prices across districts of Kerala.

## Project objectives

1. Analyze historical petrol and diesel price trends in Kerala.
2. Compare average fuel prices across districts.
3. Compare petrol and diesel price distributions.
4. Identify potential outliers and examine price variability.
5. Build and compare regression models for fuel-price prediction.

## Dataset

The file `data/Kerala_Fuel_Prices.csv` contains 13,608 rows and five columns:

- `Date` — date of the record
- `District` — district name
- `Fuel_Type` — fuel type (for example, Petrol or Diesel)
- `Price` — recorded fuel price
- `Price_Change` — recorded change in price

The original public source/URL was not provided with the file. Add it here if known; the dataset has not been independently verified against an official source.

## Analysis and visualizations

The script performs data inspection and cleaning, standardizes district and fuel labels, summarizes prices, and generates:
- Fuel-price trends over time
- Average prices by district and fuel type
- Boxplots and histograms for price distributions
- IQR-based potential-outlier analysis
- A correlation heatmap
- Regression-model metric comparisons

## Machine-learning models

The code trains and compares Simple Linear Regression, Multiple Linear Regression, Polynomial Regression, Random Forest Regression, and Multi-layer Perceptron (MLP) Regression.

Evaluation metrics include MAE, MSE, RMSE, and R². The script chooses the best model by lowest MAE on held-out test data. Results depend on the supplied dataset and are estimates, not guaranteed future prices.

## Setup

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/aquinas826/Kerala-Fuel-Price-Analysis.git
cd Kerala-Fuel-Price-Analysis
python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies and run:

```bash
pip install -r requirements.txt
python fuel_price_analysis.py
```

The script opens charts with Matplotlib and saves selected correlation/model-comparison plots in the `figures/` directory.

## Project structure

```text
Kerala-Fuel-Price-Analysis/
├── data/
│   └── Kerala_Fuel_Prices.csv
├── figures/
├── fuel_price_analysis.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Limitations and future scope

- Confirm the dataset source and validate its values against reliable records.
- Add factors such as crude-oil prices, taxes, exchange rates, and inflation.
- Tune models and test them on additional unseen periods.
- Build an interactive dashboard with Streamlit or Flask.
- Extend the analysis to other states.

## References

- [Python documentation](https://docs.python.org/3/)
- [pandas documentation](https://pandas.pydata.org/docs/)
- [NumPy documentation](https://numpy.org/doc/)
- [Matplotlib documentation](https://matplotlib.org/stable/)
- [Seaborn documentation](https://seaborn.pydata.org/)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
