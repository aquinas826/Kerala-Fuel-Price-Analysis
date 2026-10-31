import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
DATA_PATH = PROJECT_DIR / "data" / "Kerala_Fuel_Prices.csv"
df = pd.read_csv(DATA_PATH)

print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

# missing values
print("Missing values before cleaning:")
print(df.isnull().sum())

# duplicate rows
print("\nDuplicate rows before cleaning:")
print(df.duplicated().sum())

# columns to the correct data types
df["date"] = pd.to_datetime(
    df["date"],
    format="%d-%m-%Y",
    errors="coerce"
)

df["district"] = df["district"].astype("string").str.strip()

df["fuel_type"] = (
    df["fuel_type"]
    .astype("string")
    .str.strip()
    .str.title()
)

df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["price_change"] = pd.to_numeric(
    df["price_change"],
    errors="coerce"
)

df = df.drop_duplicates().copy()

district_mapping = {
    "alapuzha": "Alappuzha",
    "alleppey": "Alappuzha",
    "trivandrum": "Thiruvananthapuram",
    "calicut": "Kozhikode",
    "cochin": "Ernakulam",
    "kochi": "Ernakulam",
    "quilon": "Kollam",
    "palghat": "Palakkad",
    "cannanore": "Kannur"
}

district_lower = df["district"].str.lower()

df["district"] = (
    district_lower.map(district_mapping)
    .fillna(df["district"].str.title())
)

# only valid essential records
df = df.dropna(
    subset=["date", "district", "fuel_type", "price"]
).copy()

# Removing impossible or invalid non-positive prices
df = df[df["price"] > 0].copy()


if df["price_change"].isna().any():
    if df["price_change"].notna().any():
        df["price_change"] = df["price_change"].fillna(
            df["price_change"].median()
        )

# Sort records chronologically
df = df.sort_values(
    ["date", "district", "fuel_type"]
).reset_index(drop=True)


print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows after cleaning:")
print(df.duplicated().sum())

print("\nCleaned dataset shape:")
print(df.shape)

print("\nDistricts:")
print(sorted(df["district"].dropna().unique()))

print("\nFuel types:")
print(sorted(df["fuel_type"].dropna().unique()))


objectives = [
    "1. Analyze historical petrol and diesel price trends in Kerala.",
    "2. Compare average fuel prices across different districts.",
    "3. Compare petrol and diesel price distributions.",
    "4. Identify unusual prices and examine price variability.",
    "5. Build and compare regression models for fuel-price prediction."
]

print("PROJECT OBJECTIVES\n")

for objective in objectives:
    print(objective)
    
    
    
print("Number of records:", len(df))
print("Number of districts:", df["district"].nunique())
print("Number of fuel types:", df["fuel_type"].nunique())
print("Date range:", df["date"].min(), "to", df["date"].max())

print("\nFuel-price summary:")
print(df["price"].describe())

print("\nAverage price by fuel type:")
print(df.groupby("fuel_type")["price"].agg(
    ["mean", "median", "min", "max", "std"]
))

print("\nAverage price by district:")
print(df.groupby("district")["price"].mean().sort_values(
    ascending=False
))

print("\nAverage price by district and fuel type:")
print(df.pivot_table(
    index="district",
    columns="fuel_type",
    values="price",
    aggfunc="mean"
))


daily_prices = df.groupby(
    ["date", "fuel_type"]
)["price"].mean().reset_index()

plt.figure(figsize=(14, 6))

sns.lineplot(
    data=daily_prices,
    x="date",
    y="price",
    hue="fuel_type"
)

plt.title("Fuel Price Trends Over Time in Kerala")
plt.xlabel("Date")
plt.ylabel("Average Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



district_prices = df.groupby(
    ["district", "fuel_type"]
)["price"].mean().reset_index()

plt.figure(figsize=(14, 7))

sns.barplot(
    data=district_prices,
    x="district",
    y="price",
    hue="fuel_type"
)

plt.title("Average Fuel Prices Across Kerala Districts")
plt.xlabel("District")
plt.ylabel("Average Price")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.show()


plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="fuel_type",
    y="price"
)

plt.title("Fuel Price Distribution by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Price")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="price",
    hue="fuel_type",
    kde=True,
    element="step"
)

plt.title("Distribution of Fuel Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df["price"] < lower_bound) |
    (df["price"] > upper_bound)
]

print("Lower IQR boundary:", lower_bound)
print("Upper IQR boundary:", upper_bound)
print("Number of potential outliers:", len(outliers))

print("\nPotential outlier records:")
print(outliers.head(20))

plt.figure(figsize=(10, 5))

sns.scatterplot(
    data=df,
    x="date",
    y="price",
    hue="fuel_type",
    s=20
)

plt.axhline(
    upper_bound,
    linestyle="--",
    label="Upper IQR boundary"
)

plt.axhline(
    lower_bound,
    linestyle="--",
    label="Lower IQR boundary"
)

plt.title("Potential Fuel Price Outliers Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()



correlation_data = df[["price", "price_change"]].corr()

print("Correlation matrix:")
print(correlation_data)

plt.figure(figsize=(7, 5))

sns.heatmap(
    correlation_data,
    annot=True,
    cmap="coolwarm",
    fmt=".3f"
)

plt.title("Correlation Between Price and Price Change")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "07_correlation_heatmap.png", dpi=300)
plt.show()


# Create a separate copy for feature engineering
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_year"] = df["date"].dt.dayofyear

# Number of days since the beginning of the dataset
df["days_since_start"] = (
    df["date"] - df["date"].min()
).dt.days

print("Dataset after feature engineering:")
print(df.head())

print("\nNew features:")
print([
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "days_since_start"
])





model_data = df.copy()
model_dates = model_data["date"].copy()


model_data = model_data.drop(
    columns=["date", "price_change"]
)

# Remove missing values and keep dates aligned
model_data = model_data.dropna()
model_dates = model_dates.loc[model_data.index]

# Encode categorical columns using LabelEncoder
le_district = LabelEncoder()
le_fuel = LabelEncoder()

model_data["district"] = le_district.fit_transform(
    model_data["district"]
)

model_data["fuel_type"] = le_fuel.fit_transform(
    model_data["fuel_type"]
)

# Separate features and target
X = model_data.drop(columns=["price"])
y = model_data["price"]

# Split chronologically: 80% training, 20% testing
dates = np.sort(model_dates.unique())
split_date = dates[int(len(dates) * 0.8)]

train = model_dates < split_date
test = model_dates >= split_date

X_train, X_test = X.loc[train], X.loc[test]
y_train, y_test = y.loc[train], y.loc[test]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Number of features:", X_train.shape[1])


#Simple Linear Regression
# Use time to predict fuel price
simple_model = LinearRegression()

simple_model.fit(
    X_train[["days_since_start"]], y_train
)

simple_predictions = simple_model.predict(
    X_test[["days_since_start"]]
)

print("Simple Linear Regression completed")


#Multiple Linear Regression 
multiple_model = LinearRegression()

multiple_model.fit(X_train, y_train)

multiple_predictions = multiple_model.predict(X_test)

print("Multiple Linear Regression completed")

#Polynomial Regression
poly = PolynomialFeatures(degree=2, include_bias=False)

X_train_poly = poly.fit_transform(
    X_train[["days_since_start"]]
)

X_test_poly = poly.transform(
    X_test[["days_since_start"]]
)

polynomial_model = LinearRegression()

polynomial_model.fit(X_train_poly, y_train)

polynomial_predictions = polynomial_model.predict(X_test_poly)

print("Polynomial Regression completed")

#Random Forest Regression
random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    min_samples_leaf=2
)

random_forest_model.fit(X_train, y_train)

random_forest_predictions = random_forest_model.predict(X_test)

print("Random Forest Regression completed")

#MLP Regression
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp_model = MLPRegressor(
    hidden_layer_sizes=(50,),
    max_iter=1000,
    random_state=42,
    early_stopping=True
)

mlp_model.fit(X_train_scaled, y_train)

mlp_predictions = mlp_model.predict(X_test_scaled)

print("MLP Regression completed")





# Evaluate model performance
results = []

predictions = {
    "Simple Linear Regression": simple_predictions,
    "Multiple Linear Regression": multiple_predictions,
    "Polynomial Regression": polynomial_predictions,
    "Random Forest Regression": random_forest_predictions,
    "MLP Regression": mlp_predictions
}

for name, pred in predictions.items():
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)

    results.append([name, mae, mse, rmse, r2])

# comparison table
results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "MSE", "RMSE", "R2"]
)

results_df = results_df.sort_values("MAE").reset_index(drop=True)

print("Model Evaluation Results:")
print(results_df.round(4))



# Plot MAE comparison
plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x="Model", y="MAE")
plt.title("MAE Comparison of Regression Models")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "08_mae_comparison.png", dpi=300)
plt.show()

# Plot RMSE comparison
plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x="Model", y="RMSE")
plt.title("RMSE Comparison of Regression Models")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "08_rmse_comparison.png", dpi=300)
plt.show()

# Plot R2 comparison
plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x="Model", y="R2")
plt.title("R2 Score Comparison of Regression Models")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "08_r2_comparison.png", dpi=300)
plt.show()

# Identify the best model based on MAE
best_model = results_df.iloc[0]

print("\nBest Model:", best_model["Model"])
print("MAE:", round(best_model["MAE"], 4))
print("RMSE:", round(best_model["RMSE"], 4))
print("R2:", round(best_model["R2"], 4))


print("PROJECT FINDINGS")

print("\n1. Dataset Overview")
print("Total records:", len(df))
print("Number of districts:", df["district"].nunique())
print("Fuel types:", df["fuel_type"].unique())
print("Date range:", df["date"].min(), "to", df["date"].max())

print("\n2. Fuel Price Statistics")
print(df["price"].describe())

print("\n3. Average Price by Fuel Type")
print(df.groupby("fuel_type")["price"].mean().round(2))


print("\n4. Average Price by District")
district_prices = df.groupby("district")["price"].mean().sort_values(
    ascending=False
)
print(district_prices.round(2))

print("Highest average-price district:", district_prices.index[0])
print("Lowest average-price district:", district_prices.index[-1])

print("\n5. Potential Outliers")
q1 = df["price"].quantile(0.25)
q3 = df["price"].quantile(0.75)
iqr = q3 - q1

outliers = df[
    (df["price"] < q1 - 1.5 * iqr) |
    (df["price"] > q3 + 1.5 * iqr)
]

print("Potential outlier records:", len(outliers))

print("\n6. Model Evaluation")
print(results_df.round(4).to_string(index=False))

best_model = results_df.iloc[0]
print("\nBest model based on lowest MAE:", best_model["Model"])
print("MAE:", round(best_model["MAE"], 4))
print("RMSE:", round(best_model["RMSE"], 4))
print("R2:", round(best_model["R2"], 4))


print("\n7. Recommendations")
print("1. Verify potential outliers against original records.")
print("2. Collect more historical data for better analysis.")
print("3. Compare MAE, RMSE and R2 before selecting a model.")
print("4. Test predictions on dates not used for training.")
print("5. Treat predictions as estimates, not guaranteed future prices.")

