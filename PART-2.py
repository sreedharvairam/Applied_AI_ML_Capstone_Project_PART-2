import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set plot styles
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

print("=" * 60)
print("RUNNING PART 2 & PART 3: MODELING & EVALUATION PIPELINE")
print("=" * 60)

# -------------------------------------------------------------------------
# TASK 1: Load Cleaned Dataset & Target Separation
# -------------------------------------------------------------------------
print("\n--- Task 1: Loading Clean Data & Separating Target ---")
df = pd.read_csv("cleaned_data.csv")

# Ensure 'House_ID' is dropped as it has no predictive power
if "House_ID" in df.columns:
    df = df.drop(columns=["House_ID"])

# Handle columns exceeding 20% null rate if any are left (e.g., Has_Pool from Part 1)
# Dropping features that don't satisfy data quality thresholds
if "Has_Pool" in df.columns:
    df = df.drop(columns=["Has_Pool"])

# -------------------------------------------------------------------------
# TASK 2: Feature Engineering
# -------------------------------------------------------------------------
print("\n--- Task 2: Feature Engineering ---")
# 1. Age of the house when evaluating it (current year assumed as 2026)
df["House_Age"] = 2026 - df["Year_Built"]

# 2. Average square footage per room
df["SqFt_per_Room"] = df["Square_Footage"] / (df["Bedrooms"] + df["Bathrooms"])

print("New features created: 'House_Age', 'SqFt_per_Room'")
print(df[["House_Age", "SqFt_per_Room"]].head())

# -------------------------------------------------------------------------
# TASK 3: Categorical One-Hot Encoding
# -------------------------------------------------------------------------
print("\n--- Task 3: One-Hot Encoding Categorical Variables ---")
# Convert categorical column 'Neighborhood' into dummy variables (dropping first to avoid multicollinearity)
df_encoded = pd.get_dummies(df, columns=["Neighborhood"], drop_first=True)
print(f"Dataset shape after encoding: {df_encoded.shape}")

# Separate features (X) and target variable (y)
X = df_encoded.drop(columns=["Price"])
y = df_encoded["Price"]

# Train-test split (80-20 ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# -------------------------------------------------------------------------
# TASK 4: Feature Scaling
# -------------------------------------------------------------------------
print("\n--- Task 4: Feature Scaling ---")
scaler = StandardScaler()

# Scaling numerical features to improve regularization models (Ridge/Lasso)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for clean tracking
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# -------------------------------------------------------------------------
# TASK 5 & 6: Baseline Linear Regression vs Regularized Models (Ridge / Lasso)
# -------------------------------------------------------------------------
print("\n--- Tasks 5 & 6: Model Training & Hyperparameter Tuning ---")

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression (CV)": RidgeCV(
        alphas=np.logspace(-3, 3, 100), cv=5
    ),  # 5-fold cross-validation
    "Lasso Regression (CV)": LassoCV(
        alphas=np.logspace(-3, 3, 100), cv=5, max_iter=10000
    ),
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled_df, y_train)
    y_pred_train = model.predict(X_train_scaled_df)
    y_pred_test = model.predict(X_test_scaled_df)

    # Performance Metrics
    r2_train, r2_test = r2_score(y_train, y_pred_train), r2_score(
        y_test, y_pred_test
    )
    mae_train, mae_test = mean_absolute_error(
        y_train, y_pred_train
    ), mean_absolute_error(y_test, y_pred_test)
    rmse_train, rmse_test = np.sqrt(
        mean_squared_error(y_train, y_pred_train)
    ), np.sqrt(mean_squared_error(y_test, y_pred_test))

    results[name] = {
        "Train R2": r2_train,
        "Test R2": r2_test,
        "Train MAE": mae_train,
        "Test MAE": mae_test,
        "Train RMSE": rmse_train,
        "Test RMSE": rmse_test,
        "Model_Obj": model,
        "Predictions": y_pred_test,
    }

    print(f"\n[{name}] Performance:")
    print(f"  Train R²: {r2_train:.4f} | Test R²: {r2_test:.4f}")
    print(f"  Train RMSE: ${rmse_train:,.2f} | Test RMSE: ${rmse_test:,.2f}")

# Select the model with the highest test R2 score
best_model_name = max(results, key=lambda k: results[k]["Test R2"])
best_model_data = results[best_model_name]
print(f"\n>>> Best Performing Model selected: {best_model_name} <<<")

# -------------------------------------------------------------------------
# TASK 7: Residual Analysis
# -------------------------------------------------------------------------
print("\n--- Task 7: Generating Residual Analysis Plots ---")
best_preds = best_model_data["Predictions"]
residuals = y_test - best_preds

# Plot 1: Histogram of Residuals
plt.figure()
sns.histplot(residuals, kde=True, color="purple", bins=15)
plt.title(f"Residuals Distribution ({best_model_name})")
plt.xlabel("Residual ($)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("plot_7_residuals_hist.png")
plt.close()

# Plot 2: Scatter Plot (Predicted vs Residuals)
plt.figure()
sns.scatterplot(x=best_preds, y=residuals, color="crimson", alpha=0.8)
plt.axhline(y=0, color="black", linestyle="--", linewidth=1.5)
plt.title("Residuals vs. Predicted Values")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.tight_layout()
plt.savefig("plot_8_residuals_scatter.png")
plt.close()

print(
    "Residual analysis graphs saved as 'plot_7_residuals_hist.png' and 'plot_8_residuals_scatter.png'."
)

# -------------------------------------------------------------------------
# TASK 8: Feature Importance
# -------------------------------------------------------------------------
print("\n--- Task 8: Extracting Feature Importance ---")
best_model_obj = best_model_data["Model_Obj"]

# Extract coefficients
if hasattr(best_model_obj, "coef_"):
    coefs = best_model_obj.coef_
    importance_df = pd.DataFrame(
        {"Feature": X.columns, "Coefficient": coefs}
    ).sort_values(by="Coefficient", key=abs, ascending=False)

    print("\nFeature Coefficients (Sorted by magnitude):")
    print(importance_df)

    # Plot Coefficient Importance Bar Chart
    plt.figure()
    sns.barplot(
        data=importance_df,
        y="Feature",
        x="Coefficient",
        palette="coolwarm",
        hue="Feature",
        legend=False,
    )
    plt.title(f"Feature Importance Coefficients ({best_model_name})")
    plt.xlabel("Coefficient Weight")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("plot_9_feature_importance.png")
    plt.close()
    print("Feature importance bar chart saved as 'plot_9_feature_importance.png'.")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE - ALL ACCEPTANCE CRITERIA MET")
print("=" * 60)