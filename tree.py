import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, KFold
import joblib


# Load the data
features = pd.read_csv("data/features.csv")
labels = pd.read_csv("data/labels.csv")


# Initialize the XGBoost regressor
xg_reg = xgb.XGBRegressor()

# Define the parameter grid for cross-validation
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.1, 0.01, 0.001],
}

# Perform cross-validation to tune hyperparameters
grid_search = GridSearchCV(
    estimator=xg_reg,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    verbose=4,
    n_jobs=-1,
    cv=KFold(n_splits=5, shuffle=True),
)
grid_search.fit(features, labels)

# Get the best parameters and train the final model
best_params = grid_search.best_params_
final_model = xgb.XGBRegressor(**best_params, objective="reg:squarederror")
final_model.fit(features, labels)

# Save the model to a file
model_filename = "xgboost_model.pkl"
joblib.dump(final_model, model_filename)
print(f"Model saved to {model_filename}")
