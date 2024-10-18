import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error
import joblib

# Load the data
features = pd.read_csv("data/features.csv")
labels = pd.read_csv("data/labels.csv")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Initialize the XGBoost regressor
xg_reg = xgb.XGBRegressor()

# Define the parameter grid for cross-validation
param_grid = {
    "n_estimators": [250, 300, 350, 400],
    "max_depth": [4, 5, 6, 7],
    "learning_rate": [0.05, 0.08, 0.1, 0.14],
    "subsample": [0.6, 0.65, 0.7, 0.75],
}

# Perform cross-validation to tune hyperparameters
grid_search = GridSearchCV(
    estimator=xg_reg,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    verbose=4,
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

# Get the best parameters and train the final model
best_params = grid_search.best_params_
final_model = xgb.XGBRegressor(**best_params, objective="reg:squarederror")
final_model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = final_model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {rmse}")
# Save the model to a file
model_filename = "xgboost_model.pkl"
joblib.dump(final_model, model_filename)
print(f"Model saved to {model_filename}")
