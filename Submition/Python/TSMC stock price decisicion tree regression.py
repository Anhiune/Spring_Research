import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load local dataset
file_path = r"C:\Users\hoang\OneDrive - University of St. Thomas\Forecasting Spring 2025\project_data\price_TSM_Anh_Bui.csv"
df = pd.read_csv(file_path)

# Convert forecast_eps to numeric
df['forecast_eps'] = pd.to_numeric(df['forecast_eps'], errors='coerce')

# Drop rows with missing values
essential_cols = ['forecast_eps', 'actual_eps', 'adj_close', 'sma4_actual_eps']
model_df = df[essential_cols].dropna()
# 2. Train models and compute predictions

# Features and target
features = ['forecast_eps', 'sma4_actual_eps', 'adj_close']
X = model_df[features]
y = model_df['actual_eps']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree
tree = DecisionTreeRegressor(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
tree_preds = tree.predict(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
# 3. Visualize feature importance
# Plot feature importance from decision tree
importances = tree.feature_importances_
feature_names = np.array(features)

plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances, color='mediumseagreen')
plt.title("Feature Importance (Decision Tree)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# Save full predictions
model_df['DecisionTree_Pred'] = tree.predict(X)
model_df['LinearRegression_Pred'] = lr.predict(X)

# Save locally
output_path = r"C:\Users\hoang\OneDrive - University of St. Thomas\Forecasting Spring 2025\project_data\TSMC_Earnings_Model_Output.csv"
model_df.to_csv(output_path, index=False)

print(f"Exported predictions to: {output_path}")