import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(0)
x = np.random.rand(100)
y = 3 * x + np.random.normal(0, 0.2, size=100)

# -------------------- NumPy --------------------
# NumPy array math
x_np = x.reshape(-1, 1)
y_np = y.reshape(-1, 1)
dot_product = np.dot(x_np.T, y_np)  # Example of matrix multiplication

# -------------------- Pandas --------------------
# Create DataFrame for structured handling
df = pd.DataFrame({'x': x, 'y': y})
mean_y = df['y'].mean()  # Compute column mean using Pandas

# -------------------- Matplotlib --------------------
plt.figure(figsize=(5, 4))
plt.scatter(x, y, color='blue', label='Data points')
plt.title('Matplotlib Scatter Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# -------------------- Seaborn --------------------
plt.figure(figsize=(5, 4))
sns.regplot(x='x', y='y', data=df)
plt.title('Seaborn Regression Plot')
plt.show()

# -------------------- Scikit-learn --------------------
# Train a simple linear regression model
X_train, X_test, y_train, y_test = train_test_split(x_np, y_np, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, predictions)

# Display model coefficient and error
model_coef = model.coef_[0][0]
model_intercept = model.intercept_[0]

model_coef, model_intercept, mse
