import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Sample dataset: budget, number of stars, runtime, and IMDB rating
data = {
    'budget': [50, 100, 30, 75, 120, 80, 40, 95, 110, 65],
    'stars': [3, 4, 2, 3, 5, 4, 2, 5, 4, 3],
    'runtime': [120, 150, 90, 140, 180, 130, 100, 175, 160, 115],
    'rating': [7.2, 8.5, 6.0, 7.8, 9.0, 8.2, 6.5, 8.9, 8.1, 7.0]
}

# Create DataFrame
df = pd.DataFrame(data)
print("Dataset:")
print(df)
print("\n")

# Features (X) and target (y)
X = df[['budget', 'stars', 'runtime']]
y = df['rating']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the model
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("Model Performance:")
print(f"Training MSE: {mse_train:.4f}")
print(f"Testing MSE: {mse_test:.4f}")
print(f"Training R² Score: {r2_train:.4f}")
print(f"Testing R² Score: {r2_test:.4f}")
print("\n")

# Display predictions vs actual
print("Test Set Predictions vs Actual:")
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
print(results)
print("\n")

# Predict rating for a new movie
# New movie: budget=60M, stars=4, runtime=125 min
new_movie = np.array([[60, 4, 125]])
predicted_rating = model.predict(new_movie)[0]
print(f"New Movie Prediction:")
print(f"Budget: $60M, Stars: 4, Runtime: 125 min")
print(f"Predicted Rating: {predicted_rating:.2f}")

# Model coefficients
print("\nModel Coefficients:")
print(f"Budget coefficient: {model.coef_[0]:.4f}")
print(f"Stars coefficient: {model.coef_[1]:.4f}")
print(f"Runtime coefficient: {model.coef_[2]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, color='blue', alpha=0.6, label='Test Data')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('Movie Rating Prediction - Actual vs Predicted')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()