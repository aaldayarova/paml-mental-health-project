import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Linear Regression Class
# ------------------------------
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.loss_history = []

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.W = np.zeros(num_features)
        self.b = 0

        for _ in range(self.num_iterations):
            y_pred = np.dot(X, self.W) + self.b
            error = y_pred - y

            dW = (1 / num_samples) * np.dot(X.T, error)
            db = (1 / num_samples) * np.sum(error)

            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

            loss = (1 / (2 * num_samples)) * np.sum(error ** 2)
            self.loss_history.append(loss)

        return self.W, self.b, self.loss_history

    def predict(self, X):
        return np.dot(X, self.W) + self.b

# ------------------------------
# Load and Preprocess Dataset
# ------------------------------
df = pd.read_csv("Mental_Health_Lifestyle_2019_2024.csv")

# Drop missing values
df = df.dropna()

# Optional: encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Select target and features
target = 'stress'  # You can change to 'anxiety' or 'depression'
X = df.drop(columns=['stress', 'anxiety', 'depression'])
y = df[target]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.2, random_state=42)

# ------------------------------
# Train Linear Regression Model
# ------------------------------
model = LinearRegressionGD(learning_rate=0.01, num_iterations=1000)
W, b, loss_history = model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# ------------------------------
# Evaluate Model
# ------------------------------
mse = np.mean((y_pred - y_test) ** 2)
r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

print("MSE:", mse)
print("R² Score:", r2)

# ------------------------------
# Plot Loss Curve
# ------------------------------
plt.plot(loss_history)
plt.title("Loss Curve (Gradient Descent)")
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.grid(True)
plt.show()
