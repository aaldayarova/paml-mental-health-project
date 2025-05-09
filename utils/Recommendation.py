import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

# Load and clean data
import os

BASE_DIR = os.path.dirname(__file__)
csv_path = os.path.join(BASE_DIR, '..', 'student_depression_dataset.csv')
df = pd.read_csv(csv_path)


# Rename columns to match our code usage
df = df.rename(columns={
    'Gender': 'gender',
    'Age': 'age',
    'Academic Pressure': 'academic_pressure',
    'Work Pressure': 'work_pressure',
    'Study Satisfaction': 'study_satisfaction',
    'Job Satisfaction': 'job_satisfaction',
    'Sleep Duration': 'sleep_duration',
    'Dietary Habits': 'dietary_habits',
    'Have you ever had suicidal thoughts ?': 'suicidal_thoughts',
    'Work/Study Hours': 'work_study_hours',
    'Financial Stress': 'financial_stress',
    'Family History of Mental Illness': 'family_history',
    'Depression': 'depression'
})


df = df.drop(columns=['id', 'City', 'Degree', 'Profession', 'CGPA'])

# Split
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
y_train = train_data['depression'].values
y_test = test_data['depression'].values

# Define qualitative and quantitative variables with consistent snake_case naming
quali_vars = ['gender', 'sleep_duration', 'dietary_habits', 'suicidal_thoughts', 'family_history',
            'work_pressure', 'academic_pressure', 'study_satisfaction', 'job_satisfaction', 'financial_stress']
quanti_vars = ['age', 'work_study_hours']

# === Manual Imputation and Standardization ===

# 1. Impute with median and compute mean/std manually
means, stds = {}, {}

def standardize_manual(col_data, col_name):
    median = np.nanmedian(col_data)
    col_data = np.where(np.isnan(col_data), median, col_data)  # Fill NaNs
    mean = np.mean(col_data)
    std = np.std(col_data)
    means[col_name] = mean
    stds[col_name] = std
    return (col_data - mean) / std

# Apply manual standardization
X_train_quanti = np.column_stack([
    standardize_manual(train_data[col].values, col) for col in quanti_vars
])
X_test_quanti = np.column_stack([
    (np.where(np.isnan(test_data[col]), np.nanmedian(train_data[col]), test_data[col]) - means[col]) / stds[col]
    for col in quanti_vars
])

# === One-Hot Encode Categorical Features ===
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_quali = encoder.fit_transform(train_data[quali_vars])
X_test_quali = encoder.transform(test_data[quali_vars])

# === Combine Final Features ===
X_train_final = np.hstack((X_train_quanti, X_train_quali))
X_test_final = np.hstack((X_test_quanti, X_test_quali))

# === Linear Regression from Scratch ===
class ScratchLinearRegression:
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add bias
        self.theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta

    def evaluate(self, y_true, y_pred):
        acc = accuracy_score(y_true, np.round(y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return acc, mae, r2

# === Train and Evaluate ===
model = ScratchLinearRegression()
model.fit(X_train_final, y_train)

# Predictions
y_train_pred = model.predict(X_train_final)
y_test_pred = model.predict(X_test_final)

# Metrics
train_acc, train_mae, train_r2 = model.evaluate(y_train, y_train_pred)
test_acc, test_mae, test_r2 = model.evaluate(y_test, y_test_pred)

print(f"Train: Accuracy={train_acc:.2f}, MAE={train_mae:.2f}, R²={train_r2:.2f}")
print(f"Test:  Accuracy={test_acc:.2f}, MAE={test_mae:.2f}, R²={test_r2:.2f}")



import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

class ScratchSVM:
    def __init__(self, learning_rate=0.01, regularization_param=0.1, max_iters=1000):
        self.learning_rate = learning_rate
        self.regularization_param = regularization_param
        self.max_iters = max_iters
        self.w = None  # Weights (coefficients)
        self.b = None  # Bias term

    def hinge_loss(self, X, y):
        # Hinge loss for linear SVM
        margin = y * (np.dot(X, self.w) + self.b)
        loss = np.mean(np.maximum(0, 1 - margin))  # Hinge loss (with margin)
        return loss

    def fit(self, X, y):
        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient descent
        for i in range(self.max_iters):
            margins = y * (np.dot(X, self.w) + self.b)
            
            # Compute gradients manually
            dw = np.zeros_like(self.w)
            db = 0

            for j in range(n_samples):
                if margins[j] < 1:  # If the sample is either misclassified or within the margin
                    dw -= y[j] * X[j]
                    db -= y[j]
            
            # Add regularization term to the gradient
            dw += 2 * self.regularization_param * self.w

            # Update weights and bias
            self.w -= self.learning_rate * dw / n_samples
            self.b -= self.learning_rate * db / n_samples

            # === Print Loss Every 100 Iterations ===
            if i % 100 == 0 or i == self.max_iters - 1:
                loss = self.hinge_loss(X, y)
                print(f"Iteration {i}, Hinge Loss: {loss:.4f}")
                
    def predict(self, X):
        # Prediction rule
        return np.sign(np.dot(X, self.w) + self.b)

    def evaluate(self, y_true, y_pred):
        # Evaluate model performance
        acc = accuracy_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return acc, mae, r2

# === Train and Evaluate the ScratchSVM Model ===

svm_model_scratch = ScratchSVM(learning_rate=0.01, regularization_param=0.1, max_iters=1000)
svm_model_scratch.fit(X_train_final, y_train)

# Predictions
y_train_pred_svm_scratch = svm_model_scratch.predict(X_train_final)
y_test_pred_svm_scratch = svm_model_scratch.predict(X_test_final)

# Metrics
train_acc_svm, train_mae_svm, train_r2_svm = svm_model_scratch.evaluate(y_train, y_train_pred_svm_scratch)
test_acc_svm, test_mae_svm, test_r2_svm = svm_model_scratch.evaluate(y_test, y_test_pred_svm_scratch)

print(f"SVM Scratch Train: Accuracy={train_acc_svm:.2f}, MAE={train_mae_svm:.2f}, R²={train_r2_svm:.2f}")
print(f"SVM Scratch Test:  Accuracy={test_acc_svm:.2f}, MAE={test_mae_svm:.2f}, R²={test_r2_svm:.2f}")


import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

class ScratchLogisticRegression:
    def __init__(self, learning_rate=0.01, regularization_param=0.1, max_iters=1000):
        self.learning_rate = learning_rate
        self.regularization_param = regularization_param
        self.max_iters = max_iters
        self.theta = None  # Weights (coefficients)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))  # Sigmoid function

    def loss(self, X, y):
        m = len(y)
        predictions = self.sigmoid(np.dot(X, self.theta))
        # Compute binary cross-entropy loss with regularization
        return -1/m * (np.dot(y, np.log(predictions)) + np.dot((1 - y), np.log(1 - predictions))) + \
               (self.regularization_param / (2 * m)) * np.sum(self.theta[1:] ** 2)

    def fit(self, X, y):
        # Initialize weights
        n_samples, n_features = X.shape
        self.theta = np.zeros(n_features)
        
        # Gradient descent
        for i in range(self.max_iters):
            predictions = self.sigmoid(np.dot(X, self.theta))
            errors = predictions - y

            # Compute gradients
            gradient = np.dot(X.T, errors) / n_samples
            regularization_gradient = (self.regularization_param / n_samples) * np.r_[[0], self.theta[1:]]
            gradient += regularization_gradient

            # Update weights
            self.theta -= self.learning_rate * gradient

            # Optionally, you can print the loss to monitor convergence
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {self.loss(X, y):.4f}")

    def predict(self, X):
        predictions = self.sigmoid(np.dot(X, self.theta))
        return np.round(predictions)  # Return binary predictions (0 or 1)

    def evaluate(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return acc, mae, r2

# === Train and Evaluate the Scratch Logistic Regression Model ===

logreg_model_scratch = ScratchLogisticRegression(learning_rate=0.01, regularization_param=0.1, max_iters=1000)
logreg_model_scratch.fit(X_train_final, y_train)

# Predictions
y_train_pred_logreg_scratch = logreg_model_scratch.predict(X_train_final)
y_test_pred_logreg_scratch = logreg_model_scratch.predict(X_test_final)

# Metrics
train_acc_logreg, train_mae_logreg, train_r2_logreg = logreg_model_scratch.evaluate(y_train, y_train_pred_logreg_scratch)
test_acc_logreg, test_mae_logreg, test_r2_logreg = logreg_model_scratch.evaluate(y_test, y_test_pred_logreg_scratch)

print(f"Logistic Regression Scratch Train: Accuracy={train_acc_logreg:.2f}, MAE={train_mae_logreg:.2f}, R²={train_r2_logreg:.2f}")
print(f"Logistic Regression Scratch Test:  Accuracy={test_acc_logreg:.2f}, MAE={test_mae_logreg:.2f}, R²={test_r2_logreg:.2f}")

def process_user_assessment(assessment_data):
    """
    Process user assessment data and return predictions from our models.
    
    Args:
        assessment_data (dict): Dictionary containing user assessment responses
    Returns:
        dict: Dictionary containing predictions from different models
    """
    # Create a single row dataframe with the same structure as our training data
    user_data = pd.DataFrame([{
        'gender': assessment_data['gender'].lower(),  # Convert to lowercase to match our features
        'age': float(assessment_data['age']),
        'work_study_hours': float(assessment_data['work_study_hours']),
        'academic_pressure': assessment_data['academic_pressure'],
        'work_pressure': assessment_data['work_pressure'],
        'study_satisfaction': assessment_data['study_satisfaction'],
        'job_satisfaction': assessment_data['job_satisfaction'],
        'sleep_duration': assessment_data['sleep_duration'],
        'dietary_habits': assessment_data['dietary_habits'],
        'suicidal_thoughts': assessment_data['suicidal_thoughts'],
        'financial_stress': assessment_data['financial_stress'],
        'family_history': assessment_data['family_history']
    }])

    # Process quantitative variables
    X_user_quanti = np.column_stack([
        (np.where(np.isnan(user_data[col]), np.nanmedian(train_data[col]), user_data[col]) - means[col]) / stds[col]
        for col in quanti_vars
    ])

    # Process categorical variables
    X_user_quali = encoder.transform(user_data[quali_vars])

    # Combine features
    X_user_final = np.hstack((X_user_quanti, X_user_quali))

    # Get predictions from all models
    linear_pred = model.predict(X_user_final)[0]
    svm_pred = svm_model_scratch.predict(X_user_final)[0]
    logreg_pred = logreg_model_scratch.predict(X_user_final)[0]

    # Calculate confidence scores (using sigmoid for linear regression)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    linear_confidence = sigmoid(linear_pred) * 100
    svm_confidence = abs(np.dot(X_user_final[0], svm_model_scratch.w) + svm_model_scratch.b) * 100
    logreg_confidence = logreg_model_scratch.sigmoid(np.dot(X_user_final[0], logreg_model_scratch.theta)) * 100

    # Average the confidence scores
    average_confidence = np.mean([linear_confidence, svm_confidence, logreg_confidence])

    return {
        'linear_prediction': bool(round(linear_pred)),
        'svm_prediction': bool(svm_pred > 0),
        'logreg_prediction': bool(logreg_pred),
        'confidence': average_confidence,
        'individual_scores': {
            'linear': linear_confidence,
            'svm': svm_confidence,
            'logistic': logreg_confidence
        }
    }