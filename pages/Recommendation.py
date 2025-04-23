import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

######Backend######
class SVM(object):
    def __init__(self, learning_rate=0.001, num_iterations=500, lambda_param=0.01):
        self.model_name = 'Support Vector Machine'
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.likelihood_history = []
    
    def predict_score(self, X):
        """
        Produces raw decision values before thresholding
        Inputs:
            - X: Input features
        Outputs:
            - scores: Raw SVM decision values
        """
        try:
            scores = np.dot(X, self.W) + self.b  # Compute raw decision values
        except ValueError as err:
            st.write(str(err))  # Print error messages properly
            return None  # Return None in case of an error

        return scores  # Return the computed scores
    
    def compute_hinge_loss(self, X, Y):
        """
        Compute the hinge loss for SVM using X, Y, and self.W
        Inputs:
            - X: Input features
            - Y: Ground truth labels
        Outputs:
            - loss: Computed hinge loss
        """
        loss = None
        try:
            # Compute the decision scores
            scores = np.dot(X, self.W) + self.b
            
            # Compute hinge loss for each sample
            hinge_losses = np.maximum(0, 1 - Y * scores)
            
            # Compute total hinge loss with regularization
            loss = np.mean(hinge_losses) + (self.lambda_param / 2) * np.sum(self.W ** 2)
        
        except ValueError as err:
            st.write({str(err)})

        return loss

    def update_weights(self):
        """
        Compute SVM derivative using gradient descent and update weights.
        
        Inputs:
            - None
        
        Outputs:
            - self: The trained SVM model
            - self.W: Weight vector updated based on gradient descent
            - self.b: Bias term updated based on gradient descent
            - self.likelihood_history: History of log likelihood
        """
        try:
            # Compute decision scores (raw predictions)
            scores = self.predict_score(self.X)

            # Compute hinge loss margin
            margin = self.Y * scores
            misclassified = margin < 1  # Instances where hinge loss is active

            # Compute gradients
            N = self.X.shape[0]
            dW = np.dot(self.X.T, -self.Y * misclassified) / N
            db = np.sum(-self.Y * misclassified) / N

            # Regularization term (L2 for SVM)
            dW += self.lambda_param * self.W / N  # Regularization term

            # Update weights using gradient descent
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

            # Compute the hinge loss with regularization
            hinge_loss = np.mean(np.maximum(0, 1 - self.Y * scores)) + (self.lambda_param / 2) * np.sum(self.W ** 2)
            
            # The likelihood is typically the negative of the hinge loss
            likelihood = -hinge_loss
            
            # Store likelihood history (use negative log-likelihood if expected)
            self.likelihood_history.append(likelihood)

        except ValueError as err:
            st.write(str(err))

        return self

    def predict(self, X):
        """
        Predicts class labels using the trained SVM model.
        
        Inputs:
            - X: Input features
        
        Outputs:
            - y_pred: List of predicted classes (-1 or +1)
        """
        try:
            # Compute the raw decision scores using the trained model
            scores = self.predict_score(X)

            # Apply thresholding to classify into -1 or +1
            y_pred = np.where(scores >= 0, 1, -1)

        except ValueError as err:
            st.write(str(err))  # Handle errors properly

        return y_pred

    def fit(self, X, Y):
        """
        Train SVM using gradient descent.
        Inputs:
            - X: Input features
            - Y: True class labels (-1 or +1)
        Outputs:
            - self: Trained SVM model
        """
        try:
            num_examples, num_features = X.shape
            self.W = np.zeros(num_features)
            self.b = 0
            self.likelihood_history = []

            # Train SVM using gradient descent
            for _ in range(self.num_iterations):
                scores = self.predict_score(X)
                indicator = (Y * scores) < 1
                
                dW = (-np.dot(X.T, (Y * indicator)) + 2 * self.lambda_param * self.W ) / num_examples
                db = -np.sum(Y * indicator) / num_examples
                
                self.W -= self.learning_rate * dW
                self.b -= self.learning_rate * db
                
                loss = self.compute_hinge_loss(X, Y)
                self.likelihood_history.append(-loss)
            return self.W, self.b, self.likelihood_history

        except ValueError as err:
            st.write({str(err)})

    # Helper Function
    def decision_boundary(self, feat_ids):
        """
        Compute decision boundary values where P(y_i = +1 | x_i) equals P(y_i = -1 | x_i).
        Inputs:
            - feat_ids: Array of feature indices to compute the decision boundary.
        Outputs:
            - boundary: Array containing feature values corresponding to the decision boundary.
        """
        boundary=[]
        try:
            # Extract relevant weight components
            W_ks = self.W[feat_ids]
            # Compute boundary values
            boundary = - self.b / W_ks
            # Handle edge cases
            boundary = np.nan_to_num(boundary, posinf=0, neginf=0)
        except ValueError as err:
            st.write({str(err)})
        return boundary

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, lambda_param=0.01):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_param = lambda_param # For regularizing
        self.loss_history = []
        self.model_name = 'Logistic Regression'

    def sigmoid(self, z): # Sigmoid activation function
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.W = np.zeros(num_features)
        self.b = 0

        for _ in range(self.num_iterations):
            # Forward pass
            linear_model = np.dot(X, self.W) + self.b
            y_pred = self.sigmoid(linear_model)
            
            # Compute gradients
            dW = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)
            
            # Add L2 regularization to weights
            dW += (self.lambda_param / num_samples) * self.W
            
            # Update parameters
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
            
            # Compute loss (binary cross-entropy with regularization)
            loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
            loss += (self.lambda_param / (2 * num_samples)) * np.sum(self.W ** 2)  # L2 regularization
            
            self.loss_history.append(loss)
        
        return self.W, self.b, self.loss_history

    def predict_proba(self, X): # Predict clss probabilities
        linear_model = np.dot(X, self.W) + self.b
        return self.sigmoid(linear_model)

    def predict(self, X):
        probas = self.predict_proba(X)
        return (probas >= 0.5).astype(int)
    
    def fit_multiclass(self, X, y): # Train logistic regression for multi-class classification
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Initialize parameters for each class
        self.W_multi = np.zeros((n_classes, X.shape[1]))
        self.b_multi = np.zeros(n_classes)
        self.loss_history_multi = []
        
        # Train a binary classifier for each class
        for i, c in enumerate(self.classes_):
            # Convert to binary problem
            binary_y = (y == c).astype(int)
            
            # Train binary classifier
            W, b, loss_history = self.fit(X, binary_y)
            
            # Store parameters
            self.W_multi[i] = W
            self.b_multi[i] = b
            
            if i == 0:
                self.loss_history_multi = loss_history
            else:
                # Average loss across all classifiers
                self.loss_history_multi = [(a + b) / 2 for a, b in zip(self.loss_history_multi, loss_history)]
        
        return self
    
    def predict_multiclass(self, X): # Predict class for multi-class classification
        # Compute scores for each class
        scores = np.zeros((X.shape[0], len(self.classes_)))
        
        for i, c in enumerate(self.classes_):
            linear_model = np.dot(X, self.W_multi[i]) + self.b_multi[i]
            scores[:, i] = self.sigmoid(linear_model)
        
        # Return class with highest probability
        return self.classes_[np.argmax(scores, axis=1)]
######Frontend######

st.markdown("# Mental Well-being Prediction Page")
st.markdown("This page is under development!")

df = pd.read_csv("dataset.csv")

# Select target and features
target = 'Mental Health Condition'
X = df.drop(columns=['Mental Health Condition'])
y = df[target]

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)
# st.write(df.columns)

# Encode the target variable {e.g., 0,1,2,3 instead of PTSD, Anxiety, Bipolar, Healthy}
labeler = LabelEncoder()
y = labeler.fit_transform(y)
# To convert back to labels:
# predictions = model.predict(X_test)
# original_categories = label_encoder.inverse_transform(predictions)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ------------------------------
# Train Linear Regression Model
# ------------------------------
model_lr = LogisticRegression(learning_rate=0.01, num_iterations=1000, lambda_param=0.01)
model_lr.fit_multiclass(X_train, y_train)

# Predict on test set
y_pred_lr = model_lr.predict_multiclass(X_test)

# ------------------------------
# Train SVM Model
# ------------------------------
model_svm = SVM(learning_rate=0.001, num_iterations=500, lambda_param=0.01)
W_svm, b_svm, likelihood_history_svm = model_svm.fit(X_train, y_train)

# Predict on test set
y_pred_svm = model_svm.predict(X_test)

# ------------------------------
# Evaluate Models
# ------------------------------
# Logistic regression
accuracy_lr = accuracy_score(y_test, y_pred_lr)
st.write("Logistic Regression accuracy:", accuracy_lr)
st.write("Logistic Regression Classification report:")
st.write(classification_report(y_test, y_pred_lr, zero_division=0))

# SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
st.write("SVM Accuracy:", accuracy_svm)
st.write("SVM Classification Report:")
st.write(classification_report(y_test, y_pred_svm, zero_division=0))

# ------------------------------
# Visualizations of Results
# ------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot confusion matrices
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', ax=ax1)
ax1.set_title('Logistic Regression Confusion Matrix')
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')

cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', ax=ax2)
ax2.set_title('SVM Confusion Matrix')
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')

st.pyplot(fig)

# Plot training curves
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(model_lr.loss_history_multi)
ax1.set_title("Logistic Regression Loss Curve")
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Binary Cross-Entropy Loss")
ax1.grid(True)

ax2.plot(likelihood_history_svm)
ax2.set_title("SVM Likelihood History")
ax2.set_xlabel("Iterations")
ax2.set_ylabel("Negative Hinge Loss")
ax2.grid(True)

st.pyplot(fig2)