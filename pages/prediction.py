import streamlit as st
import numpy as np

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
        return self

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


######Frontend######

st.markdown("# Mental Well-being Prediction Page")
st.markdown("This page is under development!")

