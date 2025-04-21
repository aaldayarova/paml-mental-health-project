import streamlit as st
import numpy as np
import pandas as pd
from helper_functions import fetch_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import random
import matplotlib.pyplot as plt
import os

random.seed(10)

######Backend######
class SVM(object):
    def __init__(self, learning_rate=0.001, num_iterations=500, lambda_param=0.01):
        self.model_name = 'Support Vector Machine'
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.binary_classifiers = {}
        self.classes = None 
        self.feature_names = None 

    def train_binary_classifier(self, X, y, target_class):
        """
        Train a binary classifier for one class vs all others
        """
        # Create binary labels: 1 for target class, -1 for all other classes
        binary_y = np.where(y == target_class, 1, -1)
        
        # Initialize weights and bias
        num_features = X.shape[1]
        W = np.zeros(num_features)
        b = 0
        likelihood_history = []
        
        # Training loop using gradient descent
        for _ in range(self.num_iterations):
            # Compute scores
            scores = np.dot(X, W) + b
            
            # Compute hinge loss for SVM
            indicator = (binary_y * scores) < 1
            
            # Compute gradients
            dW = (-np.dot(X.T, (binary_y * indicator)) + 2 * self.lambda_param * W) / X.shape[0]
            db = -np.sum(binary_y * indicator) / X.shape[0]
            
            # Update weights
            W -= self.learning_rate * dW
            b -= self.learning_rate * db
            
            # Compute loss for tracking
            hinge_losses = np.maximum(0, 1 - binary_y * scores)
            loss = np.mean(hinge_losses) + (self.lambda_param / 2) * np.sum(W ** 2)
            likelihood_history.append(-loss)
        
        # Return trained model parameters
        return {'W': W, 'b': b, 'likelihood_history': likelihood_history}
    
    def fit(self, X, y):
        """
        Train a multi-class SVM using One-vs-Rest approach
        """
        self.classes = np.unique(y)
        self.feature_names = None if isinstance(X, np.ndarray) else X.columns.tolist()
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Train a binary classifier for each class
        for cls in self.classes:
            self.binary_classifiers[cls] = self.train_binary_classifier(X, y, cls)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # For each sample, compute score for each class and select the highest scoring class
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, len(self.classes)))
        
        # Compute scores for each binary classifier
        for i, cls in enumerate(self.classes):
            classifier = self.binary_classifiers[cls]
            class_scores = np.dot(X, classifier['W']) + classifier['b']
            scores[:, i] = class_scores
        
        # Return class with highest score for each sample
        return self.classes[np.argmax(scores, axis=1)]
    
    def predict_proba(self, X):
        """
        Return a matrix of class probability estimates
        (very rough approximation using softmax on raw scores)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, len(self.classes)))
        
        # Compute scores for each binary classifier
        for i, cls in enumerate(self.classes):
            classifier = self.binary_classifiers[cls]
            class_scores = np.dot(X, classifier['W']) + classifier['b']
            scores[:, i] = class_scores
        
        # Apply softmax to get probabilities
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
# Function to save multi-class model
def save_multiclass_model(model, model_path="./multiclass_svm_model.npz"):
    """
    Saves the multi-class SVM model weights, biases, and classes for later use
    """
    try:
        # Prepare data for saving
        classes = model.classes
        feature_names = model.feature_names if model.feature_names else []
        
        # Extract weights and biases for each classifier
        W_dict = {}
        b_dict = {}
        for cls in classes:
            W_dict[str(cls)] = model.binary_classifiers[cls]['W']
            b_dict[str(cls)] = model.binary_classifiers[cls]['b']
        
        # Save model parameters
        np.savez(
            model_path,
            classes=classes,
            features=feature_names,
            W_dict=W_dict,
            b_dict=b_dict
        )
        return True
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")
        return False

######Frontend######
st.markdown("# Mental Health Multi-Class Prediction")
st.markdown("This page trains a model to predict specific mental health conditions based on lifestyle factors.")

# Set SVM hyperparameters
LEARNING_RATE = 0.001
NUM_ITERATIONS = 1000
LAMBDA_PARAM = 0.01

df = None
df = fetch_dataset()

if df is not None:
    # Display the dataset
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    # Define the label column internally
    label_col = "Mental Health Condition"
    
    # Display the selected label column
    st.write(f"Target variable for prediction: '{label_col}'")
    
    if label_col in df.columns:
        # Extract features (X) and label (y)
        X = df.drop(columns=[label_col])
        y = df[label_col]
        
        # Store feature names for later use
        feature_names = X.columns.tolist()
        
        # Display information about dataset split
        st.write(f"Features: {X.shape[1]} columns, {X.shape[0]} records")
        unique_classes = y.unique()
        st.write(f"Target classes ({len(unique_classes)}): {', '.join(map(str, unique_classes))}")
        
        # Create a label encoder to convert class labels to numeric values if needed
        label_encoder = None
        if not all(isinstance(val, (int, float)) for val in y.unique()):
            st.info("Converting text class labels to numeric format for model training.")
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            # Show mapping
            class_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
            st.write(f"Class mapping: {class_mapping}")
        
        # Splitting data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Save train and test split to st.session_state
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        
        # Model training section
        st.subheader("Model Training")
        
        # Information about used parameters
        st.info(f"Training Multi-Class SVM with: Learning Rate = {LEARNING_RATE}, Iterations = {NUM_ITERATIONS}, Lambda = {LAMBDA_PARAM}")
        
        # Automatic training without user interaction
        model_file_path = "./multiclass_svm_model.npz"
        model_exists = os.path.exists(model_file_path)
        
        # Add a button to force retraining if needed
        force_retrain = st.button("Retrain Model")
        
        if not model_exists or force_retrain or 'multiclass_svm_model' not in st.session_state:
            with st.spinner("Training Multi-Class SVM model..."):
                try:
                    # Initialize and train the multi-class SVM model
                    svm_model = SVM(learning_rate=LEARNING_RATE, num_iterations=NUM_ITERATIONS, lambda_param=LAMBDA_PARAM)
                    svm_model.fit(X_train, y_train)
                    
                    # Store the trained model in session state
                    st.session_state['multiclass_svm_model'] = svm_model
                    
                    # Save model to disk for later use
                    if save_multiclass_model(svm_model, model_file_path):
                        st.success(f"Multi-Class SVM Model trained and saved successfully to {model_file_path}")
                    
                    # Make predictions on test set
                    y_pred = svm_model.predict(X_test)
                    
                    # Evaluate model
                    st.subheader("Model Evaluation")
                    
                    # Compute accuracy
                    accuracy = np.mean(y_pred == y_test)
                    st.metric("Accuracy", f"{accuracy:.4f}")
                    
                    # Convert numeric predictions back to original labels if needed
                    if label_encoder:
                        y_test_original = label_encoder.inverse_transform(y_test)
                        y_pred_original = label_encoder.inverse_transform(y_pred)
                        class_labels = label_encoder.classes_
                    else:
                        y_test_original = y_test
                        y_pred_original = y_pred
                        class_labels = np.unique(y)
                    
                    # Display classification report
                    st.subheader("Classification Report")
                    report = classification_report(y_test_original, y_pred_original, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df)
                    
                    # Display confusion matrix
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test_original, y_pred_original)
                    
                    # Create a figure and plot the confusion matrix
                    plt.figure(figsize=(10, 8))
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
                    disp.plot(cmap=plt.cm.Blues)
                    plt.title("Confusion Matrix")
                    plt.xticks(rotation=45, ha="right")
                    st.pyplot(plt)
                    
                    # Display per-class accuracy
                    st.subheader("Per-Class Performance")
                    class_metrics = {}
                    for cls in np.unique(y_test_original):
                        class_mask = (y_test_original == cls)
                        class_correct = np.sum((y_pred_original == cls) & class_mask)
                        class_total = np.sum(class_mask)
                        class_accuracy = class_correct / class_total if class_total > 0 else 0
                        class_metrics[cls] = {
                            'accuracy': class_accuracy,
                            'samples': class_total
                        }
                    
                    # Display class metrics as a dataframe
                    class_metrics_df = pd.DataFrame.from_dict(class_metrics, orient='index')
                    class_metrics_df['accuracy'] = class_metrics_df['accuracy'].map('{:.2%}'.format)
                    st.dataframe(class_metrics_df)
                    
                    # Add information on how to use the model
                    st.subheader("Using the Trained Model")
                    st.write("""
                    The Multi-Class SVM model has been trained and saved. This model can now be loaded on other pages 
                    to make predictions for new mental health conditions. You can use it by loading the saved model file and 
                    passing new feature values to the prediction function.
                    """)
                    
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")
                    st.exception(e)
        
        # If model already exists and no retraining requested
        elif 'multiclass_svm_model' in st.session_state:
            st.success(f"Multi-Class SVM Model already trained. Model saved at {model_file_path}")
            
            # Make predictions with existing model
            svm_model = st.session_state['multiclass_svm_model']
            y_pred = svm_model.predict(X_test)
            
            # Compute accuracy
            accuracy = np.mean(y_pred == y_test)
            st.metric("Accuracy", f"{accuracy:.4f}")
            
            # Convert numeric predictions back to original labels if needed
            if label_encoder:
                y_test_original = label_encoder.inverse_transform(y_test)
                y_pred_original = label_encoder.inverse_transform(y_pred)
                class_labels = label_encoder.classes_
            else:
                y_test_original = y_test
                y_pred_original = y_pred
                class_labels = np.unique(y)
            
            # Display confusion matrix for existing model
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test_original, y_pred_original)
            
            plt.figure(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
            disp.plot(cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(plt)
            
            # Display classification report
            st.subheader("Classification Report")
            report = classification_report(y_test_original, y_pred_original, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
    else:
        st.error(f"Label column '{label_col}' not found in the dataset. Please check the column name.")
else:
    st.error("Failed to load dataset. Please check if the file 'mental_health_lifestyle_ds.csv' exists.")