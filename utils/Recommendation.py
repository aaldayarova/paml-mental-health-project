# utils/Recommendation.py

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

#ScratchLogisticRegression class
class ScratchLogisticRegression:
    def __init__(self, learning_rate=0.01, regularization_param=0.1, max_iters=1000):
        self.learning_rate = learning_rate
        self.regularization_param = regularization_param
        self.max_iters = max_iters
        self.theta = None

    def sigmoid(self, z):
        # Clip z to avoid overflow in exp
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

    def loss(self, X, y):
        if self.theta is None:
            print("Warning: Loss calculated on uninitialized model.")
            return float('inf')
        m = len(y)
        predictions = self.sigmoid(np.dot(X, self.theta))
        # Clip predictions to avoid log(0)
        epsilon = 1e-5
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        cost = -1/m * (np.dot(y, np.log(predictions)) + np.dot((1 - y), np.log(1 - predictions)))
        reg_cost = 0
        if len(self.theta) > 1: 
             reg_cost = (self.regularization_param / (2 * m)) * np.sum(self.theta[1:] ** 2)
        return cost + reg_cost

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights
        self.theta = np.zeros(n_features)

        for i in range(self.max_iters):
            predictions = self.sigmoid(np.dot(X, self.theta))
            errors = predictions - y
            gradient = np.dot(X.T, errors) / n_samples

            reg_gradient_term = np.zeros_like(self.theta)
            if len(self.theta) > 1: # Regularize weights, not bias
                reg_gradient_term[1:] = (self.regularization_param / n_samples) * self.theta[1:]
            gradient += reg_gradient_term

            self.theta -= self.learning_rate * gradient

    def predict(self, X):
        if self.theta is None:
            raise ValueError("Model has not been trained or loaded properly (theta is None).")
        probabilities = self.sigmoid(np.dot(X, self.theta))
        return np.round(probabilities) # Return binary predictions

    def predict_proba(self, X):
        if self.theta is None:
            raise ValueError("Model has not been trained or loaded properly (theta is None).")
        # Returns probability of class 1
        return self.sigmoid(np.dot(X, self.theta))

# --- Paths to your saved files ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

MODEL_PATH = BASE_DIR / 'logistic_regression_mental_health_model.pkl'
PARAMS_PATH = BASE_DIR / 'preprocessing_params_lr.pkl'
ENCODER_PATH = BASE_DIR / 'one_hot_encoder_lr.pkl'

# Global variables to hold loaded objects and error status
model, one_hot_encoder, means, stds, quanti_vars, quali_vars = [None] * 6
_loading_error_message = None 

# Custom Unpickler 
class _CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == 'ScratchLogisticRegression':
            return ScratchLogisticRegression
        return super().find_class(module, name)

#Load model and preprocessors
try:
    # Load the logistic regression model using custom unpickler
    with open(MODEL_PATH, 'rb') as f:
        model = _CustomUnpickler(f).load()

    # Load other files
    with open(PARAMS_PATH, 'rb') as f:
        preprocessing_params = pickle.load(f)
    means = preprocessing_params['means']
    stds = preprocessing_params['stds']
    quanti_vars = preprocessing_params['quanti_vars']
    quali_vars = preprocessing_params['quali_vars']

    with open(ENCODER_PATH, 'rb') as f:
        one_hot_encoder = pickle.load(f) 

except FileNotFoundError as e:
    _loading_error_message = (f"Error: A required model or preprocessing file was not found.\n"
                             f"Missing file: {e.filename}\n"
                             f"Please ensure all .pkl files are in the directory: {BASE_DIR}")
    print(_loading_error_message)
except AttributeError as e:
    _loading_error_message = (f"Error during unpickling (AttributeError): {e}\n"
                             f"This often means a custom class definition (like ScratchLogisticRegression) "
                             f"was not found or its structure changed. "
                             f"Current module tried: {getattr(e, 'obj', '<unknown>')} looking for {getattr(e, 'name', '<unknown>')}")
    print(_loading_error_message)
except Exception as e:
    _loading_error_message = f"An unexpected error occurred while loading model files: {type(e).__name__} - {e}"
    print(_loading_error_message)


def process_user_assessment(assessment_data):
    global model, one_hot_encoder, means, stds, quanti_vars, quali_vars, _loading_error_message

    if _loading_error_message:
        return {
            'logreg_prediction': None,
            'individual_confidence': {'logistic': "Error: Model/data files failed to load."},
            'error': _loading_error_message
        }

    if not all([model, one_hot_encoder, means, stds, quanti_vars, quali_vars]):
        err_msg = "Critical Error: Model or preprocessors are not available. Check logs."
        return {
            'logreg_prediction': None,
            'individual_confidence': {'logistic': err_msg},
            'error': err_msg
        }

    input_features = {}
    try:
        #Data extraction and type conversion from assessment_data
        input_features['Age'] = float(assessment_data['age'])
        input_features['WSHours'] = float(assessment_data['work_study_hours'])
        input_features['Gender'] = assessment_data['gender']
        input_features['SleepDuration'] = assessment_data['sleep_duration']
        input_features['DietaryHabits'] = assessment_data['dietary_habits']
        input_features['SuicidalThoughts'] = assessment_data['suicidal_thoughts']
        input_features['FamilyHistory'] = assessment_data['family_history']
        input_features['AcademicPressure'] = float(assessment_data['academic_pressure'])
        input_features['WorkPressure'] = float(assessment_data['work_pressure'])
        input_features['StudySatisfaction'] = float(assessment_data['study_satisfaction'])
        input_features['JobSatisfaction'] = float(assessment_data['job_satisfaction'])
        input_features['FinancialStress'] = float(assessment_data['financial_stress'])

        # Preprocessing 
        # Quantitative features
        X_quanti_list = [input_features[col] for col in quanti_vars]
        X_quanti_np = np.array(X_quanti_list).reshape(1, -1)
        means_array = np.array([means[col] for col in quanti_vars])
        stds_array = np.array([stds[col] for col in quanti_vars])
        X_quanti_scaled = (X_quanti_np - means_array) / stds_array

        # Qualitative features
        quali_values_ordered = [input_features[col] for col in quali_vars]
        df_quali_input = pd.DataFrame([quali_values_ordered], columns=quali_vars)
        X_quali_encoded = one_hot_encoder.transform(df_quali_input)

        # Combine features
        X_final_input = np.hstack((X_quanti_scaled, X_quali_encoded))

        # --- Prediction ---
        binary_prediction = model.predict(X_final_input)[0]
        probability_class_1 = model.predict_proba(X_final_input)[0] # Probability of depression

        return {
            'logreg_prediction': int(binary_prediction),
            'individual_confidence': {
                'logistic': probability_class_1 * 100, 
            },
            'error': None
        }
    except KeyError as e:
        error_msg = f"Error: Missing expected data from form: {e}. Please ensure all fields are filled."
        print(error_msg)
        return {
            'logreg_prediction': None,
            'individual_confidence': {'logistic': "Error: Incomplete form data."},
            'error': error_msg
        }
    except Exception as e:
        error_msg = f"Error during processing assessment: {type(e).__name__} - {e}"
        print(error_msg)
        return {
            'logreg_prediction': None,
            'individual_confidence': {'logistic': "Error: Processing failed."},
            'error': error_msg
        }