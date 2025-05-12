# Mental Health Logistic Regression Predictor

This module provides a custom implementation of logistic regression to predict the likelihood of depression in users based on assessment data. It includes a from-scratch logistic regression model, model loading, preprocessing, and inference functionality.

## Features

- **Scratch-built Logistic Regression** implementation with L2 regularization.
- **Custom unpickler** to load saved models using local class definitions.
- **Preprocessing pipeline** including standardization for quantitative features and one-hot encoding for categorical features.
- **Prediction endpoint** that returns binary prediction and probability of mental health concern.

## File Structure

- `Recommendation.py`: Main module containing the `ScratchLogisticRegression` class and data processing logic.
- `logistic_regression_mental_health_model.pkl`: Serialized trained model.
- `preprocessing_params_lr.pkl`: Preprocessing statistics (means, stds, feature lists).
- `one_hot_encoder_lr.pkl`: Fitted one-hot encoder for categorical variables.

## Dependencies

- `numpy`
- `pandas`
- `pickle`
- `pathlib`

Install required packages using:

```bash
pip install numpy pandas
```
Run the homepage of our application using:

streamlit run home.py 

Run the application using:

streamlit run streamlit_app.py

Error Handling

The script includes detailed exception handling for:

    Missing model or encoder files

    Incorrect or incomplete input data

    Unpickling or class loading issues

Custom Model

The ScratchLogisticRegression class includes:

    Sigmoid activation

    Gradient descent with L2 regularization

    Custom prediction and probability estimation methods

Notes

    Ensure all required .pkl files are placed in the correct directory (../ relative to this script).

    The model expects specific quantitative and qualitative feature names to match training data.
