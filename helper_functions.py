# This is the homebase for any helper functions we'll need for this project.
import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.metrics import confusion_matrix

def fetch_dataset():
    """
    This function renders the file uploader that fetches the dataset either from local machine

    Input:
        - page: the string represents which page the uploader will be rendered on
    Output: None
    """
    # Check stored data
    df = None
    dataset_filename = './mental_health_lifestyle_ds.csv'
    if 'data' in st.session_state:
        df = st.session_state['data']
    else:
        if os.path.exists(dataset_filename):
            df = pd.read_csv(dataset_filename)
    if df is not None:
        st.session_state['data'] = df
    return df

def compute_accuracy(prediction_labels, true_labels):    
    """
    Compute classification accuracy
    Input
        - prediction_labels (numpy): predicted product sentiment
        - true_labels (numpy): true product sentiment
    Output
        - accuracy (float): accuracy percentage (0-100%)
    """
    accuracy=None
    try:
        correct_preds = np.sum(prediction_labels == true_labels)
        total_preds = len(true_labels)
        accuracy = correct_preds / total_preds  # return raw float, NOT percentage
    except ValueError as err:
        st.write({str(err)})
    return accuracy

def compute_precison_recall(prediction_labels, true_labels, print_summary=False):
    """
    Compute precision and recall 
    Input
        - prediction_labels (numpy): predicted product sentiment
        - true_labels (numpy): true product sentiment
    Output
        - precision (float): precision score = TP/TP+FP
        - recall (float): recall score = TP/TP+FN
    """
    precision=None
    recall=None
    try:
        # Auto-handle whatever labels are present (0/1 or -1/1)
        cm = confusion_matrix(true_labels, prediction_labels)

        # Case 1: binary classification → shape should be 2x2
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # fallback to zero values if not binary
            tn = fp = fn = tp = 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if print_summary:
            st.write(f'Precision: {precision:.4f}')
            st.write(f'Recall: {recall:.4f}')

    except ValueError as err:
        st.write(str(err))

    return precision, recall

def compute_eval_metrics(X, y_true, model, metrics, print_summary=False):
    """
    This function computes one or more metrics (precision, recall, accuracy) using the model

    Inputs:
        - X: pandas dataframe with training features
        - y_true: pandas dataframe with true targets
        - model: the model to evaluate
        - metrics: the metrics to evaluate performance (string); 'precision', 'recall', 'accuracy'
    Outputs:
        - metric_dict: a dictionary contains the computed metrics of the selected model, with the following structure:
            - {metric1: value1, metric2: value2, ...}
    """
    metric_dict = {'precision': -1,
                   'recall': -1,
                   'accuracy': -1}
    try:
        # Predict the product sentiment using the input model and data X
        y_pred = model.predict(X)

        # Compute the evaluation metrics in 'metrics = ['precision', 'recall', 'accuracy']' using the predicted sentiment
        precision, recall = compute_precison_recall(y_pred, y_true.to_numpy())
        accuracy = compute_accuracy(y_pred, y_true.to_numpy())

        if 'precision' in metrics:
            metric_dict['precision'] = precision
        if 'recall' in metrics:
            metric_dict['recall'] = recall
        if 'accuracy' in metrics:
            metric_dict['accuracy'] = accuracy
    except ValueError as err:
        st.write({str(err)})
    return metric_dict