# evaluation_metrics.py
"""
Module containing pure functions for calculating classification metrics.
Demonstrates Functional Programming paradigm. Each function takes
true labels and predicted labels and returns a metric score without side effects.
"""
import numpy as np

def calculate_accuracy(y_true, y_pred):
    """Calculates accuracy score. Pure function."""
    y_true = np.array(y_true) # Ensure numpy array for comparison
    y_pred = np.array(y_pred)
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    # Handle division by zero if y_true is empty
    return correct / total if total > 0 else 0.0

def calculate_precision(y_true, y_pred, pos_label=1):
    """Calculates precision score for the positive class. Pure function."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    true_positives = np.sum((y_true == pos_label) & (y_pred == pos_label))
    predicted_positives = np.sum(y_pred == pos_label)
    # Handle division by zero
    return true_positives / predicted_positives if predicted_positives > 0 else 0.0

def calculate_recall(y_true, y_pred, pos_label=1):
    """Calculates recall score for the positive class. Pure function."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    true_positives = np.sum((y_true == pos_label) & (y_pred == pos_label))
    actual_positives = np.sum(y_true == pos_label)
    # Handle division by zero
    return true_positives / actual_positives if actual_positives > 0 else 0.0

def calculate_f1(y_true, y_pred, pos_label=1):
    """Calculates F1-score for the positive class. Pure function."""
    # Reuse the pure precision and recall functions
    precision = calculate_precision(y_true, y_pred, pos_label)
    recall = calculate_recall(y_true, y_pred, pos_label)
    # Handle division by zero
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0