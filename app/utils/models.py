"""
models.py
---------
Machine Learning Model Definitions.

This module defines three classifiers for predicting clinical trial
treatment success. Each model is wrapped in a simple interface so the
Streamlit app can train and evaluate any of them interchangeably.

Models:
    1. Logistic Regression  — fast, interpretable baseline
    2. Random Forest        — ensemble, handles non-linearity well
    3. XGBoost              — gradient boosting, often best performance

Author: Banoth Rajesham
"""

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.metrics        import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import numpy as np
import xgboost as xgb


def get_logistic_regression(C: float = 1.0, max_iter: int = 500) -> LogisticRegression:
    """
    Returns a Logistic Regression classifier.
    
    Parameters
    ----------
    C : float
        Inverse of regularization strength. Smaller = stronger regularization.
    max_iter : int
        Maximum number of iterations for solver convergence.
    """
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        class_weight="balanced",  # handles class imbalance
        random_state=42,
    )


def get_random_forest(n_estimators: int = 100, max_depth: int = None) -> RandomForestClassifier:
    """
    Returns a Random Forest classifier.
    
    Parameters
    ----------
    n_estimators : int
        Number of decision trees in the forest.
    max_depth : int or None
        Maximum depth of each tree. None = grow until pure.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=-1,          # use all CPU cores
        random_state=42,
    )


def get_xgboost(learning_rate: float = 0.1, n_estimators: int = 100, max_depth: int = 4):
    """
    Returns an XGBoost classifier.
    
    Parameters
    ----------
    learning_rate : float
        Step size shrinkage to prevent overfitting.
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth per round.
    """
    return xgb.XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )


def train_model(model, X_train, y_train):
    """
    Fit a model on training data.
    
    Parameters
    ----------
    model : sklearn-compatible estimator
    X_train : array-like, training features
    y_train : array-like, training labels
    
    Returns
    -------
    Fitted model
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate a trained model and return all important metrics.
    
    Returns a dictionary so we can easily log to MLflow or display
    in Streamlit without extra formatting code.
    
    Parameters
    ----------
    model : fitted sklearn/xgb model
    X_test : test features
    y_test : true labels
    
    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc, confusion_matrix, y_pred, y_prob
    """
    y_pred = model.predict(X_test)

    # Some models need predict_proba for ROC-AUC
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    except AttributeError:
        y_prob  = y_pred
        roc_auc = None

    return {
        "accuracy"        : round(accuracy_score(y_test, y_pred), 4),
        "precision"       : round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall"          : round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1"              : round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc"         : round(roc_auc, 4) if roc_auc else "N/A",
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_pred"          : y_pred,
        "y_prob"          : y_prob,
    }


def get_feature_importance(model, feature_names: list) -> dict:
    """
    Extract feature importance scores from the model (if available).
    
    Works for Random Forest and XGBoost. Returns empty dict for
    Logistic Regression (use coefficients instead).
    
    Parameters
    ----------
    model : fitted model
    feature_names : list of str
    
    Returns
    -------
    dict : {feature_name: importance_score}
    """
    importance_dict = {}

    if hasattr(model, "feature_importances_"):
        # Random Forest / XGBoost
        scores = model.feature_importances_
        importance_dict = dict(zip(feature_names, scores))

    elif hasattr(model, "coef_"):
        # Logistic Regression — use absolute coefficients
        scores = np.abs(model.coef_[0])
        importance_dict = dict(zip(feature_names, scores))

    # Sort by importance descending
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
