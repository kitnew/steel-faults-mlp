# baseline_model.py

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

def get_baseline_model(
    loss='log_loss',
    n_estimators=200,
    learning_rate=0.1,
    max_depth=10,
    random_state=5296
):
    """
    Baseline model from Assignment 1.
    Must be evaluated on the SAME test split as MLP.
    """
    return GradientBoostingClassifier(
        loss=loss,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )

def train_baseline(model, X_train, y_train, X_test, y_test, split_name):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, average="weighted"
        ),
        "recall": recall_score(
            y_test, y_pred, average="weighted"
        ),
        "f1": f1_score(
            y_test, y_pred, average="weighted"
        ),
    }

    learning_curves = {
        "training_loss": model.loss_curve_ if hasattr(model, "loss_curve_") else None,
        "validation_score": model.validation_scores_
        if hasattr(model, "validation_scores_") else None
    }

    best_epoch = (
        np.argmax(model.validation_scores_) + 1
        if hasattr(model, "validation_scores_")
        else None
    )

    results = {
        "split_name": split_name,
        "best_epoch": best_epoch,
        "metrics": metrics,
        "learning_curves": learning_curves,
        "model_params": model.get_params()
    }

    return results