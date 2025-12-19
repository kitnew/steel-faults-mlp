# mlp_model.py

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

def get_mlp_classifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-4,               # L2 regularization
    batch_size=32,
    learning_rate_init=1e-3,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=15,
    random_state=5296
):
    """
    Fixed MLP architecture for all experiments.
    DO NOT change hyperparameters between data splits.
    """
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        random_state=random_state
    )

def train_mlp(model, X_train, y_train, X_test, y_test, split_name):
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
        "training_loss": model.loss_curve_,
        "validation_score": model.validation_scores_
        if hasattr(model, "validation_scores_") else None
    }

    best_epoch = (
        np.argmax(model.validation_scores_) + 1
        if hasattr(model, "validation_scores_")
        else len(model.loss_curve_)
    )

    results = {
        "split_name": split_name,
        "n_epochs_trained": model.n_iter_,
        "best_epoch": best_epoch,
        "metrics": metrics,
        "learning_curves": learning_curves,
        "model_params": model.get_params()
    }

    return results