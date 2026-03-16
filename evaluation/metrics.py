from typing import Any, Dict, List

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_intent_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_intent_report(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


def compute_confusion(
    y_true: List[str], y_pred: List[str], labels: List[str]
):
    return confusion_matrix(y_true, y_pred, labels=labels)

