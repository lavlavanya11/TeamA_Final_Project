import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from src.utils.logger import setup_logger

logger = setup_logger("EVALUATOR")

class NLUEvaluator:
    """
    Evaluates NLU model predictions against ground truth labels.
    Calculates Accuracy, Precision, Recall, F1 and Confusion Matrix.
    Also provides per-intent breakdown table (extra vs original).
    """

    def evaluate_with_results(self, df: pd.DataFrame):
        """
        Takes a DataFrame with true and predicted intents and returns metrics.

        Args:
            df: DataFrame with columns — text, intent, predicted_intent

        Returns:
            metrics: dict with accuracy, precision, recall, f1
            confusion_df: confusion matrix as DataFrame
            per_intent_df: per-intent breakdown DataFrame
        """
        y_true = df["intent"].tolist()
        y_pred = df["predicted_intent"].tolist()

        # Overall metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "f1": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
        }

        logger.info(f"Evaluation metrics: {metrics}")

        # Confusion matrix
        labels = sorted(list(set(y_true + y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        confusion_df = pd.DataFrame(cm, index=labels, columns=labels)

        # Per-intent breakdown (extra feature vs original)
        per_intent_rows = []
        for intent in sorted(set(y_true)):
            intent_true = [1 if t == intent else 0 for t in y_true]
            intent_pred = [1 if p == intent else 0 for p in y_pred]

            per_intent_rows.append({
                "Intent": intent,
                "Precision": round(precision_score(intent_true, intent_pred, zero_division=0), 2),
                "Recall": round(recall_score(intent_true, intent_pred, zero_division=0), 2),
                "F1 Score": round(f1_score(intent_true, intent_pred, zero_division=0), 2),
                "Correct": sum(1 for t, p in zip(y_true, y_pred) if t == intent and p == intent),
                "Total": sum(1 for t in y_true if t == intent)
            })

        per_intent_df = pd.DataFrame(per_intent_rows)
        logger.info("Per-intent breakdown generated")

        return metrics, confusion_df, per_intent_df