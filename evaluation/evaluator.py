import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .metrics import compute_confusion, compute_intent_metrics, compute_intent_report
from src.nlu_pipeline import run_pipeline
from utils.config_loader import load_config


BASE_DIR = Path(__file__).resolve().parent.parent


def _load_test_data() -> List[Dict]:
    cfg = load_config()
    data_path = BASE_DIR / cfg.test_data_path
    data = json.loads(data_path.read_text(encoding="utf-8"))
    return data.get("test_examples", [])


def evaluate_intents(
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Tuple[Dict[str, float], List[List[int]], List[str], Dict, List[Dict]]:
    """
    Run all test examples through the NLU pipeline and compute metrics.

    Returns:
      metrics: dict with accuracy, precision, recall, f1, exact_match
      confusion: 2D list confusion matrix
      labels: label order used in the confusion matrix
      report: sklearn classification_report output_dict
      examples: list of dict rows with per-example predictions/ground truth
    """
    test_examples = _load_test_data()

    y_true: List[str] = []
    y_pred: List[str] = []
    examples: List[Dict] = []

    # Track how often the model gets BOTH the intent and
    # the full entity dictionary exactly right. This helps
    # avoid overly optimistic metrics when the intent is
    # easy but entity extraction is not perfect.
    exact_matches = 0

    total = len(test_examples)
    for idx, ex in enumerate(test_examples, start=1):
        true_intent = ex.get("intent", "")
        true_entities = ex.get("entities", {}) or {}

        y_true.append(true_intent)

        result = run_pipeline(ex["text"])
        pred_intent = result.get("intent", "")
        pred_entities = result.get("entities", {}) or {}

        y_pred.append(pred_intent)

        exact_match = (
            pred_intent == true_intent
            and isinstance(pred_entities, dict)
            and pred_entities == true_entities
        )
        if exact_match:
            exact_matches += 1

        examples.append(
            {
                "text": ex.get("text", ""),
                "true_intent": true_intent,
                "pred_intent": pred_intent,
                "true_entities": true_entities,
                "pred_entities": pred_entities,
                "exact_match": exact_match,
            }
        )

        if progress_cb:
            progress_cb(idx, total)

    labels = sorted(set(y_true))
    metrics = compute_intent_metrics(y_true, y_pred)

    # Add an additional, stricter metric that captures when the
    # entire prediction (intent + entities) matches the ground truth.
    if total > 0:
        metrics["exact_match"] = float(exact_matches) / float(total)
    else:
        metrics["exact_match"] = 0.0
    cm = compute_confusion(y_true, y_pred, labels)
    report = compute_intent_report(y_true, y_pred)

    # Convert numpy array to plain list for easier consumption (e.g. Streamlit)
    cm_list = cm.tolist()

    return metrics, cm_list, labels, report, examples

