from evaluation.metrics import compute_intent_metrics


def test_compute_intent_metrics_basic():
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    m = compute_intent_metrics(y_true, y_pred)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["f1"] <= 1.0

