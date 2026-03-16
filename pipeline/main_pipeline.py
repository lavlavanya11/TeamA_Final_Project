import json

from evaluation.evaluator import evaluate_intents


def main() -> None:
    metrics, cm, labels, report, _examples = evaluate_intents()

    print("=== BotTrainer Evaluation ===")
    print("Metrics:")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"  - {k}: {v:.4f}")
        else:
            print(f"  - {k}: {v}")

    print("\nLabels:")
    print(labels)

    print("\nConfusion matrix:")
    print(json.dumps(cm, indent=2))

    print("\nClassification report:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

