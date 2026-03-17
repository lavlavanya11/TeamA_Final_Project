from src.pipeline.predict import predict
from src.utils.data_loader import load_dataset


def evaluate():

    data = load_dataset()

    correct = 0
    total = len(data)

    for _, row in data.iterrows():

        text = row["utterance"]
        true_intent = row["intent"]

        predicted_intent = predict(text)

        if predicted_intent.lower() == true_intent.lower():
            correct += 1

    accuracy = correct / total

    print("Total samples:", total)
    print("Correct predictions:", correct)
    print("Model Accuracy:", round(accuracy * 100, 2), "%")

    return accuracy