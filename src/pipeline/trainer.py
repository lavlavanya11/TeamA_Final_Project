from src.utils.data_loader import load_dataset
from src.utils.logger import setup_logger

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import json

logger = setup_logger()

def train():

    dataset = load_dataset()

    X = dataset["utterance"]
    y = dataset["intent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer()

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression()

    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    # Save metrics
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }

    os.makedirs("models", exist_ok=True)

    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("models/confusion_matrix.png")
    plt.close()

    pickle.dump(model, open("models/intent_model.pkl", "wb"))
    pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

    logger.info("Training completed")

    print("Training completed successfully")