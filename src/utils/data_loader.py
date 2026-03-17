import pandas as pd
import os


def load_dataset(path="data/raw_data/full_nlu_dataset_200.csv"):
    """
    Load NLU dataset from CSV file
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    try:
        data = pd.read_csv(path)
        return data

    except Exception as e:
        print("Error loading dataset:", e)
        return None