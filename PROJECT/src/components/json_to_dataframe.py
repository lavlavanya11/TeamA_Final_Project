import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("JSON_TO_DATAFRAME")

def flatten_intents_json(intents_data: dict) -> pd.DataFrame:
    """
    Flattens intents.json into a pandas DataFrame.
    Each row contains a training example and its corresponding intent label.
    
    Output columns:
        - text: user utterance
        - true_intent: ground truth intent label
    """
    rows = []

    for intent in intents_data["intents"]:
        intent_name = intent["name"]
        examples = intent["examples"]

        for example in examples:
            rows.append({
                "text": example,
                "true_intent": intent_name
            })

    df = pd.DataFrame(rows, columns=["text", "true_intent"])

    logger.info(f"Flattened dataset created — {len(df)} rows, {df['true_intent'].nunique()} intents")

    return df