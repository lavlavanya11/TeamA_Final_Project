import pandas as pd
from src.components.json_loader import IntentJSONLoader
from src.components.json_to_dataframe import flatten_intents_json
from src.components.gemini_nlu import GeminiNLU
from src.components.evaluator import NLUEvaluator
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("MAIN_PIPELINE")

def run_evaluation_pipeline():
    """
    End-to-end evaluation pipeline:
    1. Load config
    2. Load intents.json
    3. Flatten to DataFrame
    4. Pick 1 sample per intent
    5. Run each through GeminiNLU
    6. Evaluate predictions
    7. Return metrics, confusion matrix, per-intent breakdown
    """

    # Step 1 — Load config
    logger.info("Loading config...")
    config = load_config()

    # Step 2 — Load intents
    logger.info("Loading intents...")
    loader = IntentJSONLoader(config["paths"]["intents_path"])
    intents_data = loader.load()

    # Step 3 — Flatten to DataFrame
    logger.info("Flattening intents to DataFrame...")
    df = flatten_intents_json(intents_data)

    # Step 4 — Pick 1 sample per intent
    logger.info("Selecting 1 sample per intent for evaluation...")
    df_eval = (
        df.groupby("true_intent", group_keys=False)
          .head(1)
          .reset_index(drop=True)
    )

    # Step 5 — Initialize GeminiNLU
    logger.info("Initializing GeminiNLU...")
    nlu = GeminiNLU(
        model_name=config["llm"]["model_name"],
        api_key=config["llm"]["api_key"]
    )

    # Step 6 — Run predictions
    logger.info("Running predictions...")
    predictions = []
    for idx, row in df_eval.iterrows():
        logger.info(f"Predicting [{idx+1}/{len(df_eval)}]: {row['text']}")
        result = nlu.predict(row["text"], intents_data)
        predictions.append(result["intent"])

    df_eval["predicted_intent"] = predictions

    # Step 7 — Evaluate
    logger.info("Evaluating predictions...")
    evaluator = NLUEvaluator()
    metrics, confusion_df, per_intent_df = evaluator.evaluate_with_results(
        df_eval.rename(columns={"true_intent": "intent"})
    )

    logger.info("Pipeline completed successfully")
    return metrics, confusion_df, per_intent_df, df_eval