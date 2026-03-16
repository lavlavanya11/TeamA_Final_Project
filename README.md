## BotTrainer – LLM-Based NLU Model Trainer & Evaluator

BotTrainer is a modular NLU (Natural Language Understanding) playground built around
Large Language Models (LLMs). It lets you:

- Classify user intents
- Extract entities
- Evaluate performance (accuracy, precision, recall, F1)
- Interactively test the NLU model via a Streamlit UI

The current backend uses **Groq-hosted Llama 3.1** (via the `groq` Python SDK) with structured JSON output.

---

### Project Structure

- `data/`
  - `intents.json` – training-style NLU dataset (intents + examples + entities)
  - `test_data.json` – labeled test set for evaluation
- `prompts/`
  - `base_prompt.txt` – core instructions and JSON format for the LLM
  - `few_shot_examples.txt` – example NLU interactions for few-shot prompting
- `src/`
  - `llm_client.py` – Groq client wrapper for Llama models
  - `prompt_builder.py` – builds the full NLU prompt
  - `intent_classifier.py` – runs intent classification + entity extraction
  - `entity_extractor.py` – helper for entity extraction
  - `nlu_pipeline.py` – high-level NLU pipeline entrypoint
- `evaluation/`
  - `metrics.py` – accuracy, precision, recall, F1, confusion matrix helpers
  - `evaluator.py` – runs the pipeline on `test_data.json`
- `utils/`
  - `json_parser.py` – robust JSON parsing for LLM outputs
  - `logger.py` – simple logging helper
- `app.py` – Streamlit web app for testing and evaluation
- `requirements.txt` – Python dependencies
- `.env` – environment variables (Groq API key, etc.)

---

### Setup

1. **Create and activate a virtual environment (optional but recommended)**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # on Windows
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   - Open `.env` and set (API key only):

     ```text
     GROQ_API_KEY=your_groq_api_key_here
     ```

   - Model name, RPM, retry settings, and dataset paths are configured in `config/config.yaml`.

---

### Run the Streamlit App

From the `bottrainer` project root:

```bash
streamlit run app.py
```

You will see:

- **Test NLU tab** – type any user message and view:
  - Predicted intent
  - Confidence
  - Extracted entities
  - Raw JSON from the LLM
- **Evaluation tab** – run evaluation on `data/test_data.json` and see:
  - Accuracy, precision, recall, F1
  - Confusion matrix

---

### Next Steps / Extensions

- Add more intents and entities to `intents.json` and expand `test_data.json`.
- Improve prompts and few-shot examples to reduce confusion between similar intents.
- Add charts (e.g., confusion matrix heatmap) in the Streamlit evaluation tab.
- Swap or add other LLM providers (OpenAI, Mistral, etc.) behind a common client interface.

