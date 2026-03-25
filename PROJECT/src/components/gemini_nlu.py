import json
from groq import Groq
from src.utils.prompt_template import build_nlu_prompt
from src.utils.logger import setup_logger

logger = setup_logger("GROQ_NLU")

class GeminiNLU:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = Groq(api_key=api_key)
        logger.info(f"GroqNLU initialized with model: {model_name}")

    def predict(self, user_text: str, intents_data: dict) -> dict:
        try:
            prompt = build_nlu_prompt(user_text, intents_data)
            logger.info(f"Sending prompt for input: {user_text}")

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict NLU engine. Always return valid JSON only. No explanation, no markdown."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=500
            )

            raw_text = response.choices[0].message.content.strip()
            logger.info(f"Raw response received: {raw_text}")

            # Strip markdown fences if present
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("```").strip()
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:].strip()

            result = json.loads(raw_text)

            if "intent" not in result:
                result["intent"] = "unknown"
            if "confidence" not in result:
                result["confidence"] = 0.0
            if "entities" not in result:
                result["entities"] = {}

            logger.info(f"Predicted intent: {result['intent']} | Confidence: {result['confidence']}")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return {"intent": "unknown", "confidence": 0.0, "entities": {}}

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return {"intent": "unknown", "confidence": 0.0, "entities": {}}