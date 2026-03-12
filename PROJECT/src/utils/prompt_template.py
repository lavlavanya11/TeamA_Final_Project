def build_nlu_prompt(user_text: str, intents_data: dict) -> str:
    """
    Builds a structured prompt for Gemini NLU inference.
    Uses system + user role structure with few-shot example.
    """
    intent_names = [intent["name"] for intent in intents_data["intents"]]
    entity_schema = intents_data["entities"]

    # Few-shot example to guide the model
    few_shot_example = """
EXAMPLE:
User Input: "Book a flight to Delhi tomorrow"
Output:
{
  "intent": "book_flight",
  "confidence": 0.95,
  "entities": {
    "location": "Delhi",
    "date": "tomorrow"
  }
}
"""

    prompt = f"""
You are a strict and precise NLU (Natural Language Understanding) engine for a chatbot system.
Your job is to analyze user messages and return structured JSON output only.

ALLOWED INTENTS (choose exactly ONE):
{intent_names}

ENTITY SCHEMA (extract ONLY these entity types):
{entity_schema}

STRICT RULES:
- You MUST choose one intent from the ALLOWED INTENTS list only
- If no intent matches, use "unknown"
- Extract entities ONLY from the ENTITY SCHEMA above
- If an entity is not present in the user message, do not include it
- All entity values must be strings
- Confidence must be a float between 0.0 and 1.0
- Do NOT hallucinate or guess entities
- Return ONLY valid JSON — no explanation, no extra text, no markdown

{few_shot_example}

OUTPUT FORMAT:
{{
  "intent": "<intent_name_or_unknown>",
  "confidence": <float between 0.0 and 1.0>,
  "entities": {{
    "<entity_name>": "<entity_value>"
  }}
}}

Now analyze this user input:
"{user_text}"
"""
    return prompt