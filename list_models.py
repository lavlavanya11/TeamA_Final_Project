import os

import google.generativeai as genai
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set in .env")

    genai.configure(api_key=api_key)

    print("Models that support generateContent for your key:\n")
    for m in genai.list_models():
        methods = getattr(m, "supported_generation_methods", []) or []
        if "generateContent" in methods:
            print(m.name)


if __name__ == "__main__":
    main()

