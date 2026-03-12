import yaml
import os
from dotenv import load_dotenv

load_dotenv()

def load_config(config_path: str = "config/config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Always override API key from .env — ignores blank value in config.yaml
    env_key = os.getenv("GROQ_API_KEY", "")
    if env_key:
        config["llm"]["api_key"] = env_key
    
    return config