from dotenv import load_dotenv

import os

load_dotenv()

api_key  = os.getenv('API_KEY') or os.getenv('OPENAI_API_KEY') or ''
base_url = os.getenv('BASE_URL') or ''
model_id = os.getenv('MODEL_ID') or ''
platform = os.getenv('PLATFORM') or 'openai_compat'

def req_key() -> str:
    return api_key

def req_url(fallback: str = '') -> str:
    return base_url if base_url else fallback

def req_model(fallback: str = '') -> str:
    return model_id if model_id else fallback