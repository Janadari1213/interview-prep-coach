"""
Model Router Module for Interview Preparation Coach.

Centralizes LLM API calls across Groq and OpenRouter providers:
- Groq API: Selected for Question Generation (Interviewer Agent) and Answer Assessment (Evaluator Agent)
  due to extremely low latency and high throughput.
- OpenRouter API: Selected for Coaching Feedback Generation (Coach Agent) due to access to
  top-tier reasoning models (e.g. Llama 3.3 70B / Claude / GPT-4o).

API key resolution checks `st.secrets` first (for Streamlit Community Cloud deployment),
falling back to `os.getenv()` from `.env` for local development.
"""

import json
import os
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# Try importing streamlit for secrets checking
try:
    import streamlit as st
except ImportError:
    st = None

# Import SDK clients
try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

load_dotenv()


def get_api_key(key_name: str) -> Optional[str]:
    """
    Retrieve API key checking st.secrets first, then os.getenv().
    """
    if st is not None:
        try:
            if key_name in st.secrets:
                return st.secrets[key_name]
        except Exception:
            pass
    return os.getenv(key_name)


def call_groq(prompt: str, model_name: str = "llama-3.1-8b-instant", system_prompt: str = "You are an AI assistant.", json_mode: bool = False) -> str:
    """
    Route LLM call to Groq API.
    
    Why Groq? Rapid inference speed (<500ms response time) makes it ideal for real-time 
    interview question generation and rapid answer scoring.
    """
    api_key = get_api_key("GROQ_API_KEY")

    if api_key and api_key != "your_groq_api_key_here" and Groq is not None:
        try:
            client = Groq(api_key=api_key)
            kwargs = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.4,
                "max_tokens": 4096
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ModelRouter Warning] Groq API call failed: {e}")

    # Return empty string to signal fallback in calling agent
    return ""


def call_openrouter(prompt: str, model_name: str = "meta-llama/llama-3.3-70b-instruct", system_prompt: str = "You are an AI assistant.", json_mode: bool = False) -> str:
    """
    Route LLM call to OpenRouter API.
    
    Why OpenRouter? Access to state-of-the-art reasoning models delivers deep reflection, 
    pedagogical clarity, and high-quality coaching recommendations.
    """
    api_key = get_api_key("OPENROUTER_API_KEY")

    if api_key and api_key != "your_openrouter_api_key_here" and OpenAI is not None:
        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ModelRouter Warning] OpenRouter API call failed: {e}")

    return ""
