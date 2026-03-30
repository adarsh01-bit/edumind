# config.py
# Automatically switches between Ollama (local)
# and Groq (deployed cloud) based on environment

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file

# detect if running on Streamlit Cloud
# Streamlit Cloud sets this env variable automatically
IS_CLOUD = os.environ.get("STREAMLIT_SHARING_MODE") is not None

# also check if Groq key is present
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
USE_GROQ = bool(GROQ_API_KEY) or IS_CLOUD

print(f"🔧 Running in {'☁️ Cloud (Groq)' if USE_GROQ else '🖥️ Local (Ollama)'} mode")
