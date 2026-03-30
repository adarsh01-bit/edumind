import os

# Check if running on cloud (Streamlit secrets)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# If API key exists → use cloud
USE_GROQ = GROQ_API_KEY is not None
