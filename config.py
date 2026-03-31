import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# If API key exists → use cloud
USE_CLOUD = GROQ_API_KEY is not None
