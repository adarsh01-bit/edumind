import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# TRUE on cloud, FALSE on local
USE_GROQ = GROQ_API_KEY is not None
