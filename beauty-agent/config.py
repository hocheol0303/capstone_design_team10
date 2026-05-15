import os
from dotenv import load_dotenv

load_dotenv()

root_dir = os.path.dirname(os.path.dirname(__file__))
openai_api_key_path = os.path.join(root_dir, 'no_track', 'openai_api_key.txt')

with open(openai_api_key_path, 'r') as f:
    OPENAI_API_KEY = f.read().strip()

OPENAI_MODEL = "gpt-4.1-mini"

# PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "")

USE_MOCK_VISION = True
USE_MOCK_PUBMED = False
USE_MOCK_SCENARIO = int(os.getenv("USE_MOCK_SCENARIO", "1"))
