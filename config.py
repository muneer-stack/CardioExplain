from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "readme_kardiologie_bilingual_audit.tsv"
INDEX_DIR = BASE_DIR / "storage" / "faiss_cardio_readme"
TERM_CATALOG_PATH = BASE_DIR / "storage" / "term_catalog.csv"
INDEX_MANIFEST_PATH = INDEX_DIR / "index_manifest.json"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

DEFAULT_TOP_K = 5
MAX_PATIENT_TEXT_CHARS = 6000