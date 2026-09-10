from pathlib import Path

from src.lyrics_nlp.pipeline.data_loader import LyricsDataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FOLDER = PROJECT_ROOT / "data" / "raw"

loader = LyricsDataLoader(RAW_FOLDER)
df = loader.load()

x = 1