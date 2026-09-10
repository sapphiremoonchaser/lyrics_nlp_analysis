"""Run the pipeline and create the song and album level dataframes."""

from lyrics_nlp.pipeline.pipeline import pipeline

df = pipeline("../../../data/raw")
