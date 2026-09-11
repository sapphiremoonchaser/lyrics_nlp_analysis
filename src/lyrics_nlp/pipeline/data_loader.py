"""
LyricsDataLoader does the following:
    - reads the csv files ending with "_lyrics.csv"
    - verifies required columns exist
    - stores the dataframe
    - returns a copy of the dataframe
"""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "artist",
    "album",
    "song",
    "year",
    "lyrics"
}


class LyricsDataLoader:
    """
    This class loads the lyrics data from a csv file.
    """
    def __init__(
        self,
        filepath: str | Path
    ):
        # Set up the initial variables
        self.filepath = Path(filepath)
        self.df = None

    def load(self):
        """
        Load all *_lyrics.csv files from the raw folder.
        """
        files = sorted(
            self.filepath.glob("*_lyrics.csv")
        )

        if not files:
            raise FileNotFoundError(
                f"No *_lyrics.csv files found in {self.filepath}"
            )

        dataframes = []

        for filepath in files:
            df = pd.read_csv(filepath)

            missing = REQUIRED_COLUMNS - set(df.columns)

            if missing:
                raise ValueError(
                    f"{filepath} is missing required columns: {missing}"
                )

            dataframes.append(df)

        self.df = pd.concat(
            dataframes,
            ignore_index=True
        )

        return self.df.copy()


