"""
The purpose of this file is to prepare data for visualizations.
"""
import pandas as pd
import ast


metric_groups = {
        "Song Structure": [
            "avg_words_per_song",
            "avg_lines_per_song",
            "avg_words_per_line",
            "reading_minutes",
        ],
        "Vocabulary": [
            "unique_words",
            "lexical_diversity",
            "avg_word_length",
        ],
        "Readability": [
            "flesch_reading_ease",
            "flesch_kincaid",
            "gunning_fog",
        ],
        "Sentiment and Emotion": [
            "positive_word_ratio",
            "negative_word_ratio",
            "sentiment_polarity",
            "subjectivity"
        ]
    }

EMOTION_ORDER = [
        "Positive",
        "Negative",
        "Anger",
        "Anticipation",
        "Disgust",
        "Fear",
        "Joy",
        "Sadness",
        "Surprise",
        "Trust",
    ]

song_structure_metrics = {
    "avg_words_per_song": {
        "title": "Average Words per Song",
        "y_label": "Words",
        "color": "#636EFA",
    },
    "avg_lines_per_song": {
        "title": "Average Lines per Song",
        "y_label": "Lines",
        "color": "#636EFA",
    },
    "avg_words_per_line": {
        "title": "Average Words per Line",
        "y_label": "Words per Line",
        "color": "#636EFA",
    },
    "reading_minutes": {
        "title": "Average Reading Time",
        "y_label": "Minutes",
        "color": "#636EFA",
    },
}


def prepare_words_by_album(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a dataframe with words by album to be used in the visualizations
    of word count over time or word count comparisons on the album level.

    Args:
        df: DataFrame to be aggregated

    Returns:
        Aggregated dataframe with word count by album sorted by year.
    """
    words_by_album = (
        df[
            ["album", "year", "avg_words_per_song"]
        ]
        .drop_duplicates()
        .sort_values("year")
    )

    words_by_album["album"] = pd.Categorical(
        words_by_album["album"],
        categories=words_by_album["album"],
        ordered=True
    )

    return words_by_album


def prepare_structure_metric_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a dataframe with words by album, lines per song, average words per line,
    average reading minutes by album.
    """
    df = (
        df.groupby(["album", "year"], as_index=False)
        .agg(
            avg_words_per_song=("word_count", "mean"),
            avg_lines_per_song=("line_count", "mean"),
            avg_words_per_line=("words_per_line", "mean"),
            avg_reading_minutes=("reading_minutes", "mean"),
        )
        .sort_values("year")
    )

    return df


def prepare_structural_comparison(comparison_df) -> pd.DataFrame:
    """
    Prepare a dataframe for lyrical structural analysis for Album Comparison.
    """
    structural_comparison = comparison_df[
        [
            "album",
            "avg_words_per_song",
            "avg_lines_per_song",
            "avg_words_per_line",
            "avg_reading_time"
        ]
    ].set_index("album").T

    structural_comparison.index = [
        "Average Words per Song",
        "Average Lines per Song",
        "Average Words per Line",
        "Average Reading Time (minutes)"
    ]

    return structural_comparison.round(2)


def prepare_vocabulary_comparison(comparison_df) -> pd.DataFrame:
    """
    Prepare a dataframe for the vocabulary comparison for Album Comparison.
    """
    vocabulary_comparison = comparison_df[
        [
            "album",
            "vocabulary_size",
            "lexical_diversity",
            "average_word_length"
        ]
    ].set_index("album").T

    vocabulary_comparison.index = [
        "Average Vocabulary Size",
        "Average Lexical Diversity",
        "Average Word Length"
    ]

    return vocabulary_comparison.round(2)


def prepare_readability_comparison(comparison_df) -> pd.DataFrame:
    """
    Prepare a dataframe for the readability comparison for Album Comparison.
    """
    readability_comparison = comparison_df[
        [
            "album",
            "flesch_reading_ease",
            "flesch_kincaid",
            "gunning_fog"
        ]
    ].set_index("album").T

    readability_comparison.index = [
        "Average Flesch Reading Ease Score",
        "Average Flesch Kinematic Score",
        "Average Gunning Fog Score"
    ]

    return readability_comparison.round(2)


def prepare_sentiment_comparison(comparison_df) -> pd.DataFrame:
    """
    Prepare a dataframe for the positive and negative word ratio for Album Comparison.
    """
    sentiment_comparison = comparison_df[
        [
            "album",
            "positive_word_ratio",
            "negative_word_ratio",
            "sentiment_polarity",
            "subjectivity"
        ]
    ].set_index("album").T

    sentiment_comparison.index = [
        "Positive Language",
        "Negative Language",
        "Sentiment Polarity",
        "Sentiment Subjectivity"
    ]

    return sentiment_comparison.round(2)


def prepare_wordcloud_text(
    df: pd.DataFrame,
    column: str,
    value: str
) -> str:
    """
    Prepare a string for creating a word cloud
    Args:
        df (pd.DataFrame): Dataframe containing lyrics.
        column: album column
        value: album name
    """
    lyrics = df[
        df[column] == value
    ]["nlp_cleaned_lyrics"]

    return " ".join(
        " ".join(ast.literal_eval(lyrics))
        if isinstance(lyrics, str)
        else " ".join(lyrics)
        for lyrics in lyrics
    )


def prepare_words_by_artist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average words per song by artist and year.

    Args:
        df: dataframe containing artist, year, and word count.

    Returns:
        Dataframe containing artist, year, and average words per song.
    """
    return (
        df.groupby(["artist", "year"])["word_count"]
        .mean()
        .reset_index(name="avg_words_per_song")
    )


def prepare_reading_time_by_artist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average reading time for each artist and year.

    Args:
        df: dataframe containing artist, year, and reading_minutes

    :return:
        Dataframe containing artist, year, and average reading time for each artist.
    """
    return (
        df.groupby(["artist", "year"])["reading_minutes"]
        .mean()
        .reset_index(name="avg_reading_time")
    )


def prepare_flesch_reading_ease_by_artist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcualte average fleschh reading ease for each artist and year.

    Args:
        df: dataframe containing artist, year, and flesch_reading_ease

    Returns:
        DataFrame containing artist, year, and average flesch reading easy fore each
        artist and album.
    """
    return (
        df.groupby(["artist", "year"])["flesch_reading_ease"]
        .mean()
        .reset_index(name="avg_flesch_reading_ease")
    )

def prepare_flesch_kincaid_by_artist(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate average Flesch-Kincaid grade level by artist and year."""
    return (
        df.groupby(["artist", "year"])["flesch_kincaid"]
        .mean()
        .reset_index(name="avg_flesch_kincaid")
    )


def prepare_gunning_fog_by_artist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average Gunning Fog grade level by artist and year.

    Args:
         df: dataframe containing artist, year, and gunning_fog

    Returns:
        DataFrame containing artist, year, and average Gunning Fog grade level
    """
    return (
        df.groupby(["artist", "year"])["gunning_fog"]
        .mean()
        .reset_index(name="avg_gunning_fog")
    )

def prepare_lexical_diversity_by_artist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average lexical diversity by artist and year.

    Args:
        df: dataframe containing artist, year, and lexical diversity

    Returns:
        DataFrame containing artist, year, and average lexical diversity.
    """
    return (
        df.groupby(["artist", "year"])["lexical_diversity"]
        .mean()
        .reset_index(name="avg_lexical_diversity")
    )