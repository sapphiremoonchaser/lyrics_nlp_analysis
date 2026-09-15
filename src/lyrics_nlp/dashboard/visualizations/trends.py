import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure

from lyrics_nlp.dashboard.visualizations.preparation import (
    prepare_words_by_album,
    prepare_words_by_artist,
    prepare_reading_time_by_artist,
    prepare_flesch_reading_ease_by_artist,
    prepare_flesch_kincaid_by_artist,
    prepare_gunning_fog_by_artist,
    prepare_lexical_diversity_by_artist
)

def average_words_scatterplot(
    df: pd.DataFrame,
    by: str
) -> Figure:
    """
    Create a scatter plot to show average words over time by album.
    Args:
        df: dataframe containing album/artist and word count
        by: what to group by ("artist" or "album")

    Returns:
        plotly scatter plot with avg word over time
    """
    if by == "album":
        df = prepare_words_by_album(df)
        hover_column = "album"

    elif by == "artist":
        df = prepare_words_by_artist(df)
        hover_column = "artist"

    else:
       raise ValueError("Invalid value for by")

    fig = px.scatter(
        df,
        x="year",
        y="avg_words_per_song",
        color="artist" if by == "artist" else None,
        hover_name=hover_column,
        labels={
            "year": "Year",
            "avg_words_per_song": "Average Words per Song",
            "artist": "Artist"
        },
        title="Average Words per Song Over Time"
    )

    return fig


def average_reading_time_scatterplot(
    df: pd.DataFrame,
    by: str
) -> Figure:
    """
    Create a scatter plot to show average reading time by year.

    Args:
        df: dataframe containing album/artist and reading time
        by: what to group by ("artist" or "album")

    Returns:
        plotly scatter plot with avg reading time over time
    """
    if by == "album":
        df = prepare_reading_time_by_artist(df)
        hover_column = "album"

    elif by == "artist":
        df = prepare_reading_time_by_artist(df)
        hover_column = "artist"

    else:
       raise ValueError("Invalid value for by")

    fig = px.scatter(
        df,
        x="year",
        y="avg_reading_time",
        color="artist" if by == "artist" else None,
        hover_name=hover_column,
        labels={
            "year": "Year",
            "avg_reading_time": "Average Reading Time",
            "artist": "Artist"
        },
        title="Average Reading Time Over Time"
    )

    return fig


def average_flesch_reading_ease_scatterplot(
    df: pd.DataFrame
) -> Figure:
    """
    Create a scatter plot to show flesch reading ease score over time.

    Args:
        df: dataframe containing album/artist and flesch reading ease score
        by: what to group by ("artist" or "album")

    Returns:
        plotly scatter plot flesch reading ease score over time
    """
    df = prepare_flesch_reading_ease_by_artist(df)

    fig = px.scatter(
        df,
        x="year",
        y="avg_flesch_reading_ease",
        color="artist",
        hover_name="artist",
        labels={
            "year": "Year",
            "avg_flesch_reading_ease": "Flesch Reading Ease",
            "artist": "Artist"
        },
        title="Flesch Reading Ease Over Time"
    )

    return fig


def average_flesch_kincaid_scatterplot(
    df: pd.DataFrame
) -> Figure:
    """
    Create a scatter plot showing average Flesch-Kincaid
    grade level over time.

    Args:
        df: dataframe containing artist and flesch kincaid reading ease score

    Returns:
        plotly scatter plot flesch kincaid grade level over time
    """
    df = prepare_flesch_kincaid_by_artist(df)

    fig = px.scatter(
        df,
        x="year",
        y="avg_flesch_kincaid",
        color="artist",
        hover_name="artist",
        labels={
            "year": "Year",
            "avg_flesch_kincaid": "Average Grade Level",
            "artist": "Artist"
        },
        title="Average Flesch-Kincaid Grade Level Over Time"
    )

    return fig


def average_gunning_fog_scatterplot(
    df: pd.DataFrame
) -> Figure:
    """
    Create a scatter plot showing average Gunning Fog scores over time.

    Args:
        df: dataframe containing artist and gunning fog score

    Returns:
        plotly scatter plot gunning fog score over time
    """
    df = prepare_gunning_fog_by_artist(df)

    fig = px.scatter(
        df,
        x="year",
        y="avg_gunning_fog",
        color="artist",
        hover_name="artist",
        labels={
            "year": "Year",
            "avg_gunning_fog": "Average Gunning Fog",
            "artist": "Artist"
        },
        title="Average Gunning Fog Score over Time"
    )

    return fig


def average_lexical_diversity_scatterplot(
    df: pd.DataFrame
) -> Figure:
    """
    Create a scatterplot showing lexical diversity over time.

    Args:
        df: dataframe containing artist and lexical diversity

    Returns:
        plotly scatter plot lexical diversity over time
    """
    df = prepare_lexical_diversity_by_artist(df)

    fig = px.scatter(
        df,
        x="year",
        y="avg_lexical_diversity",
        color="artist",
        hover_name="artist",
        labels={
            "year": "Year",
            "avg_lexical_diversity": "Average Lexical Diversity",
            "artist": "Artist"
        },
        title="Average Lexical Diversity Over Time"
    )

    return fig


def create_metrics_scatter(
    df: pd.DataFrame,
    metric: str,
    title: str,
    y_label: str,
    color: str
) -> Figure:
    """
    Create four scatter plots with average word count, average line count, average
    words per line, and readability score.
    """
    fig = px.scatter(
        df,
        x="year",
        y=metric,
        hover_name="album",
        hover_data={
            "year": True,
            metric: ":.2f"
        },
        title=title,
    )

    fig.update_traces(
        marker=dict(color=color),
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=y_label
    )

    return fig


def create_sentiment_scatter(df: pd.DataFrame) -> Figure:
    """
    Create a scatter plot with positive and negative word ratio.
    """
    fig = px.scatter(
        df,
        x="year",
        y="positive_word_ratio",
        hover_name="album",
        title="Positive and Negative Language"
    )

    fig.update_traces(
        marker=dict(color="#6699ff"),
        name="Positive",
    )

    fig.add_scatter(
        x=df["year"],
        y=df["negative_word_ratio"],
        mode="markers",
        name="Negative",
        marker=dict(color="#ff99cc")
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Word Ratio"
    )

    return fig


