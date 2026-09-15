import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure

from lyrics_nlp.dashboard.visualizations.preparation import (
    prepare_words_by_album,
    prepare_words_by_artist
)

def average_words_over_time_scatterplot(
        df: pd.DataFrame,
        by: str
) -> Figure:
    """
    Create a scatter plot to show average words over time by album.
    Args:
        df: dataframe containing album and word count
        by: what to group by ("artist" or "album")

    Returns:
        plotly scatter plot with avg word over time
    """
    if by == "album":
        df = prepare_words_by_album(df)

    elif by == "artist":
        df = prepare_words_by_artist(df)

    else:
       raise ValueError("Invalid value for by")

    fig = px.scatter(
        df,
        x="year",
        y="avg_words_per_song",
        color="artist" if by == "artist" else None,
        hover_name="album",
        labels={
            "year": "Year",
            "avg_words_per_song": "Average Words per Song",
            "artist": "Artist"
        },
        title="Average Words per Song over Time"
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


