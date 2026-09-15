"""
This is for emotional profile related visualizations.
"""
import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure
import plotly.graph_objects as go


def album_emotion_heatmap(df: pd.DataFrame) -> Figure:
    """
    Create a heatmap showing average emotion per album.
    """

    # Define emotion order
    emotion_columns = [
        "emotion_Positive",
        "emotion_Negative",
        "emotion_Anger",
        "emotion_Anticipation",
        "emotion_Disgust",
        "emotion_Fear",
        "emotion_Joy",
        "emotion_Sadness",
        "emotion_Surprise",
        "emotion_Trust",
    ]

    emotion_labels = [
        column.removeprefix("emotion_")
        for column in emotion_columns
    ]

    heatmap_df = df[["album"] + emotion_columns].copy()

    heatmap_df = heatmap_df.set_index("album")

    heatmap_df.columns = emotion_labels

    fig = px.imshow(
        heatmap_df,
        labels={
            "x": "Emotion",
            "y": "Album",
            "color": "Emotion Score"
        },
        title="Emotional Profile by Album",
        aspect="auto"
    )

    return fig


def create_emotion_heatmap(comparison_df) -> Figure:
    """
    Prepare a dataframe for emotion scores heatmap for Album Comparison.
    """
    emotion_columns = [
        "emotion_Positive",
        "emotion_Negative",
        "emotion_Anger",
        "emotion_Anticipation",
        "emotion_Disgust",
        "emotion_Fear",
        "emotion_Joy",
        "emotion_Sadness",
        "emotion_Surprise",
        "emotion_Trust",
    ]

    emotion_labels = [
        column.removeprefix("emotion_")
        for column in emotion_columns
    ]

    heatmap_data = comparison_df.set_index("album")[emotion_columns].T

    heatmap_data.index = emotion_labels

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            text=heatmap_data.values,
            texttemplate="%{text:.2f}",
            colorscale="Viridis",
            colorbar={"title": "Score"},
        )
    )

    fig.update_layout(
        title="Emotional Profile",
        xaxis_title="Album",
        yaxis_title="Emotion"
    )

    return fig


def create_emotion_bar_chart(emotion_data):
    fig = px.bar(
        emotion_data,
        x="score",
        y="emotion",
        orientation="h",
        title="Emotion Profile",
    )

    fig.update_layout(
        xaxis_title="Score",
        yaxis_title="",
        yaxis=dict(
            categoryorder="array",
            categoryarray=emotion_data["emotion"].tolist(),
        ),
    )

    return fig


def create_artist_sentiment_metrics_dataframe(
        df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a dataframe containing artist sentiment metrics and emotional profile.

    Args:
        df (pd.DataFrame): dataframe containing sentiment and emotional profile data
        for each artist

    Returns:
        pd.DataFrame: dataframe containing artist sentiment and emotional profile data
    """
    return (
        df.groupby("artist")
        .agg(
            sentiment_polarity=("sentiment_polarity", "mean"),
            subjectivity=("subjectivity", "mean"),
            positivity=("emotion_Positive", "mean"),
            negativity=("emotion_Negative", "mean"),
            anger=("emotion_Anger", "mean"),
            anticipation=("emotion_Anticipation", "mean"),
            disgust=("emotion_Disgust", "mean"),
            fear=("emotion_Fear", "mean"),
            joy=("emotion_Joy", "mean"),
            sadness=("emotion_Sadness", "mean"),
            surprise=("emotion_Surprise", "mean"),
            trust=("emotion_Trust", "mean")
        )
        .reset_index()
    )