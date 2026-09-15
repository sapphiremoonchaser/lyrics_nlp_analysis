"""
This is the main script that runs the dashboard.
"""
import streamlit as st
import pandas as pd
from pathlib import Path

from lyrics_nlp.dashboard.visualizations.color_palettes import album_palettes

from lyrics_nlp.dashboard.visualizations.preparation import (
    metric_groups,
    EMOTION_ORDER,
    prepare_structural_comparison,
    prepare_vocabulary_comparison,
    prepare_readability_comparison,
    prepare_sentiment_comparison,
    prepare_wordcloud_text
)

from lyrics_nlp.dashboard.visualizations.comparisons import (
    create_album_boxplot,
    create_wordcloud,
    create_album_timeline_multi_artists
)

from lyrics_nlp.dashboard.visualizations.trends import (
    average_words_over_time_scatterplot,
    create_metrics_scatter,
    create_sentiment_scatter,
    average_reading_time_over_time_scatterplot
)

from lyrics_nlp.dashboard.visualizations.emotions import (
    album_emotion_heatmap,
    create_emotion_heatmap,
    create_emotion_bar_chart
)

st.set_page_config(
    page_title="NLP Lyric Analysis",
    page_icon="🎵",
    layout="wide"
)

# --------------------
# Load data
# --------------------
DATA_DIR = Path("data/processed")

song_df: pd.DataFrame = pd.read_csv(
    DATA_DIR / "song_level_stats.csv"
)

album_df: pd.DataFrame = pd.read_csv(
    DATA_DIR / "album_level_stats.csv"
)

# --------------------
# Sidebar
# --------------------

st.sidebar.title("Lyric Analysis")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Lyrical Style",
        "Album Comparison",
        "Song Explorer",
        "Artist Comparison"
    ]
)

# --------------------
# Page content
# --------------------

if page == "Overview":
    st.title("NLP Lyric Analysis")

    st.write(
        "An exploration of lyrics "
        "using natural language processing."
    )

    artist_names = (
        album_df["artist"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    artist = st.selectbox(
        "Artist",
        artist_names,
        index=0
    )

    selected_album_df = album_df[
        album_df["artist"] == artist
    ]

    selected_song_df = song_df[
        song_df["artist"] == artist
    ]

    # KPI Cards
    album_count = selected_album_df["album"].nunique()
    song_count = selected_song_df["song"].nunique()
    first_year = selected_album_df["year"].min()
    last_year = selected_album_df["year"].max()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Albums", album_count)

    with col2:
        st.metric("Songs", song_count)

    with col3:
        st.metric("First Album", first_year)

    with col4:
        st.metric("Latest Album", last_year)

    # Scatter Plot
    fig = average_words_over_time_scatterplot(
        selected_album_df,
        "album"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Average number of words per song across albums."
    )

    # Emotional Profile Heatmap
    fig = album_emotion_heatmap(selected_album_df)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Higher values indicate a higher association to a particular emotion."
    )


elif page == "Lyrical Style":

    st.title("Lyrical Style Over Time")

    st.write(
        "How does lyrical style change "
        "across her discography?"
    )

    artist_names = (
        album_df["artist"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    artist = st.selectbox(
        "Artist",
        artist_names,
        index=0
    )

    selected_album_df = album_df[
        album_df["artist"] == artist
        ]

    selected_group = st.selectbox(
        "Metric Group",
        metric_groups.keys()
    )

    if selected_group == "Song Structure":

        col1, col2 = st.columns(2)

        with col1:
            fig = create_metrics_scatter(
                selected_album_df,
                "avg_words_per_song",
                "Average Words per Song",
                "Words",
                 "#00ffff"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = create_metrics_scatter(
                selected_album_df,
                "avg_lines_per_song",
                "Average Lines per Song",
                "Lines",
                "#99ff99"
            )
            st.plotly_chart(fig, use_container_width=True)

        fig = create_metrics_scatter(
            selected_album_df,
            "avg_reading_time",
            "Average Reading Time per Song",
            "Minutes",
            "#cc99ff"
        )
        st.plotly_chart(fig, use_container_width=True)

    if selected_group == "Vocabulary":
        col1, col2 = st.columns(2)

        with col1:
            fig = create_metrics_scatter(
                selected_album_df,
                "vocabulary_size",
                "Vocabulary Size",
                "words",
                "#00ffff"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Based on unique word count"
            )

        with col2:
            fig = create_metrics_scatter(
                selected_album_df,
                "lexical_diversity",
                "Lexical Diversity",
                "score",
                "#99ff99"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Higher values may indicate less repetition"
            )

        fig = create_metrics_scatter(
            selected_album_df,
            "average_word_length",
            "Word Length",
            "characters",
            "#cc99ff"
        )
        st.plotly_chart(fig, use_container_width=True)

    if selected_group == "Readability":
        col1, col2 = st.columns(2)

        with col1:
            fig = create_metrics_scatter(
                selected_album_df,
                "flesch_reading_ease",
                "Flesch Reading Ease",
                "score",
                "#00ffff"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Smaller negative values indicate a more difficult reading level."
            )


        with col2:
            fig = create_metrics_scatter(
                selected_album_df,
                "flesch_kincaid",
                "Flesch-Kincaid",
                "score",
                "#99ff99"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Lower values indicate a more difficult reading level."
            )

        fig = create_metrics_scatter(
            selected_album_df,
            "gunning_fog",
            "Gunning Fog",
            "score",
            "#cc99ff"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Higher values indicate a more difficult reading level."
        )

    if selected_group == "Sentiment and Emotion":

        fig = create_sentiment_scatter(selected_album_df)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Larger values indicates positive language, with 0 being 0 positive "
            "words."
        )


elif page == "Album Comparison":

    st.title("Album Comparison")

    st.write(
        "Compare lyrical characteristics across albums."
    )

    st.subheader("Select Albums to Compare")

    # Get unique artists
    artist_names = (
        album_df["artist"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    album_names = (
        album_df["album"]
        .dropna()
        .astype(str)
        .tolist()
    )

    # Album Selectors
    col1, col2 = st.columns(2)

    with col1:
        artist_1 = st.selectbox(
            "Artist 1",
            artist_names,
            index=0
        )

        album_names_1 = (
            album_df.loc[
                album_df["artist"] == artist_1,
                "album"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        album_1 = st.selectbox(
            "Album 1",
            album_names_1,
            index=0
        )

    with col2:
        artist_2 = st.selectbox(
            "Artist 2",
            artist_names,
            index=1
        )

        album_names_2 = (
            album_df.loc[
                album_df["artist"] == artist_2,
                "album"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        album_2 = st.selectbox(
            "Album 2",
            album_names_2,
            index=0
        )

    selected_group = st.selectbox(
        "Metric Group",
        metric_groups.keys()
    )

    comparison_df = album_df[
        (
                (album_df["artist"] == artist_1)
                & (album_df["album"] == album_1)
        )
        |
        (
                (album_df["artist"] == artist_2)
                & (album_df["album"] == album_2)
        )
        ].copy()

    # Song level df for box plots
    comparison_songs = song_df[
        (
                (song_df["artist"] == artist_1)
                & (song_df["album"] == album_1)
        )
        |
        (
                (song_df["artist"] == artist_2)
                & (song_df["album"] == album_2)
        )
        ].copy()

    if selected_group == "Song Structure":
        # Table for structural comparison
        structural_comparison = prepare_structural_comparison(comparison_df)

        st.subheader("Structural Comparison")

        st.dataframe(
            structural_comparison,
            use_container_width=True
        )

        col1, col2 = st.columns(2)

        with col1:
            fig = create_album_boxplot(
                comparison_songs,
                "word_count",
                "Song Word Count Distribution",
                "Words per Song"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = create_album_boxplot(
                comparison_songs,
                "line_count",
                "Song Line Count Distribution",
                "lines per Song"
            )
            st.plotly_chart(fig, use_container_width=True)

    if selected_group == "Vocabulary":
        # Table for Vocabulary Comparison
        vocabulary_comparison = prepare_vocabulary_comparison(comparison_df)

        st.subheader("Vocabulary Comparison")

        st.dataframe(
            vocabulary_comparison,
            use_container_width=True
        )

        col1, col2 = st.columns(2)

        with col1:
            fig = create_album_boxplot(
                comparison_songs,
                "vocabulary_size",
                "Vocabulary Size Distribution",
                "Words"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = create_album_boxplot(
                comparison_songs,
                "lexical_diversity",
                "Lexical Diversity Distribution",
                "Score"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption("Lexical Diversity is the percent of words that are unique.")

        # Wordclouds
        st.subheader("Word Clouds")

        col3, col4 = st.columns(2)

        with col3:
            # Filter song level dataframe
            album_1_text = prepare_wordcloud_text(
                song_df,
                "album",
                album_1
            )

            album_1_palette = album_palettes[album_1]

            album_1_wordcloud = create_wordcloud(
                album_1_text,
                album_1_palette,
            )

            st.image(
                album_1_wordcloud.to_array(),
                use_container_width=True
            )

            st.caption(album_1)

        with col4:
            # Filter song level dataframe
            album_2_text = prepare_wordcloud_text(
                song_df,
                "album",
                album_2
            )

            album_2_palette = album_palettes[album_2]

            album_2_wordcloud = create_wordcloud(
                album_2_text,
                album_2_palette,
            )

            st.image(
                album_2_wordcloud.to_array(),
                use_container_width=True
            )

            st.caption(album_2)

    if selected_group == "Readability":
        # Table for readability comparison
        readability_comparison = prepare_readability_comparison(comparison_df)

        st.subheader("Readability Comparison")

        st.dataframe(
            readability_comparison,
            use_container_width=True
        )

        col1, col2 = st.columns(2)

        with col1:
            fig = create_album_boxplot(
                comparison_songs,
                "flesch_reading_ease",
                "Flesch Reading Ease Score Distribution",
                "Score"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = create_album_boxplot(
                comparison_songs,
                "flesch_kincaid",
                "Flesch Kincaid Score Distribution",
                "Score"
            )
            st.plotly_chart(fig, use_container_width=True)

        fig = create_album_boxplot(
            comparison_songs,
            "gunning_fog",
            "Gunning Fog Score Distribution",
            "Score"
        )
        st.plotly_chart(fig, use_container_width=True)

    if selected_group == "Sentiment and Emotion":
        # Table for sentiment comparison
        sentiment_comparison = prepare_sentiment_comparison(comparison_df)

        st.subheader("Sentiment Comparison")

        st.dataframe(
            sentiment_comparison,
            use_container_width=True
        )

        col1, col2 = st.columns(2)

        with col1:
            fig = create_album_boxplot(
                comparison_songs,
                "sentiment_polarity",
                "Sentiment Polarity Distribution",
                "Rating"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = create_album_boxplot(
                comparison_songs,
                "subjectivity",
                "Sentiment Subjectivity Distribution",
                "Score"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Emotional Profile")

        fig = create_emotion_heatmap(comparison_df)

        st.plotly_chart(fig, use_container_width=True)


elif page == "Song Explorer":

    st.title("Song Explorer")

    st.write(
        "Explore the lyrical, sentiment, and emotional "
        "characteristics of individual songs."
    )

    # Artist selector
    artists = sorted(
        song_df["artist"]
        .dropna()
        .unique()
    )

    selected_artist = st.selectbox(
        "Choose an artist.",
        artists
    )

    # Album selector
    artist_albums = sorted(
        song_df.loc[
            song_df["artist"] == selected_artist,
            "album"
        ]
        .dropna()
        .unique()
    )

    selected_album = st.selectbox(
        "Choose an album.",
        artist_albums
    )

    # Song selector
    album_songs = (
        song_df[
            (song_df["artist"] == selected_artist)
            & (song_df["album"] == selected_album)
        ]
        .sort_values("song")
    )

    selected_song = st.selectbox(
        "Choose a song.",
        album_songs["song"].tolist()
    )

    selected_row = album_songs[
        album_songs["song"] == selected_song
    ].iloc[0]

    st.header(selected_song)

    st.caption(
        f"{selected_album} - {selected_row['year']}"
    )

    # KPI columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Words",
            int(selected_row["word_count"])
        )

    with col2:
        st.metric(
            "Lexical Diversity",
            f"{selected_row['lexical_diversity']:.2f}"
        )

    with col3:
        st.metric(
            "Subjectivity",
            f"{selected_row['subjectivity']:.2f}"
        )

    # Emotion profile
    st.subheader("Emotion Profile")

    emotion_order = EMOTION_ORDER

    emotion_columns = [
        f"emotion_{emotion}"
        for emotion in emotion_order
    ]

    emotion_data = (
        selected_row[emotion_columns]
        .rename(lambda x: x.replace("emotion_", ""))
        .to_frame(name="score")
        .reset_index()
        .rename(columns={"index": "emotion"})
    )

    fig = create_emotion_bar_chart(emotion_data)
    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Lyrics and Word Cloud
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Lyrics")

        st.text_area(
            "Song lyrics",
            selected_row["lyrics"],
            height=400
        )

    with col2:
        st.subheader("Word Cloud")

        st.markdown(
            '<div style="padding-top: 100px;"></div>',
            unsafe_allow_html=True
        )

        # Filter song-level dataframe
        song_text = prepare_wordcloud_text(
            song_df,
            "song",
            selected_song
        )

        # Use selected artist and album for palette
        song_palette = album_palettes[
            selected_artist
        ][selected_album]

        song_wordcloud = create_wordcloud(
            song_text,
            song_palette,
        )

        st.image(
            song_wordcloud.to_array(),
            use_container_width=True
        )


elif page == "Artist Comparison":
    st.title("Artist Comparison")

    st.write(
        "Compare lyrical characteristics and catalog size across artists."
    )

    artists = sorted(
        song_df["artist"].dropna().unique()
    )

    selected_artists = st.multiselect(
        "Choose artists",
        options=artists,
        default=artists
    )

    comparison_albums = album_df[
        album_df["artist"].isin(selected_artists)
    ].copy()

    # Make sure year is numeric
    comparison_albums["year"] = (
        comparison_albums["year"].astype(int)
    )

    st.plotly_chart(
        create_album_timeline_multi_artists(comparison_albums),
        use_container_width=True
    )

    st.subheader(
        "Artist Catalog"
    )

    for artist in artists:

        artist_df = song_df[
            song_df["artist"] == artist
        ]

        album_count = artist_df["album"].nunique()
        song_count = artist_df["song"].nunique()

        st.subheader(artist)

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Albums",
                value=album_count
            )

        with col2:
            st.metric(
                label="Songs",
                value=song_count
            )

    # Only continue if artists have been selected
    if selected_artists:

        comparison_songs = song_df[
            song_df["artist"].isin(selected_artists)
        ].copy()

        comparison_songs["year"] = (
            comparison_songs["year"].astype(int)
        )

        col1, col2 = st.columns(2)

        with col1:

            # Average words over time
            fig = average_words_over_time_scatterplot(
                comparison_songs,
                by="artist"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            # Average reading time over time
            fig = average_reading_time_over_time_scatterplot(
                comparison_songs,
                by="artist"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

