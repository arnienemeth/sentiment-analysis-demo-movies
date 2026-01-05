import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from transformers import pipeline

# ============================================================
# MOVIE DATA (Pre-analyzed results)
# ============================================================

MOVIES_DATA = [
    {"rank": 1, "title": "The Shawshank Redemption", "year": 1994, "sentiment": "Positive", "score": 98, "reviews": 2500},
    {"rank": 2, "title": "The Godfather", "year": 1972, "sentiment": "Positive", "score": 96, "reviews": 1800},
    {"rank": 3, "title": "The Dark Knight", "year": 2008, "sentiment": "Positive", "score": 94, "reviews": 3200},
    {"rank": 4, "title": "Pulp Fiction", "year": 1994, "sentiment": "Positive", "score": 92, "reviews": 2100},
    {"rank": 5, "title": "Forrest Gump", "year": 1994, "sentiment": "Positive", "score": 91, "reviews": 2800},
    {"rank": 6, "title": "Inception", "year": 2010, "sentiment": "Positive", "score": 89, "reviews": 3500},
    {"rank": 7, "title": "The Matrix", "year": 1999, "sentiment": "Positive", "score": 87, "reviews": 2200},
    {"rank": 8, "title": "Interstellar", "year": 2014, "sentiment": "Positive", "score": 85, "reviews": 2900},
    {"rank": 9, "title": "Parasite", "year": 2019, "sentiment": "Positive", "score": 84, "reviews": 1500},
    {"rank": 10, "title": "The Lion King", "year": 1994, "sentiment": "Positive", "score": 82, "reviews": 1900},
    {"rank": 11, "title": "Avatar", "year": 2009, "sentiment": "Mixed", "score": 65, "reviews": 4100},
    {"rank": 12, "title": "Transformers", "year": 2007, "sentiment": "Mixed", "score": 55, "reviews": 2300},
    {"rank": 13, "title": "Twilight", "year": 2008, "sentiment": "Mixed", "score": 45, "reviews": 1800},
    {"rank": 14, "title": "The Room", "year": 2003, "sentiment": "Negative", "score": 15, "reviews": 900},
    {"rank": 15, "title": "Cats", "year": 2019, "sentiment": "Negative", "score": 12, "reviews": 1200},
    {"rank": 16, "title": "Battlefield Earth", "year": 2000, "sentiment": "Negative", "score": 8, "reviews": 600},
    {"rank": 17, "title": "Disaster Movie", "year": 2008, "sentiment": "Negative", "score": 5, "reviews": 400},
]

MOVIES_DF = pd.DataFrame(MOVIES_DATA)

# Model metrics
MODEL_METRICS = {
    "accuracy": 85,
    "precision": 86,
    "recall": 84,
    "f1_score": 85
}

# Dataset info
DATASET_INFO = {
    "total_reviews": 10662,
    "train_samples": 8530,
    "test_samples": 1066,
    "validation_samples": 1066,
    "positive_reviews": 5331,
    "negative_reviews": 5331
}

# ============================================================
# LOAD MODEL FOR CUSTOM ANALYSIS
# ============================================================

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def create_movie_ranking_chart():
    """Create horizontal bar chart of movie rankings."""
    df = MOVIES_DF.sort_values("score", ascending=True).tail(10)
    
    colors = ["#00CC66" if s == "Positive" else "#FF9800" if s == "Mixed" else "#FF4444" 
              for s in df["sentiment"]]
    
    fig = go.Figure(go.Bar(
        x=df["score"],
        y=df["title"],
        orientation="h",
        marker_color=colors,
        text=df["score"].apply(lambda x: f"{x}%"),
        textposition="outside"
    ))
    
    fig.update_layout(
        title="🏆 Top 10 Movies by Sentiment Score",
        xaxis_title="Sentiment Score (%)",
        yaxis_title="",
        height=500,
        template="plotly_dark",
        xaxis=dict(range=[0, 105])
    )
    
    return fig

def create_sentiment_distribution():
    """Create pie chart of sentiment distribution."""
    sentiment_counts = MOVIES_DF["sentiment"].value_counts()
    
    colors = {"Positive": "#00CC66", "Mixed": "#FF9800", "Negative": "#FF4444"}
    
    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        marker_colors=[colors[s] for s in sentiment_counts.index],
        hole=0.4,
        textinfo="label+percent"
    )])
    
    fig.update_layout(
        title="📊 Sentiment Distribution Across Movies",
        height=400,
        template="plotly_dark"
    )
    
    return fig

def create_metrics_chart():
    """Create bar chart of model performance metrics."""
    metrics = list(MODEL_METRICS.keys())
    values = list(MODEL_METRICS.values())
    
    fig = go.Figure(go.Bar(
        x=metrics,
        y=values,
        marker_color=["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"],
        text=values,
        textposition="outside",
        texttemplate="%{text}%"
    ))
    
    fig.update_layout(
        title="🎯 Model Performance Metrics",
        yaxis_title="Score (%)",
        height=400,
        template="plotly_dark",
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def create_dataset_chart():
    """Create chart showing dataset split."""
    labels = ["Train", "Validation", "Test"]
    values = [DATASET_INFO["train_samples"], DATASET_INFO["validation_samples"], DATASET_INFO["test_samples"]]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=["#4CAF50", "#2196F3", "#FF9800"],
        hole=0.3,
        textinfo="label+value"
    )])
    
    fig.update_layout(
        title=f"📁 Dataset Split (Total: {DATASET_INFO['total_reviews']:,} reviews)",
        height=400,
        template="plotly_dark"
    )
    
    return fig

def create_year_analysis():
    """Create scatter plot of movies by year and score."""
    fig = px.scatter(
        MOVIES_DF,
        x="year",
        y="score",
        size="reviews",
        color="sentiment",
        hover_name="title",
        color_discrete_map={"Positive": "#00CC66", "Mixed": "#FF9800", "Negative": "#FF4444"},
        title="📅 Movies by Year and Sentiment Score"
    )
    
    fig.update_layout(
        height=450,
        template="plotly_dark",
        xaxis_title="Release Year",
        yaxis_title="Sentiment Score (%)"
    )
    
    return fig

# ============================================================
# INTERACTIVE FUNCTIONS
# ============================================================

def get_movie_details(movie_name):
    """Get details for selected movie."""
    movie = MOVIES_DF[MOVIES_DF["title"] == movie_name].iloc[0]
    
    emoji = "✅" if movie["sentiment"] == "Positive" else "⚠️" if movie["sentiment"] == "Mixed" else "❌"
    
    details = f"""
    ## {emoji} {movie['title']} ({movie['year']})
    
    | Metric | Value |
    |--------|-------|
    | **Sentiment** | {movie['sentiment']} |
    | **Score** | {movie['score']}% |
    | **Rank** | #{movie['rank']} |
    | **Reviews Analyzed** | {movie['reviews']:,} |
    """
    
    # Create gauge chart for this movie
    color = "#00CC66" if movie["sentiment"] == "Positive" else "#FF9800" if movie["sentiment"] == "Mixed" else "#FF4444"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=movie["score"],
        title={"text": f"Sentiment Score"},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#FFEEEE"},
                {"range": [40, 70], "color": "#FFFFEE"},
                {"range": [70, 100], "color": "#EEFFEE"},
            ],
        }
    ))
    
    fig.update_layout(height=300, template="plotly_dark")
    
    return details, fig

def analyze_custom_review(text):
    """Analyze custom review text."""
    if not text.strip():
        return "Please enter a review to analyze.", None
    
    result = classifier(text)[0]
    label = result["label"]
    score = result["score"]
    
    emoji = "✅ POSITIVE" if label == "POSITIVE" else "❌ NEGATIVE"
    
    output = f"""
    ## {emoji}
    
    **Confidence:** {score*100:.1f}%
    
    **Analysis:** This review is classified as **{label.lower()}** with {score*100:.1f}% confidence.
    """
    
    # Create gauge
    color = "#00CC66" if label == "POSITIVE" else "#FF4444"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        title={"text": label},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50], "color": "#F