# app.py
"""
Streamlit app for Sentiment Analysis visualization.
Interactive demo for movie review sentiment prediction.
"""

import streamlit as st
import torch
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sentiment Analysis Demo",
    page_icon="🎬",
    layout="wide",
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():
    """Load trained model and tokenizer."""
    model_path = "./model_output/final_model"
    
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        model.eval()
        return tokenizer, model, True
    except Exception as e:
        st.warning(f"Could not load trained model: {e}")
        st.info("Loading pre-trained model from Hugging Face instead...")
        
        # Fallback to pre-trained model
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(model_name)
        model.eval()
        return tokenizer, model, False


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_sentiment(text: str, tokenizer, model) -> dict:
    """
    Predict sentiment for given text.
    
    Returns:
        Dictionary with label, confidence, and probabilities
    """
    # Tokenize
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get results
    probs = probabilities[0].tolist()
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    
    labels = ["Negative", "Positive"]
    
    return {
        "label": labels[predicted_class],
        "confidence": probs[predicted_class],
        "negative_prob": probs[0],
        "positive_prob": probs[1],
    }


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def create_gauge_chart(confidence: float, label: str) -> go.Figure:
    """Create a gauge chart for confidence score."""
    color = "#00CC66" if label == "Positive" else "#FF4444"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"Confidence ({label})", "font": {"size": 20}},
        number={"suffix": "%", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0, 50], "color": "#FFEEEE"},
                {"range": [50, 75], "color": "#FFFFEE"},
                {"range": [75, 100], "color": "#EEFFEE"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": confidence * 100,
            },
        },
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_probability_bar(neg_prob: float, pos_prob: float) -> go.Figure:
    """Create horizontal bar chart for probabilities."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=["Sentiment"],
        x=[neg_prob * 100],
        name="Negative",
        orientation="h",
        marker_color="#FF4444",
        text=f"{neg_prob*100:.1f}%",
        textposition="inside",
    ))
    
    fig.add_trace(go.Bar(
        y=["Sentiment"],
        x=[pos_prob * 100],
        name="Positive",
        orientation="h",
        marker_color="#00CC66",
        text=f"{pos_prob*100:.1f}%",
        textposition="inside",
    ))
    
    fig.update_layout(
        barmode="stack",
        height=100,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(range=[0, 100], title="Probability (%)"),
    )
    
    return fig


# ============================================================
# SAMPLE REVIEWS
# ============================================================

SAMPLE_REVIEWS = {
    "Positive Example 1": "This movie was absolutely fantastic! The acting was superb and the story kept me engaged from start to finish. Highly recommend!",
    "Positive Example 2": "A masterpiece of modern cinema. The director's vision is clear and the execution is flawless. One of the best films I've seen this year.",
    "Negative Example 1": "Terrible waste of time. The plot made no sense and the acting was wooden. I wanted to leave after 30 minutes.",
    "Negative Example 2": "Disappointing and boring. The movie dragged on forever with no real payoff. Save your money and skip this one.",
    "Mixed/Neutral": "The movie had some good moments but also some weak points. The visuals were nice but the story felt incomplete.",
}


# ============================================================
# MAIN APP
# ============================================================

def main():
    """Main application."""
    
    # Header
    st.title("🎬 Movie Review Sentiment Analysis")
    st.markdown("*Powered by DistilBERT + PyTorch + AWS*")
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading model..."):
        tokenizer, model, is_custom = load_model()
    
    if is_custom:
        st.success("✅ Custom trained model loaded successfully!")
    else:
        st.info("ℹ️ Using pre-trained model (train your model first for custom results)")
    
    # Layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        # Sample selector
        sample_option = st.selectbox(
            "Try a sample review:",
            ["Custom Input"] + list(SAMPLE_REVIEWS.keys())
        )
        
        # Text input
        if sample_option == "Custom Input":
            user_input = st.text_area(
                "Enter your movie review:",
                height=150,
                placeholder="Type or paste a movie review here..."
            )
        else:
            user_input = st.text_area(
                "Enter your movie review:",
                value=SAMPLE_REVIEWS[sample_option],
                height=150,
            )
        
        # Predict button
        predict_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    
    with col2:
        st.header("📊 Results")
        
        if predict_btn and user_input:
            # Get prediction
            with st.spinner("Analyzing..."):
                result = predict_sentiment(user_input, tokenizer, model)
            
            # Display result
            if result["label"] == "Positive":
                st.success(f"## ✅ {result['label']}")
            else:
                st.error(f"## ❌ {result['label']}")
            
            # Gauge chart
            gauge_fig = create_gauge_chart(result["confidence"], result["label"])
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Probability bar
            st.markdown("#### Probability Distribution")
            bar_fig = create_probability_bar(result["negative_prob"], result["positive_prob"])
            st.plotly_chart(bar_fig, use_container_width=True)
            
        elif predict_btn:
            st.warning("Please enter some text to analyze.")
        else:
            st.info("Enter a review and click 'Analyze Sentiment' to see results.")
    
    # Divider
    st.markdown("---")
    
    # Batch Analysis Section
    st.header("📋 Batch Analysis")
    
    batch_text = st.text_area(
        "Enter multiple reviews (one per line):",
        height=150,
        placeholder="Review 1\nReview 2\nReview 3"
    )
    
    if st.button("🔍 Analyze All", use_container_width=False):
        if batch_text:
            reviews = [r.strip() for r in batch_text.split("\n") if r.strip()]
            
            if reviews:
                results = []
                progress = st.progress(0)
                
                for i, review in enumerate(reviews):
                    result = predict_sentiment(review, tokenizer, model)
                    results.append({
                        "Review": review[:100] + "..." if len(review) > 100 else review,
                        "Sentiment": result["label"],
                        "Confidence": f"{result['confidence']*100:.1f}%",
                        "Positive %": f"{result['positive_prob']*100:.1f}%",
                        "Negative %": f"{result['negative_prob']*100:.1f}%",
                    })
                    progress.progress((i + 1) / len(reviews))
                
                # Display results table
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Summary chart
                sentiment_counts = results_df["Sentiment"].value_counts()
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color_discrete_map={"Positive": "#00CC66", "Negative": "#FF4444"}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please enter at least one review.")
    
    # Footer
    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    
    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
    
    with tech_col1:
        st.markdown("**Model**")
        st.markdown("DistilBERT")
    
    with tech_col2:
        st.markdown("**Framework**")
        st.markdown("PyTorch")
    
    with tech_col3:
        st.markdown("**Cloud**")
        st.markdown("AWS S3")
    
    with tech_col4:
        st.markdown("**Dataset**")
        st.markdown("Rotten Tomatoes")
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built by Arnold Nemeth | 2026"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()