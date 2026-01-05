Sentiment Analysis — Movie Reviews
Professional, shareable demo that classifies movie reviews as Positive or Negative using a DistilBERT model, visualizes results with interactive Plotly charts, and is deployed on AWS-backed infrastructure and Hugging Face Spaces for instant sharing.

Project Overview
Purpose: Demonstrate end-to-end NLP skills: data handling, transfer learning, inference, visualization, and cloud deployment.
Audience: Hiring managers, ML engineers, and portfolio reviewers who want to see practical experience with modern NLP tooling and cloud integration.
Live demo: https://huggingface.co/spaces/Arnie1980/sentiment-analysis-movies

Key Features
Model: DistilBERT fine-tuned for sentiment classification (Hugging Face Transformers).

Inference: PyTorch-based, CPU-friendly inference pipeline.

Visualizations: Plotly gauge for confidence and bar chart for probability breakdown.

UI: Gradio interface for single and batch review analysis.

Cloud: Data and artifacts stored on AWS S3; app hosted on Hugging Face Spaces.

Reproducibility: requirements.txt and Procfile included for consistent deployment.

Architecture and Process Workflow
High level flow

Kód
User Input (single or batch) 
      ↓
Gradio UI
      ↓
Tokenization (DistilBERT tokenizer)
      ↓
Model Inference (DistilBERT via PyTorch)
      ↓
Postprocessing (softmax → probabilities)
      ↓
Visualizations (Plotly gauge + bar chart) + Results table
      ↓
Optional: Save logs / results to AWS S3
Components

Data: IMDB or Rotten Tomatoes datasets (Hugging Face Datasets or CSV).

Training: Fine-tune DistilBERT locally or on Colab; save model artifacts.

Storage: Upload datasets and model artifacts to AWS S3 for persistence.

Serving: Gradio app loads model from local path or S3 and serves inference.

Hosting: Hugging Face Spaces for public demo and sharing.

Visualizations
What the app shows

Confidence Gauge — single-value indicator showing model confidence for the predicted label.

Probability Bar Chart — side-by-side bars for Positive and Negative probabilities with percentage labels.

Batch Results Table — sentiment and confidence for each review and a pie chart for distribution.

Plotly snippets

python
# Gauge chart
import plotly.graph_objects as go
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=confidence*100,
    title={"text": f"Confidence ({label})"},
    number={"suffix": "%"},
    gauge={"axis": {"range": [0,100]}, "bar": {"color": color}}
))
python
# Probability bar chart
import plotly.express as px
df = pd.DataFrame({"Sentiment": ["Negative","Positive"], "Probability": [neg*100, pos*100]})
fig = px.bar(df, x="Sentiment", y="Probability", color="Sentiment",
             color_discrete_map={"Positive":"#00CC66","Negative":"#FF4444"})
fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
Quick Start
Clone

bash
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-movies.git
cd sentiment-analysis-movies
Environment

bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
pip install -r requirements.txt
Run locally

bash
python app.py
Or visit the hosted demo: https://huggingface.co/spaces/Arnie1980/sentiment-analysis-movies

AWS Integration and Deployment Notes
S3 stores datasets and model artifacts for reproducible runs. Use boto3 to upload/download CSVs and model folders.

IAM: create a least-privilege user with S3 access for automation scripts.

Optional: use SageMaker or an EC2 instance for larger training runs; for quick demos, Colab or local CPU/GPU is sufficient.

Hosting: Hugging Face Spaces serves the Gradio app; S3 provides persistent storage for datasets and saved models.

Tech Stack
Layer	Technology
Model	DistilBERT (Hugging Face Transformers)
Inference	PyTorch
UI	Gradio
Visualizations	Plotly
Storage	AWS S3
Hosting	Hugging Face Spaces
Project Structure
Kód
sentiment-analysis-movies/
├── app.py
├── requirements.txt
├── Procfile
├── src/
│   ├── train_model.py
│   └── inference.py
├── data/
│   └── *.csv
└── README.md
How to extend
Add explainability (LIME, SHAP) to highlight tokens that drive predictions.

Add multi-language support and additional datasets.

Add CI/CD to retrain and redeploy when new data is added to S3.

Contact
Project by Arnold Nemeth. Feedback, PRs, and collaboration welcome.
