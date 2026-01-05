Sentiment Analysis — Movie Reviews
Production‑oriented demo that classifies movie reviews as Positive or Negative using DistilBERT, visualizes results with Plotly charts, and demonstrates cloud integration with AWS S3. The app is deployed as a public Hugging Face Space for instant sharing.

Live demo: https://huggingface.co/spaces/Arnie1980/sentiment-analysis-movies

Project summary
Goal: End‑to‑end NLP demo showing tokenization, inference, visualization, and cloud storage.

Model: DistilBERT (Hugging Face Transformers)

Inference: PyTorch

UI: Gradio (shareable web UI)

Visuals: Plotly gauge + probability bar chart (saved as static images for README compatibility)

Cloud: AWS S3 for datasets and model artifacts; Hugging Face Spaces for hosting

Why the charts in README were failing
GitHub README files do not render interactive Plotly charts. To make visuals visible in the README we:

Render Plotly charts to static PNG/SVG during your build or locally, and commit them to assets/ (or generate them in CI).

Reference those static images in the README so they display reliably on GitHub and in other viewers.

Below are the exact steps and code snippets to generate and include static visuals.

How to generate static Plotly images (recommended)
Install the renderer dependencies (Kaleido is recommended for static export):

bash
pip install plotly kaleido
Add a small script to export the figures used in the app:

python
# scripts/export_visuals.py
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

os.makedirs("assets", exist_ok=True)

# Example gauge
def save_gauge(confidence=0.931, label="Positive", out="assets/gauge.png"):
    color = "#00CC66" if label == "Positive" else "#FF4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={"text": f"Confidence ({label})"},
        number={"suffix": "%"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}}
    ))
    fig.update_layout(height=300)
    fig.write_image(out, scale=2)  # PNG output

# Example bar chart
def save_bar(neg=0.069, pos=0.931, out="assets/probabilities.png"):
    df = pd.DataFrame({
        "Sentiment": ["Negative", "Positive"],
        "Probability": [neg * 100, pos * 100]
    })
    fig = px.bar(df, x="Sentiment", y="Probability", color="Sentiment",
                 color_discrete_map={"Positive": "#00CC66", "Negative": "#FF4444"},
                 text="Probability")
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    fig.update_layout(yaxis_range=[0, 100], height=300)
    fig.write_image(out, scale=2)

if __name__ == "__main__":
    save_gauge()
    save_bar()
Run the script and commit the generated images:

bash
python scripts/export_visuals.py
git add assets/gauge.png assets/probabilities.png
git commit -m "Add static visuals for README"
git push
Then reference them in the README:

markdown
![Confidence Gauge](assets/gauge.png)
![Probability Distribution](assets/probabilities.png)
Architecture and process workflow
mermaid
flowchart LR
  A[User Input: single or batch] --> B[Gradio UI]
  B --> C[Tokenizer (DistilBERT)]
  C --> D[Model Inference (PyTorch)]
  D --> E[Postprocess: softmax → probabilities]
  E --> F[Visuals: Plotly gauge + bar chart]
  E --> G[Results table / CSV]
  G --> H[AWS S3 (optional): store results & artifacts]
  F --> I[Hugging Face Spaces: hosted UI]
Quick start (local)
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
Generate static visuals (optional but recommended for README)

bash
python scripts/export_visuals.py
Run the app

bash
python app.py
Or visit the hosted demo: https://huggingface.co/spaces/Arnie1980/sentiment-analysis-movies

AWS integration (concise)
S3: store datasets and model artifacts. Use boto3 to upload/download CSVs and model folders.

IAM: create a least‑privilege user for automation scripts.

Training: use Colab or SageMaker for larger runs; for demo purposes, local or Colab is sufficient.

Deployment: Hugging Face Spaces hosts the Gradio app; S3 provides persistent storage.

Project structure
Kód
sentiment-analysis-movies/
├── app.py
├── requirements.txt
├── Procfile
├── scripts/
│   └── export_visuals.py
├── assets/
│   ├── gauge.png
│   └── probabilities.png
├── src/
│   ├── train_model.py
│   └── inference.py
├── data/
└── README.md
Notes & best practices
Commit static visuals to assets/ so the README displays correctly on GitHub.

For reproducible CI builds, add a step that runs scripts/export_visuals.py and uploads the generated images to the repo or artifacts.

Keep credentials out of the repo; use environment variables or CI secrets for AWS keys.

Contact
Project by Arnold Nemeth. Feedback and PRs welcome.
