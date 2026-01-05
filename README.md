Sentiment Analysis — Movie Reviews
Lightweight demo that classifies movie reviews as Positive or Negative using a DistilBERT model and provides interactive visualizations (confidence gauge and probability bar chart). The app is packaged as a Hugging Face Space so anyone can try it in the browser.

Key features
Model: DistilBERT fine-tuned for sentiment (Hugging Face model).

Inference: Fast, CPU-friendly PyTorch inference.

Visuals: Plotly gauge for confidence and bar chart for probability breakdown.

UI: Gradio-based web interface for single and batch review analysis.

Deployment: Hosted on Hugging Face Spaces — no local install required to try the demo.

Repository contents
app.py — Gradio app that loads the model, runs inference, and renders Plotly visualizations.

requirements.txt — Python dependencies needed to run the app.

Procfile (optional) — explicit startup command for some hosting platforms.

README.md — this file.

Quick start (local)
Clone the repo:

bash
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-movies.git
cd sentiment-analysis-movies
Create and activate a virtual environment:

bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
Install dependencies:

bash
pip install -r requirements.txt
Run the app locally:

bash
python app.py
The Gradio app will start and print a local URL.

How it works — high level
Input — user enters a single review or multiple reviews (one per line).

Tokenization — text is tokenized with the DistilBERT tokenizer.

Inference — model predicts logits; softmax converts logits to probabilities.

Output — label (Positive/Negative), confidence score, Plotly gauge, and probability bar chart.

Batch mode — processes multiple reviews and shows a results table and distribution chart.

Deployment
Hosted on Hugging Face Spaces: try it live at https://huggingface.co/spaces/Arnie1980/sentiment-analysis-movies.

To deploy your own copy, push the repo to a new Space or use Streamlit/Gradio hosting with the included files.

Dependencies
Minimum required packages (see requirements.txt):

gradio

torch

transformers

pandas

plotly

Notes & tips
The demo uses a pre-trained DistilBERT model for speed and low resource usage. For production, consider larger models or additional fine-tuning on domain-specific data.

For faster experimentation, run on Google Colab or a GPU-enabled environment.

If you want to extend the demo: add sentiment explanation (saliency), support more languages, or add a small dataset upload + retrain flow.

License
Open-source friendly. Add your preferred license file (e.g., MIT) to the repo.

Contact
Project by Arnold Nemeth — improvements and PRs welcome.
