Sentiment Analysis — Movie Reviews
Production oriented demo that classifies movie reviews as Positive or Negative using DistilBERT, visualizes results with Plotly charts, and demonstrates cloud integration with AWS S3. The app is deployed as a public Hugging Face Space for instant sharing.

Live demo: https://huggingface.co/spaces/Arnie1980/sentiment-analysis-movies

Project Overview
Purpose  
Showcase an end to end NLP workflow: data ingestion, model inference, visualization, and cloud storage. This repo is designed for reproducibility and portfolio presentation.

Highlights

DistilBERT via Hugging Face Transformers

PyTorch inference

Gradio UI with Plotly visualizations

AWS S3 for dataset and artifact storage

Hosted on Hugging Face Spaces for instant sharing

Visuals
The README now embeds static images directly so visuals render reliably on GitHub. Place the images you uploaded into the assets/ folder with the filenames below.

Files to add to assets folder

assets/pipeline.jpeg — pipeline infographic and key metrics

assets/dashboard1.png — live dashboard screenshot 1

assets/gauge_bar.png — gauge and probability bar screenshot

Embedded images
  
  

Architecture and Process Workflow
User input flows through a compact, reproducible pipeline:

Kód
User Input (single or batch)
  ↓
Gradio UI
  ↓
Tokenization with DistilBERT tokenizer
  ↓
Model Inference with PyTorch
  ↓
Postprocessing softmax → probabilities
  ↓
Visualizations Plotly gauge and probability bars
  ↓
Optional persistence to AWS S3 for datasets and model artifacts
Quick Start Local
Clone repository

bash
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-movies.git
cd sentiment-analysis-movies
Create virtual environment and install

bash
python -m venv venv
source venv/bin/activate   # Linux macOS
venv\Scripts\activate      # Windows
pip install -r requirements.txt
Add the static images to assets/ as described above.

Run the app

bash
python app.py
Or visit the hosted demo: https://huggingface.co/spaces/Arnie1980/sentiment-analysis-movies

AWS Integration Notes
S3 stores datasets and model artifacts. Use boto3 for upload and download.

IAM: create a least privilege user for automation scripts.

Training: use Colab or SageMaker for larger runs; local CPU or Colab GPU is fine for demos.

Deployment: Hugging Face Spaces hosts the Gradio app; S3 provides persistent storage for artifacts.

Tech Stack
DistilBERT via Hugging Face Transformers

PyTorch for inference

Gradio for UI

Plotly for visualizations

AWS S3 for storage

Hugging Face Spaces for hosting

Project Structure
Kód
sentiment-analysis-movies/
├── app.py
├── requirements.txt
├── Procfile
├── scripts/
│   └── export_visuals.py
├── assets/
│   ├── pipeline.jpeg
│   ├── dashboard1.png
│   └── gauge_bar.png
├── src/
│   ├── train_model.py
│   └── inference.py
├── data/
└── README.md
Contact
Project by Arnold Nemeth. Feedback and PRs welcome.

LinkedIn Post (English, visually engaging)
🎬 New demo live — Movie Review Sentiment Analyzer  
Try it now: https://huggingface.co/spaces/Arnie1980/sentiment-analysis-movies

What I built  
A compact, production‑oriented NLP demo that classifies movie reviews as Positive or Negative and visualizes results with an interactive confidence gauge and probability chart. The app supports single review inference and batch processing, and it’s designed to be portfolio ready.

Quick facts

Model DistilBERT via Hugging Face Transformers

Inference PyTorch for efficient execution

UI Gradio for a fast, shareable interface

Visuals Plotly gauge and probability bars

Cloud AWS S3 for datasets and model artifacts; hosted on Hugging Face Spaces

Key metrics from the demo

Accuracy 85% • Precision 86% • Recall 84% • F1 85%

How it works

Input text (single or multiple reviews)

Tokenize with DistilBERT tokenizer

Run model inference and convert logits to probabilities with softmax

Visualize predicted label and confidence with Plotly

Persist artifacts and data to AWS S3 for reproducibility

Why this matters  
This project demonstrates practical skills across the ML lifecycle: transfer learning, inference engineering, cloud integration, and visualization — all in a compact, reproducible demo you can run locally or in the cloud.

See it live  
🔗 https://huggingface.co/spaces/Arnie1980/sentiment-analysis-movies

Want the code or a quick walkthrough?  
I can share a short guide to reproduce the demo, export static visuals for documentation, or extend the app with explainability and multi‑language support.
