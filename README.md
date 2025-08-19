# LLM_Project
Financial News Sentiment Pipeline using FinBERT & GPT-4

This project builds a modular NLP pipeline to extract sentiment signals from financial news articles.  
It combines traditional NLP (triplet extraction, clustering) with modern LLMs (FinBERT, GPT-4) to generate structured company-level sentiment insights.

---

## Features
- News data collection from multiple financial sources  
- Preprocessing and NLP enrichment (NER, sentiment, triplets)  
- Clustering with embeddings (UMAP + HDBSCAN)  
- Sentiment generation with **FinBERT**  
- Sentiment generation with **GPT-4**  
- JSON outputs with company tickers, sentiment, and justifications  

---

## Installation (Windows fresh setup)

1. **Install Python 3.10+**  
   👉 [Download from python.org](https://www.python.org/downloads/)

2. **Open PowerShell and create a virtual environment**
   ```powershell
   cd "C:\Users\YourName\Documents\LLM_Project"
   python -m venv venv
   .\venv\Scripts\Activate.ps1

If you get a PowerShell policy error:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

3. **Install dependencies**
   ```powershell
   python -m pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   pip install --upgrade spacy thinc packaging
   python -m spacy download en_core_web_sm
   python -m textblob.download_corpora
---
## OpenAI API Key (for GPT step)

To use GPT-4 you need an OpenAI API key:

1. **Get your key from the OpenAI website** https://platform.openai.com/docs/overview
2. **Add credits to your account (the GPT API is paid)**
3. **Set your key in PowerShell before running the pipeline:**

   setx OPENAI_API_KEY "your_api_key_here"
---

## Running the Pipeline

**To run the full pipeline from start to finish:**

python run_pipeline.py













