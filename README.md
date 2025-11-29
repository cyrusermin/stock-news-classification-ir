# Stock Market News Classification & Information Retrieval

This project builds an NLP-based system that predicts next-day stock
movement using news articles and retrieves the most relevant articles
for any financial search query. It combines IR, ML, and financial data engineering.

## Features
- Fetch historical stock prices using yfinance
- Fetch and parse real news articles using RSS + newspaper3k
- Create labels using next-day price movement
- NLP text processing with TF-IDF and SBERT
- Machine learning models: Logistic Regression, Random Forest, XGBoost
- Information Retrieval system using TF-IDF + Cosine Similarity
- End-to-end prediction pipeline

 ## Installation
pip install -r requirements.txt
!pip install yfinance feedparser newspaper3k scikit-learn pandas numpy tqdm xgboost sentence-transformers

 ## How to Run

1. Open `notebook.ipynb` in Google Colab.
2. Run all cells from top to bottom.
3. Replace ticker and company name if needed.
4. Use `query_and_predict("your query", top_k=5)` to retrieve relevant articles.

 ## Algorithms Used
### Machine Learning
- Logistic Regression
- Random Forest
- XGBoost

### NLP & IR
- TF-IDF Vectorizer
- Cosine Similarity
- SBERT Embeddings (optional)

### Finance
- Next-day return calculation
- Supervised learning target creation

## Results
- TF-IDF baseline accuracy: ~65-70%
- XGBoost ROC-AUC: ~0.70+
- IR system retrieves top-k relevant news effectively

