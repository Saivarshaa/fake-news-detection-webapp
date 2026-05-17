# Fake News Detection Web App

A simple and effective web app built with **Streamlit** that detects whether a news article is real or fake using a **Support Vector Machine (SVM)** classifier trained on TF-IDF features. The model achieves **99% accuracy** on the test set.

The **attached notebook** will walk you through the data preprocessing steps and model design and evaluation. 

---

## Project Structure

This repository contains the following:

- `app.py` – The main Streamlit application  
- `fake_news_model.pkl` – The trained machine learning model (SVM)  
- `tfidf_vectorizer.pkl` – The TF-IDF vectorizer fitted on the training data  
- `requirements.txt` – List of Python dependencies  
- `Fake News Prediction Work Notebook.ipynb` – Jupyter notebook used for data cleaning, EDA, model training and evaluation  
---

## Dataset

This project uses this [Kaggle Dataset](https://www.kaggle.com/datasets/saratchendra/fake-news). The dataset contains thousands of news articles labeled as either **real** or **fake**. It includes fields like title, author, and content. The focus was on the `content` column, which underwent preprocessing before feature extraction.

---

## Model Details

- **Vectorizer:** TF-IDF with `max_features=5000`  
- **Classifier:** Support Vector Machine (SVM) with linear kernel  
- **Accuracy:** 99% on the held-out test set  
- **Preprocessing Steps:**
  - Lowercasing  
  - Removal of non-alphabetic characters  
  - Stopword removal  
  - Stemming (PorterStemmer)

---

## Live Demo

👉 [Click here to try the app](https://fake-news-detection-app-adam.streamlit.app/)

---

## How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/fake-news-detection-streamlit-app.git
   cd fake-news-detection-streamlit-app
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the app:
   ```bash
   streamlit run app.py

## Deployment (Streamlit Cloud)

To deploy this app to the web:

1. Push this repo to your GitHub account
2. Go to https://streamlit.io/cloud
3. Sign in and create a new app
4. Choose the repo and set app.py as the entry point
5. Click Deploy

#### Author
Built by Muhammad Adam Umar as part of his data science portfolio.
Connect on [LinkedIn](https://www.linkedin.com/in/muhammad-adam-umar-26baaa2b5/)
[Github](https://github.com/MAdamUmar/)
