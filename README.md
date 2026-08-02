# 🎬 IMDB Review Classifier using RNN

A deep learning web application built with **TensorFlow / Keras** and **Streamlit** that performs sentiment analysis on movie reviews using a **Simple Recurrent Neural Network (RNN)** model trained on the IMDB dataset.

---

## 📌 Project Overview

This project classifies text input into two sentiment categories:
- 🟢 **Positive Sentiment** (Score > 0.5)
- 🔴 **Negative Sentiment** (Score ≤ 0.5)

It features an end-to-end Natural Language Processing (NLP) pipeline:
1. **Preprocessing & Tokenization:** Text cleaning, sequence mapping with IMDB vocabulary, and uniform padding.
2. **Deep Learning Model:** An Embedding layer paired with a SimpleRNN layer (`tanh` activation) and a Sigmoid Dense output layer.
3. **Interactive Web Interface:** A modern UI designed using Streamlit for live text evaluation.

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python
- **Deep Learning Framework:** TensorFlow 2.x / Keras
- **Web App Framework:** Streamlit
- **Data & Numeric Processing:** NumPy, Pandas

---

## 📂 Project Structure

```text
imdb-review-classifier-rnn/
│
├── main.py                  # Streamlit web application
├── simple_rnn_imdb.h5       # Trained Keras model
├── requirements.txt         # Required Python dependencies
├── simplernn.ipynb          # Jupyter notebook for model training
├── prediction.ipynb         # Jupyter notebook for model testing
└── README.md                # Project documentation
