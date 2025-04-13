# 🧠 Text Vectorization Visualizer

[![Streamlit App](https://img.shields.io/badge/🔗%20Live%20App-Streamlit-ff4b4b?logo=streamlit)](https://text-vectorization-visualizer.streamlit.app/)  
[![Read the Blog](https://img.shields.io/badge/📘%20Zero%20to%20Hero%20Guide-Hashnode-blue)](https://data-science-notes.hashnode.dev/zero-to-hero-text-vectorization)

## 📌 About the Project

**Text Vectorization Visualizer** is an interactive Streamlit app that helps you **understand, visualize, and compare** different text vectorization techniques in NLP, including:

- **Bag of Words (BoW)**
- **TF-IDF (Term Frequency–Inverse Document Frequency)**
- **Word Embeddings (Word2Vec, GloVe , FastText)**
- **Tokenization , Lemmatization , Stemming**

This project is part of the **Zero to Hero** NLP blog series aimed at demystifying core NLP concepts with hands-on demos and intuitive explanations.

---

## 🚀 Try the App

🔗 [Click here to launch the live demo](https://text-vectorization-visualizer.streamlit.app/)

Enter your own text and see how different vectorizers convert it into numerical representations that can be used in Machine Learning & Deep Learning models.

---

## 📖 Read the Blog

Get the full theoretical background and code walkthrough in the detailed blog post:  
📘 [Zero to Hero — Text Vectorization Techniques in NLP](https://data-science-notes.hashnode.dev/zero-to-hero-text-vectorization)

Covers:
- Theory and intuition
- Sparse vs dense vectors
- Real-world use cases
- Code examples using `sklearn`, `gensim`, and `spaCy`

---

## 🧑‍💻 Features

- ✅ Real-time text vectorization
- ✅ Side-by-side comparison of BoW and TF-IDF
- ✅ View raw matrix outputs and feature names
- ✅ Gensim-based Word2Vec support

---

## 🛠️ Tech Stack

- [Python 3.12+](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [spaCy](https://spacy.io/)
- [Gensim](https://radimrehurek.com/gensim/)
- [Matplotlib / Seaborn](https://seaborn.pydata.org/) *(for visualizations)*

---

## 🧰 Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/text-vectorization-visualizer.git
cd text-vectorization-visualizer

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the app
streamlit run main.py
