import subprocess
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import gensim.downloader as api
import spacy
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from sklearn.manifold import TSNE
import os


# Set up the app
st.set_page_config(page_title="Text Vectorization Visualizer", layout="wide")
st.title("Text Vectorization Visualization")

# Initialize spaCy
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")
    return nlp


nlp = load_spacy_model()



# Sidebar for method selection
method = st.sidebar.selectbox(
    "Select Vectorization Method",
    ["Bag of Words", "N-Grams", "TF-IDF", "Word2Vec", "GloVe", "fastText"]
)

# Text input
text_input = st.text_area("Enter your text here:", "Natural language processing is a fascinating field of study. Machine learning helps computers understand human language.")

# Text preprocessing options
st.sidebar.subheader("Text Preprocessing Options")
preprocessing_params = {
    "lowercase": st.sidebar.checkbox("Lowercase", True),
    "remove_stopwords": st.sidebar.checkbox("Remove Stopwords", False),
    "lemmatization": st.sidebar.checkbox("Lemmatization", False),
    "stemming": st.sidebar.checkbox("Stemming", False),
    "remove_punctuation": st.sidebar.checkbox("Remove Punctuation", False),
    "remove_numbers": st.sidebar.checkbox("Remove Numbers", False),
    "normalize_text": st.sidebar.checkbox("Normalize Text", False)
}

# Tokenization options
tokenization_method = st.sidebar.selectbox(
    "Tokenization Method",
    ["Word Tokenization", "BPE", "WordPiece", "SentencePiece"],
    index=0
)

# Additional parameters based on method
params = {}
if method in ["Bag of Words", "N-Grams", "TF-IDF"]:
    if method == "N-Grams":
        params["ngram_range"] = (
            st.sidebar.slider("Min N-Gram", 1, 5, 1),
            st.sidebar.slider("Max N-Gram", 1, 5, 2)
        )
elif method in ["Word2Vec", "GloVe", "fastText"]:
    params["vector_size"] = st.sidebar.slider("Vector Size", 2, 300, 50)
    if method == "Word2Vec":
        params["window"] = st.sidebar.slider("Window Size", 1, 10, 5)
        params["min_count"] = st.sidebar.slider("Min Count", 1, 10, 1)

# Load pretrained models for visualization with lazy loading
@st.cache_resource(show_spinner=False)
def load_pretrained(model_name):
    try:
        st.warning(f"Loading {model_name} model (this may take several minutes for the first time)...")
        if model_name == "GloVe":
            model = api.load("glove-wiki-gigaword-50")
        elif model_name == "fastText":
            model = api.load("fasttext-wiki-news-subwords-300")
        st.success(f"{model_name} model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading {model_name}: {str(e)}")
        return None

pretrained_models = {}

def preprocess_text(text, params):
    """Apply all selected preprocessing steps to the text using spaCy"""
    doc = nlp(text)
    processed_tokens = []

    for token in doc:
        token_text = token.text.lower() if params["lowercase"] else token.text

        if params["remove_stopwords"] and token.is_stop:
            continue
        if params["remove_punctuation"] and token.is_punct:
            continue
        if params["remove_numbers"] and token.like_num:
            continue
        if params["lemmatization"]:
            token_text = token.lemma_

        processed_tokens.append(token_text)

    processed_text = " ".join(processed_tokens)

    if processed_text != text:
        st.subheader("Text After Preprocessing")
        st.text_area("Processed Text", processed_text, height=150)

    return processed_text

def tokenize_text(text, method):
    """Tokenize text using the selected method"""
    try:
        if method == "Word Tokenization":
            doc = nlp(text)
            tokens = [token.text for token in doc]
            st.subheader("Word Tokenization Results")
            st.write(tokens)
            return tokens

        tokenizer = None
        trainer = None

        if method == "BPE":
            tokenizer = Tokenizer(models.BPE())
            trainer = trainers.BpeTrainer(
                vocab_size=10000,
                min_frequency=2,
                special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
            )
        elif method == "WordPiece":
            tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
            trainer = trainers.WordPieceTrainer(
                vocab_size=10000,
                min_frequency=2,
                special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
            )
        else:  # SentencePiece
            tokenizer = Tokenizer(models.Unigram())
            trainer = trainers.UnigramTrainer(
                vocab_size=10000,
                special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
            )

        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        train_file = "train_text.txt"

        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training file {train_file} not found")

        tokenizer.train([train_file], trainer)
        output = tokenizer.encode(text)

        st.subheader(f"{method} Tokenization Results")
        st.write("Tokens:", output.tokens)
        st.write("Token IDs:", output.ids)

        vocab = tokenizer.get_vocab()
        st.write(f"Vocabulary size: {len(vocab)}")
        st.write("Sample vocabulary:", list(vocab.keys())[:20])

        return output.tokens

    except Exception as e:
        st.error(f"Error in tokenization ({method}): {str(e)}")
        doc = nlp(text)
        return [token.text for token in doc]

def animate_vectorization(method, text, params, preprocessing_params, tokenization_method):

    processed_text = preprocess_text(text, preprocessing_params)
    tokens = tokenize_text(processed_text, tokenization_method)
    sentences = [sentence.strip() for sentence in processed_text.split('.') if sentence.strip()]
    words = [word for sentence in sentences for word in sentence.split()]
    unique_words = list(set(words))

    if method in ["Bag of Words", "N-Grams", "TF-IDF"]:
        # ... [previous Bag of Words/N-Grams/TF-IDF code remains exactly the same] ...
        pass

    elif method in ["Word2Vec", "GloVe", "fastText"]:
        # Load or train word vectors
        if method in ["GloVe", "fastText"] and method not in pretrained_models:
            with st.spinner(f"Loading {method} model (this may take several minutes for the first time)..."):
                pretrained_models[method] = load_pretrained(method)

        if method == "Word2Vec":
            model = Word2Vec(
                sentences=[sentence.split() for sentence in sentences],
                vector_size=params["vector_size"],
                window=params["window"],
                min_count=params["min_count"],
                workers=4
            )
            word_vectors = model.wv
        else:
            word_vectors = pretrained_models.get(method)
            if word_vectors is None:
                st.warning(f"Using a simple Word2Vec model as fallback (pretrained {method} not available)")
                model = Word2Vec(
                    sentences=[sentence.split() for sentence in sentences],
                    vector_size=params["vector_size"],
                    window=5,
                    min_count=1,
                    workers=4
                )
                word_vectors = model.wv

        # Get vectors with consistent size
        vectors = {}
        vector_size = params["vector_size"]
        for word in unique_words:
            try:
                vec = word_vectors[word]
                if len(vec) != vector_size:
                    vec = vec[:vector_size] if len(vec) > vector_size else np.pad(vec, (0, vector_size - len(vec)))
                vectors[word] = vec
            except:
                vectors[word] = np.random.randn(vector_size)

        vectors_array = np.array([vectors[word] for word in unique_words])
        n_words = len(unique_words)

        # Handle dimensionality reduction with proper error checking
        if n_words <= 10:
            st.subheader("Word Vectors (Raw)")
            vector_df = pd.DataFrame(
                vectors_array,
                index=unique_words,
                columns=[f"Dim {i}" for i in range(vector_size)]
            )
            st.dataframe(vector_df.style.background_gradient(cmap='viridis'))
            
            if n_words < 5:
                st.warning("Visualization may not be meaningful with very few words")
        else:
            # Dynamic t-SNE parameters
            perplexity = min(30, n_words - 1)
            
            try:
                # 2D visualization
                tsne_2d = TSNE(
                    n_components=2,
                    perplexity=perplexity,
                    random_state=42,
                    init='pca',
                    learning_rate=200
                )
                vectors_2d = tsne_2d.fit_transform(vectors_array)
                
                # 3D visualization if enough words and dimensions
                if n_words > 10 and params["vector_size"] > 2:
                    st.subheader("Word Embeddings (3D)")
                    tsne_3d = TSNE(
                        n_components=3,
                        perplexity=perplexity,
                        random_state=42,
                        init='pca'
                    )
                    vectors_3d = tsne_3d.fit_transform(vectors_array)
                    
                    fig_3d = go.Figure(data=[go.Scatter3d(
                        x=vectors_3d[:, 0],
                        y=vectors_3d[:, 1],
                        z=vectors_3d[:, 2],
                        mode='markers+text',
                        text=unique_words,
                        textposition="top center",
                        marker=dict(
                            size=8,
                            color=vectors_3d[:, 0],
                            colorscale='Viridis',
                            opacity=0.8
                        )
                    )])
                    fig_3d.update_layout(
                        title=f"{method} Word Embeddings (3D)",
                        scene=dict(
                            xaxis_title='Dimension 1',
                            yaxis_title='Dimension 2',
                            zaxis_title='Dimension 3'
                        )
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)

                # 2D plot
                st.subheader("Word Embeddings (2D)")
                fig_2d = go.Figure(data=[go.Scatter(
                    x=vectors_2d[:, 0],
                    y=vectors_2d[:, 1],
                    mode='markers+text',
                    text=unique_words,
                    textposition="top center",
                    marker=dict(
                        size=12,
                        color=np.arange(len(unique_words)),
                        colorscale='Rainbow',
                        opacity=0.8
                    )
                )])
                fig_2d.update_layout(
                    title=f"{method} Word Embeddings (2D)",
                    xaxis_title='Dimension 1',
                    yaxis_title='Dimension 2'
                )
                st.plotly_chart(fig_2d, use_container_width=True)

            except Exception as e:
                st.error(f"Error in dimensionality reduction: {str(e)}")
                # Fallback to PCA
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                vectors_2d = pca.fit_transform(vectors_array)
                st.warning("Used PCA instead of t-SNE due to error")
                
                # Plot PCA results
                fig = go.Figure(data=[go.Scatter(
                    x=vectors_2d[:, 0],
                    y=vectors_2d[:, 1],
                    mode='markers+text',
                    text=unique_words,
                    textposition="top center"
                )])
                fig.update_layout(title="Word Embeddings (PCA)")
                st.plotly_chart(fig, use_container_width=True)

# Run the visualization
if st.button("Visualize"):
    with st.spinner(f"Creating {method} visualization..."):
        animate_vectorization(method, text_input, params, preprocessing_params, tokenization_method)
