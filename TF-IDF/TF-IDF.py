import numpy as np
import PyPDF2
from collections import Counter
from tabulate import tabulate
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + " "
    return text.strip()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    # Do I use removal of the stop words or not

    # stop_words = set(stopwords.words('english'))
    # tokens = [word for word in tokens if word not in stop_words]
    return set(tokens)

#using Normalized Term Frequency (TF)
def compute_tf(document_tokens):
    word_counts = Counter(document_tokens)
    total_words = len(document_tokens)
    tf = {word: count / total_words for word, count in word_counts.items()}
    return tf

# #using logarithmically Scaled Term Frequency
# def compute_tf_log(document_tokens):
#     word_counts = Counter(document_tokens)
#     return {word:np.log(count+1) for word, count in word_counts.items() if count > 0}


# #using Augmented Term Frequency
# def compute_tf_augmented(document_tokens):
#     word_counts = Counter(document_tokens)
#     max_count = max(word_counts.values(), default=1)  # Avoid division by zero
#     return {word: 0.5 + 0.5 * (count / max_count) for word, count in word_counts.items()}


def compute_idf(documents):
    N = len(documents)
    word_doc_count = Counter()

    for doc in documents:

        unique_words = set(doc)


        for word in unique_words:
            word_doc_count[word] += 1

    idf = {word: np.log((N) / (count + 1)) + 1 for word, count in word_doc_count.items()}
    return idf


def compute_tfidf(documents):
    processed_docs = [preprocess_text(doc) for doc in documents]
    idf = compute_idf(processed_docs)

    tfidf_values = []
    tfidf_vectors = []
    vocabulary = sorted(set(word for doc in processed_docs for word in doc))  # Sorted for consistent ordering

    for doc in processed_docs:
        tf = compute_tf(doc)
        tfidf = {word: tf.get(word, 0) * idf[word] for word in vocabulary}
        tfidf_values.append(tfidf)
        tfidf_vectors.append([round(tfidf[word], 4) for word in vocabulary])

    return tfidf_values, tfidf_vectors, vocabulary


def display_top_words(tfidf_values, top_n=2):
    for i, tfidf in enumerate(tfidf_values):
        sorted_words = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:top_n]
        table_data = [[word, f"{score:.4f}"] for word, score in sorted_words]
        print(f"\nDocument {i + 1} Top {top_n} Words:\n")
        print(tabulate(table_data, headers=["Word", "TF-IDF Score"], tablefmt="grid"))


def encode_documents_as_vectors(tfidf_vectors, vocabulary):
    print("\nTF-IDF Encoded Vectors:\n")
    header = ["Document"] + vocabulary
    rows = [[f"Doc {i + 1}"] + vector for i, vector in enumerate(tfidf_vectors)]
    print(tabulate(rows, headers=header, tablefmt="grid"))


pdf_files = ["1.pdf", "2.pdf", "3.pdf"]


# Extract text from PDFs
document = [extract_text_from_pdf(pdf) for pdf in pdf_files]

# Compute TF-IDF
tfidf_value, tfidf_vector, vocabulary = compute_tfidf(document)

# Display top words
display_top_words(tfidf_value)

# Display TF-IDF encoded document vectors
encode_documents_as_vectors(tfidf_vector, vocabulary)
