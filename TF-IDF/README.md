
# TF-IDF Implementation 

This project implements the TF-IDF (Term Frequency–Inverse Document Frequency) algorithm **from scratch** in Python and applies it to multiple PDF documents. It was developed as part of the AI414 course – *Processing of Formal and Natural Languages* at Cairo University.

## 📝 Description

- Extracts text from a set of PDF files.
- Preprocesses the text (case normalization, punctuation removal, tokenization).
- Computes TF (Term Frequency) and IDF (Inverse Document Frequency).
- Calculates TF-IDF values for all words.
- Displays the top TF-IDF words per document.
- Encodes documents into TF-IDF vector form for further analysis.

## 📂 Folder Contents

- `TFIDF.py` – main Python script.
- `1.pdf`, `2.pdf`, `3.pdf` – example documents used for testing.
- `README.md` – this file.

## 🧠 Key Concepts

- **TF (Term Frequency)**: Measures how frequently a term appears in a document.
- **IDF (Inverse Document Frequency)**: Measures how important a term is, based on how many documents it appears in.
- **TF-IDF**: Combines both metrics to highlight important words in context.

## 🔍 Preprocessing Steps

- Lowercasing
- Punctuation removal
- Tokenization using NLTK
- (Optional) Stopword removal *(commented in code)*

## 🧪 Example Output

- Top N words with the highest TF-IDF score in each document
- Table of TF-IDF encoded vectors

## 🚀 How to Run

1. Place your PDF files (e.g., `1.pdf`, `2.pdf`, `3.pdf`) in the same folder as the script.
2. Make sure required libraries are installed:
   ```bash
   pip install numpy PyPDF2 tabulate nltk
