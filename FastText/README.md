
# FastText Embedding and Visualization on Yelp Tips Dataset

This project processes a portion of the [Yelp Academic Dataset](https://www.yelp.com/dataset) and builds word embeddings using **FastText**, then visualizes semantic relationships between words.

## 📌 Overview

The notebook performs the following tasks:

1. **Text Preprocessing**: Cleans and tokenizes the `text` field from `yelp_academic_dataset_tip.json`.
2. **Word Embedding**: Trains a FastText model on the preprocessed corpus.
3. **Similarity Analysis**: Finds words similar and opposite to a given word (e.g., `"food"`).
4. **Visualization**: Uses t-SNE to project word embeddings into 2D space for visual exploration.

## 🧪 Requirements

Install dependencies with:

```bash
pip install nltk gensim matplotlib scikit-learn
```

Download required NLTK data:

```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
```

## 📂 Dataset

Make sure you have the file `yelp_academic_dataset_tip.json` in the same directory. You can download it from the official [Yelp Dataset page](https://www.yelp.com/dataset).

The notebook reads and processes the first 500 lines of this JSON file.

## ⚙️ How It Works

1. **Preprocessing**:
   - Lowercasing
   - Removing punctuation and digits
   - Removing stopwords

2. **FastText Training**:
   ```python
   model = FastText(sentences=corp, vector_size=300, window=2, workers=8)
   model.save("yelp_fasttext.bin")
   ```

3. **Similarity Example**:
   ```python
   model.wv.most_similar("food", topn=10)
   model.wv.most_similar(negative=["food"], topn=10)
   ```

4. **Visualization**:
   A custom function uses t-SNE to reduce dimensionality and `matplotlib` to plot related words.

## 📈 Output

- `yelp_fasttext.bin`: Trained FastText model.
- Visual plots of similar words using t-SNE.

## 🔍 Example Use Case

Use this model to explore how Yelp users talk about food, service, or ambiance — and understand word semantics in user tips.
