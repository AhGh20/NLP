
# 🧠Sequence Modeling with RNNs (LSTM & GRU)

This repository contains a collection of Jupyter notebooks for experimenting with character-level and word-level sequence modeling using Recurrent Neural Networks (RNNs), specifically LSTM and GRU architectures. All models are implemented using PyTorch’s built-in modules.

## 📚 Overview

The primary objective is to train models that can predict the next character or word in a sequence. This is a fundamental task in natural language processing (NLP) and is essential for applications like text generation, autocomplete, and machine translation.

The notebooks explore different levels of granularity (character vs. word) and compare the performance of LSTM and GRU-based models.

---

## 🗂️ Notebooks

### 🔠 Character-Level Modeling

- **`assign5_Char_LSTM_Build_In.ipynb`**  
  Trains a character-level language model using PyTorch’s built-in **LSTM** module. The model learns to generate text one character at a time.

- **`Char_GRU_Build_In.ipynb`**  
  Similar to the above, but uses a **GRU** instead of an LSTM. This notebook compares the GRU’s efficiency and performance on the same task.

### 📝 Word-Level Modeling

- **`assign5_Word_LSTM_Build_In.ipynb`**  
  Trains a word-level language model using an **LSTM**. The model learns to predict the next word in a sentence based on prior context.

- **`assign5_Word_GRU_Build_In.ipynb`**  
  Implements the same word-level task using a **GRU**. Highlights trade-offs between model complexity and performance.

---

## ⚙️ Setup Instructions

### ✅ Requirements

Ensure the following Python packages are installed:

- `torch`
- `numpy`
- `matplotlib`
- `jupyter`

Install them via pip:

```bash
pip install torch numpy matplotlib jupyter
