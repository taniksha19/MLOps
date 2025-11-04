# Changes made to Lab 1
# Lab 1: Causal LM Data Preprocessing and Training Pipeline

##  Overview

This project implements a complete data pipeline to prepare, train, and evaluate a Causal Language Model (CLM). The starting point was a basic data-loading notebook, which was **significantly modified** to build a unique and robust data lab.

The key changes involve swapping out the dataset and tokenizer and, most importantly, **fundamentally changing the preprocessing strategy** from "grouping" texts to "padding/truncating" individual examples.

We also extended the lab far beyond its original scope by adding a complete model training and evaluation loop, using **Perplexity** as a key metric to prove that our pipeline successfully prepared the data for a real-world machine-learning task.

### Core Components
* **Dataset:** `imdb` (Movie reviews)
* **Model:** `distilgpt2` (A smaller, faster version of GPT-2)
* **Tokenizer:** `AutoTokenizer` (from `distilgpt2`)
* **Key Metric:** Perplexity (PPL)

---

##  Summary of Significant Changes

This notebook is substantially different from the original lab. The changes focus on the data pipeline, the model, and the addition of a complete validation framework.

1.  **Dataset Changed:**
    * **Original:** `wikitext-2-raw-v1`
    * **New:** `imdb` (train and test splits)
    * **Reason:** To work with a familiar, document-oriented dataset where each example (a review) has a distinct beginning and end.

2.  **Tokenizer Changed:**
    * **Original:** `gpt2`
    * **New:** `distilgpt2`
    * **Reason:** To create a different set of vocabulary and token IDs, ensuring the entire pipeline produces unique data. `distilgpt2` is also faster to load and train.

3.  **Fundamental Preprocessing Strategy Change:**
    * **Original:** Used a `group_texts` function. This strategy concatenates all text in the dataset into one giant string and then chops it into fixed-size chunks (e.g., 128 tokens).
    * **New:** We **deleted** the `group_texts` function. We now use a **Pad/Truncate** strategy. Each movie review is treated as an individual example.
        * If a review is *longer* than our `block_size` (256), it is **truncated** (cut off).
        * If a review is *shorter* than 256, it is **padded** with special `<pad_token>` tokens to fill the remaining space.

4.  **Addition of Attention Masks:**
    * Our new padding strategy *requires* an `attention_mask`. This is a tensor of 1s and 0s that tells the model which tokens are real (`1`) and which are just padding (`0`) that should be ignored. This is a critical component of modern NLP data pipelines.

5.  **Parameter Changes:**
    * **`block_size`:** Increased from `128` to `256` to allow the model to learn from longer contexts.
    * **`batch_size`:** Increased from `8` to `16` to improve training efficiency and GPU utilization.

6.  **Added Full Training & Evaluation (New Functionality):**
    * The original lab *only* prepared data. We added code to:
        * Load the `distilgpt2` model (`AutoModelForCausalLM`).
        * Load an optimizer (`AdamW`).
        * Load and process a separate **test dataset** (`imdb` test split).
        * Implement an **evaluation loop** to calculate **Perplexity**.
        * Implement a **training loop** to fine-tune the model for one full epoch.

---

## 📈 Results & Analysis

The success of our custom pipeline is proven by the dramatic drop in perplexity after just one epoch of training. Perplexity measures how well the model predicts the test data (lower is better).

* **Baseline Perplexity (Untrained Model):** `154.8215`
* **Average Training Loss (1 Epoch):** `3.0952`
* **Final Perplexity (After 1 Epoch):** `19.1798`

**Conclusion:** The perplexity dropped from **154.8** to **19.2**. This massive improvement confirms that the model is learning the patterns of the `imdb` dataset, validating that our end-to-end data pipeline (loading, padding, truncating, and masking) is correct and effective.