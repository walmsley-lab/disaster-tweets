# Natural Language Processing with Disaster Tweets

This repository contains a natural language processing project for binary classification of tweets (disaster vs. non-disaster) using both traditional machine learning and deep learning approaches. The work was developed and evaluated using the Kaggle "Natural Language Processing with Disaster Tweets" competition dataset and executed primarily in a Kaggle environment.

> **Important**  
> The Jupyter notebook contains justifications and analysis as well as code. Please refer to the plot renderings in `results/plots/` or run them in Kaggle.
>
> https://www.kaggle.com/code/YOUR_USERNAME/nlp-disaster-tweets

## Problem Overview

The task is to classify tweets as either describing real disasters or using disaster-related terminology metaphorically/non-literally. This is a supervised NLP problem that requires distinguishing between literal disaster reports (e.g., "Forest fire near La Ronge Sask. Canada") and figurative language (e.g., "my social life is a disaster"), handling noisy social media text, and managing moderate class imbalance (43% disaster, 57% non-disaster).

## Repository Structure

```
nlp-disaster-tweets/
├── notebook/                    # Kaggle-compatible experiment notebook
├── results/
│   ├── model_comparison.csv     # Aggregated evaluation results
│   ├── plots/                   # Saved EDA and performance figures
│   └── summary.md               # Short experiment summary
├── data/                        # Dataset files (not included, download from Kaggle)
├── README.md
└── requirements.txt
```

## Models Evaluated

Multiple architectures were implemented and compared:

- **TF-IDF + Logistic Regression** – sparse lexical baseline with term frequency weighting
- **BiLSTM** – bidirectional Long Short-Term Memory network with learned embeddings
- **BiGRU** – bidirectional Gated Recurrent Unit network with learned embeddings

Text preprocessing strategies (clean vs. regular), dropout regularization, early stopping, and stratified train-validation splits were applied across all models.

## Key Results

| Model | Data Type | Best Val Accuracy | Val F1 Score |
|-------|-----------|-------------------|--------------|
| TF-IDF + LR | Clean | 0.8200 | **0.7706** |
| TF-IDF + LR | Regular | 0.8100 | 0.7678 |
| BiGRU | Clean | 0.8116 | 0.7638 |
| BiGRU | Regular | 0.8148 | 0.7581 |
| BiLSTM | Clean | 0.8168 | 0.7476 |
| BiLSTM | Regular | 0.8109 | 0.7312 |

The TF-IDF + Logistic Regression model achieved the best overall F1 performance, demonstrating that sparse lexical representations remain highly competitive for short-text classification tasks where keyword presence provides strong predictive signal.

## Results and Visualizations

All plots and evaluation artifacts generated during training are saved under `results/plots/`.

These include:

- Class distribution and tweet length analysis
- Word frequency bar charts and word clouds by class
- Training accuracy and loss curves (BiLSTM, BiGRU)
- Model comparison bar charts (Regular vs. Clean data)
- Confusion matrices for all models
- F1 performance by tweet length analysis
- Keyword association analysis

## How to Run

The primary workflow is designed to run inside Kaggle.

1. Upload the repository notebook to Kaggle or open the existing project
2. Attach the [NLP Getting Started dataset](https://www.kaggle.com/c/nlp-getting-started/data)
3. Run all cells sequentially
4. Download outputs from `/kaggle/working/`

Local execution is also possible with a compatible GPU setup and the required dependencies.

## Notes

- All evaluation metrics are computed on a held-out 20% validation split with stratified sampling
- Early stopping with patience of 5 epochs was used for neural models to prevent overfitting
- Text preprocessing includes URL/mention replacement, HTML entity decoding, and punctuation normalization
- The moderate class imbalance (43/57 split) was addressed through stratified splits and F1-focused evaluation
- Sequence models used a vocabulary limit of 10,000 tokens and maximum sequence length of 40 tokens

## References

1. Kaggle. [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started), 2019.
2. Ian Goodfellow, Yoshua Bengio, and Aaron Courville. *Deep Learning*. MIT Press, 2016. Chapter 10.
3. Christopher Olah. [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/), 2015.
