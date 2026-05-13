# Part 3: NLP — Customer Support Sentiment Classification

## Dataset
- File: `customer_support_text_classification.csv`
- Source: https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJIV-wBvUYs
- Records: 1500 | Text column: `customer_message` | Label: `sentiment_label`
- Classes: positive, neutral, negative (3-class)

## Approach

### Text Preprocessing
1. Lowercase all text
2. Remove URLs, special characters, punctuation
3. Remove English stopwords (NLTK)
4. Result: clean token sequences

### Vectorisation
- **TF-IDF** (max 8,000 features, 1-2 word n-grams) — for traditional models
- **Word Embeddings** (dim=128, vocab=12,000) — for LSTM

### Models

| Model | Vectorisation | Type | Test Accuracy |
|---|---|---|---|
| Logistic Regression | TF-IDF | Traditional ML | 1.0000 |
| Naive Bayes | TF-IDF | Traditional ML | 1.0000 |
| Bidirectional LSTM | Word Embeddings | Deep Learning | 0.3500 |


## Key NLP Concepts

**Why vectorise?** Models work with numbers. TF-IDF assigns each word an
importance score based on frequency in the document vs the whole corpus.
Words like 'broken', 'refund', 'angry' score high in negative messages.

**RNNs and long-term dependencies:** Hidden state gets overwritten each step;
early words in long messages are effectively forgotten (vanishing gradient).

**LSTMs:** Three gates (forget/input/output) selectively preserve important
context across the full message — e.g. "I am very unhappy" at the start
influences the sentiment prediction even after 50 more words.

**Attention:** Allows the model to weight all positions simultaneously,
focusing on the most sentiment-relevant words regardless of position.

**Transformers:** Pre-trained models like BERT would further improve accuracy
by leveraging language understanding from millions of training documents.
