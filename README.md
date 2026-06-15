# 🐦 Helping Twitter Combat Hate Speech Using NLP and Machine Learning

> *A complete, end-to-end NLP pipeline — from raw, messy tweets to a tuned, recall-optimized hate speech classifier.*

---

## 📌 What is this project about?

Social media platforms like Twitter are incredible for free expression, but they're also breeding grounds for hate speech — content that demeans people based on race, religion, gender, or other characteristics. Manually moderating millions of tweets every day is humanly impossible.

This project answers one core question:

> **Can we train a machine learning model to automatically detect hate speech in tweets — reliably and at scale?**

The answer is yes — and this repository walks you through every single step of how that was done: cleaning raw Twitter data, engineering features, training models, fixing class imbalance, and tuning hyperparameters to maximize the model's ability to *catch* hate speech (not just be broadly "accurate").

---

## 🗂️ Table of Contents

- [Project Motivation](#-project-motivation)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Project Workflow — Step by Step](#-project-workflow--step-by-step)
  - [Step 1: Loading & Exploring the Data](#step-1-loading--exploring-the-data)
  - [Step 2: Tweet Cleaning — Why It Matters](#step-2-tweet-cleaning--why-it-matters)
  - [Step 3: Tokenization](#step-3-tokenization)
  - [Step 4: Removing Stopwords & Noise](#step-4-removing-stopwords--noise)
  - [Step 5: Exploratory Token Analysis](#step-5-exploratory-token-analysis)
  - [Step 6: Feature Engineering with TF-IDF](#step-6-feature-engineering-with-tf-idf)
  - [Step 7: Train-Test Split](#step-7-train-test-split)
  - [Step 8: Baseline Logistic Regression Model](#step-8-baseline-logistic-regression-model)
  - [Step 9: The Class Imbalance Problem](#step-9-the-class-imbalance-problem)
  - [Step 10: Fixing Class Imbalance](#step-10-fixing-class-imbalance)
  - [Step 11: Hyperparameter Tuning with GridSearchCV](#step-11-hyperparameter-tuning-with-gridsearchcv)
  - [Step 12: Recall-Optimized Final Model](#step-12-recall-optimized-final-model)
  - [Step 13: Final Evaluation on Test Set](#step-13-final-evaluation-on-test-set)
- [Key Concepts Explained](#-key-concepts-explained)
- [Why Recall Matters More Than Accuracy Here](#-why-recall-matters-more-than-accuracy-here)
- [Results Summary](#-results-summary)
- [What Could Be Done Next](#-what-could-be-done-next)
- [How to Run This Project](#-how-to-run-this-project)

---

## 💡 Project Motivation

Hate speech detection is one of the most important and challenging problems in NLP today. The challenges are real:

- Tweets are short, informal, full of slang, abbreviations, and emojis
- Hate speech is a **minority class** — most tweets are normal
- Being "accurate" is misleading: a model that labels everything as "not hate" is 90%+ accurate but completely useless
- What we actually need is high **recall** — catch as many hate tweets as possible, even if it means a few false alarms

This project addresses all of these challenges systematically.

---

## 📊 Dataset

**File:** `TwitterHate.csv`

The dataset contains labeled tweets with two columns:

| Column | Description |
|--------|-------------|
| `tweet` | The raw tweet text |
| `label` | `1` = Hate speech, `0` = Not hate speech |

The dataset is inherently **imbalanced** — a small fraction of tweets are labeled as hate speech. This is realistic: on any real platform, offensive content is the minority.

---

## 🛠️ Tech Stack

| Category | Library / Tool |
|----------|----------------|
| Data Handling | `pandas`, `numpy` |
| Text Processing | `re`, `nltk` (TweetTokenizer, stopwords) |
| Feature Engineering | `sklearn.feature_extraction.text.TfidfVectorizer` |
| Machine Learning | `sklearn.linear_model.LogisticRegression` |
| Model Tuning | `GridSearchCV`, `StratifiedKFold` |
| Evaluation | `accuracy_score`, `recall_score`, `f1_score`, `classification_report`, `confusion_matrix`, `roc_auc_score` |
| Visualization | `matplotlib`, `seaborn` |

---

## 🔁 Project Workflow — Step by Step

### Step 1: Loading & Exploring the Data

```python
df = pd.read_csv("TwitterHate.csv")
df.head()
df.tail(10)
df['label'].unique()
df.shape
```

**What happened here?**

Before building anything, we need to understand what we're working with. `df.shape` tells us how many tweets we have. `df['label'].unique()` confirms that labels are binary (0 and 1). Looking at the first and last few rows gives a feel for the tweet structure — you'll see messy, real-world text full of handles, hashtags, and URLs.

**Why this matters:** Never skip EDA (Exploratory Data Analysis). Rushing into modeling without understanding your data leads to bad decisions down the line.

---

### Step 2: Tweet Cleaning — Why It Matters

Raw tweets are noisy. Really noisy. Before any NLP can happen, we need to clean them up. Here's what was done and why:

#### 2a. Lowercase Normalization

```python
tweets = [tweet.lower() for tweet in tweets]
```

"Hate" and "hate" and "HATE" mean the same thing. By converting everything to lowercase, we ensure the model treats them as identical. This reduces vocabulary size and prevents the model from treating the same word as different features.

#### 2b. Remove User Handles (@username)

```python
tweets = [re.sub(r'@\w+', '', tweet) for tweet in tweets]
```

`@username` mentions carry no semantic meaning for hate speech detection — they're just pointers to accounts. Keeping them would just add noise. The regex `@\w+` matches the `@` symbol followed by any word characters.

#### 2c. Remove URLs

```python
tweets = [re.sub(r'http\S+|www.\S+', '', tweet) for tweet in tweets]
```

URLs don't tell us anything about the sentiment or nature of a tweet's content. They're stripped out using a regex that catches both `http://...` and `www....` patterns.

---

### Step 3: Tokenization

```python
from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
tweets_tokens = [tokenizer.tokenize(tweet) for tweet in tweets]
```

**What is tokenization?**

Tokenization breaks a sentence into individual units — words, punctuation, emoji — called **tokens**. Instead of treating the tweet as one big string, the model now sees a list of meaningful pieces.

**Why TweetTokenizer specifically?**

Regular tokenizers (like `word_tokenize`) were designed for formal text. They struggle with tweet-specific features like:
- Emoticons 😢
- Repeated characters ("sooooo good")
- Hashtags and handles

`TweetTokenizer` was built for Twitter. The parameters used:
- `preserve_case=False` → lowercases everything again (safety net)
- `strip_handles=True` → additional handle removal
- `reduce_len=True` → "loooove" becomes "loove" (max 3 repeated chars)

---

### Step 4: Removing Stopwords & Noise

#### 4a. Stopword Removal

```python
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
tweets_tokens = [
    [word for word in tweet if word not in stop_words]
    for tweet in tweets_tokens
]
```

**Stopwords** are extremely common words — "the", "is", "at", "which" — that appear in almost every sentence but carry no distinguishing meaning for classification. Removing them:
- Reduces vocabulary size significantly
- Forces the model to focus on content words that actually carry meaning

#### 4b. Removing Redundant Twitter-Specific Terms

```python
redundant_words = ['rt', 'amp']
```

- `rt` = "Retweet" — a structural Twitter thing, not content
- `amp` = "&amp;" HTML encoding that leaked into the text

These appear frequently but mean nothing for classification.

#### 4c. Strip # from Hashtags (Keep the Word)

```python
tweets_tokens = [
    [word.replace('#', '') for word in tweet]
    for tweet in tweets_tokens
]
```

`#BlackLivesMatter` becomes `BlackLivesMatter`. The `#` symbol itself adds no meaning — the word behind it does. This ensures hashtag content is treated the same as regular words.

#### 4d. Remove Single-Character Tokens

```python
tweets_tokens = [
    [word for word in tweet if len(word) > 1]
    for tweet in tweets_tokens
]
```

Single characters like "a", "i", or stray punctuation that survived earlier cleaning are removed. They carry almost no information.

---

### Step 5: Exploratory Token Analysis

```python
from collections import Counter
all_tokens = []
for tweet in tweets_tokens:
    all_tokens.extend(tweet)

word_freq = Counter(all_tokens)
word_freq.most_common(10)
```

After all that cleaning, we flatten all tokens into one big list and find the top 10 most frequent terms across the entire corpus. This step is purely exploratory — it tells us what the dominant vocabulary looks like after preprocessing. It's a sanity check: the top words should now actually be meaningful content words, not noise.

---

### Step 6: Feature Engineering with TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
```

**The core problem:** Machine learning models don't understand text. They need numbers.

**What is TF-IDF?**

TF-IDF stands for **Term Frequency–Inverse Document Frequency**. It converts text into numerical vectors where each dimension represents a word.

- **TF (Term Frequency):** How often does a word appear in *this* tweet?
- **IDF (Inverse Document Frequency):** How rare is this word across *all* tweets?

The TF-IDF score for a word is high when:
- It appears often in a specific tweet AND
- It's rare across the whole dataset

This means generic words that appear everywhere (even after stopword removal) get down-weighted, while distinctive, meaningful words get up-weighted.

**`max_features=5000`:** We keep only the top 5000 most informative terms. This controls dimensionality and prevents overfitting.

**Critical rule:** `fit_transform` is called ONLY on training data. The test data is only `transform`-ed using the vocabulary learned from training. This prevents **data leakage** — the model cannot "see" test data patterns during training.

---

### Step 7: Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

80% of data goes to training, 20% to testing. Two important parameters:

- **`random_state=42`:** Makes the split reproducible — same split every time you run it
- **`stratify=y`:** Ensures the proportion of hate vs. non-hate tweets is preserved in both train and test sets. Without this, the already-imbalanced minority class could be underrepresented in test data by chance

---

### Step 8: Baseline Logistic Regression Model

```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)

y_train_pred = lr_model.predict(X_train_tfidf)
y_test_pred = lr_model.predict(X_test_tfidf)
```

**Why Logistic Regression?**

Despite the name, Logistic Regression is a classification algorithm. It estimates the probability that a given tweet belongs to the hate speech class. It's:
- Interpretable (we can see which words drive predictions)
- Fast to train
- Strong baseline for text classification
- Works well with sparse TF-IDF vectors

**Evaluation:**

```python
train_accuracy = accuracy_score(y_train, y_train_pred)   # ~95.6%
train_recall   = recall_score(y_train, y_train_pred)     # Low
train_f1       = f1_score(y_train, y_train_pred)         # Moderate
```

The model looked great on accuracy (~95.6%) but had **low recall**. This is the class imbalance trap — and it's the most important problem this project solves.

---

### Step 9: The Class Imbalance Problem

**This is critical. Read carefully.**

Suppose 93% of tweets in our dataset are non-hate (label=0) and only 7% are hate speech (label=1).

A completely dumb model that predicts "not hate" for EVERY tweet achieves 93% accuracy. Impressive number, useless model.

**Recall** measures something different:

```
Recall = True Positives / (True Positives + False Negatives)
```

In plain English: **Of all the actual hate tweets, what fraction did the model correctly catch?**

A model with low recall is missing most of the hate speech — which defeats the entire purpose of building this system. For content moderation, a missed hate tweet (false negative) is far more harmful than a false alarm (false positive).

---

### Step 10: Fixing Class Imbalance

```python
lr_balanced = LogisticRegression(class_weight='balanced')
lr_balanced.fit(X_train_tfidf, y_train)
```

**`class_weight='balanced'`** is a simple but powerful fix. It tells the model to pay more attention to the minority class (hate speech) during training by automatically adjusting the penalty for misclassifying rare classes.

Mathematically, it weights each class inversely proportional to its frequency:

```
weight for class k = total_samples / (n_classes × samples_in_class_k)
```

So hate speech tweets — the rare ones — get a much higher weight. Misclassifying them costs the model more during training, so it learns to catch them.

**Result:** Recall and F1-score improved substantially. The model went from ignoring hate tweets to actually detecting them. F1 improved from ~0.55 to ~0.71.

---

### Step 11: Hyperparameter Tuning with GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight='balanced'),
    param_grid,
    cv=5,
    scoring='f1'
)
grid.fit(X_train_tfidf, y_train)
best_lr = grid.best_estimator_
```

**What are we tuning?**

| Hyperparameter | What it Controls |
|----------------|-----------------|
| `penalty` | Type of regularization (L1 = Lasso, L2 = Ridge, ElasticNet = both) |
| `C` | Regularization strength (lower C = stronger regularization = simpler model) |
| `solver` | Optimization algorithm used internally |

**Regularization** prevents overfitting. It penalizes the model for having too many large coefficients, forcing it to generalize rather than memorize training data.

**GridSearchCV** exhaustively tests every combination of parameters and uses 5-fold cross-validation on the training set. In 5-fold CV:
- Training data is split into 5 equal parts
- The model is trained 5 times, each time using 4 parts for training and 1 for validation
- The average validation score across all 5 folds is used to select the best parameters

This is much more robust than a single train-validation split.

**`scoring='f1'`:** The grid search optimizes for F1-score — the harmonic mean of precision and recall. This is more appropriate than accuracy for imbalanced datasets.

---

### Step 12: Recall-Optimized Final Model

```python
from sklearn.model_selection import StratifiedKFold

skf_recall = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

grid_recall = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight='balanced'),
    param_grid=param_grid,
    cv=skf_recall,
    scoring='recall'
)

grid_recall.fit(X_train_tfidf, y_train)
best_params = grid_recall.best_params_
best_score = grid_recall.best_score_
```

**Why StratifiedKFold?**

Regular KFold splits data randomly. With an imbalanced dataset, some folds might end up with very few (or even zero) hate speech examples — making evaluation unstable. `StratifiedKFold` ensures each fold preserves the original class ratio. This is best practice for imbalanced classification.

**Why switch from F1 to Recall scoring?**

At this stage, we deliberately prioritize recall over precision. We would rather:
- Flag a few non-hate tweets for review (false positives) 
- Than miss actual hate speech (false negatives)

This is a **business decision** about the cost of errors, baked directly into the model optimization process.

The best model from `grid_recall.best_estimator_` is our final classifier.

---

### Step 13: Final Evaluation on Test Set

```python
best_lr_final = grid_recall.best_estimator_
y_test_pred_final = best_lr_final.predict(X_test_tfidf)

test_recall = recall_score(y_test, y_test_pred_final)
test_f1     = f1_score(y_test, y_test_pred_final)
```

The final model is evaluated on the held-out test set — data it has never seen during training or tuning. This gives us an honest estimate of real-world performance.

The key metrics checked are:
- **Recall:** Are we catching actual hate speech?
- **F1:** Is there a reasonable balance between catching hate and not triggering too many false alarms?

---

## 🧠 Key Concepts Explained

### What is TF-IDF, really?

Imagine two tweets:
- Tweet A: "I hate this product, I hate it so much"
- Tweet B: "The weather today is nice"

The word "hate" appears a lot in Tweet A but nowhere in Tweet B. And "hate" is not a common word across all tweets. So TF-IDF gives "hate" a high score for Tweet A — making it a strong signal for classification.

### What is Logistic Regression doing?

It learns a weighted combination of TF-IDF features. Words associated with hate speech get positive weights; neutral words get near-zero weights. The model's output is a probability between 0 and 1 — above a threshold (default 0.5), it predicts hate speech.

### What is cross-validation?

Instead of evaluating the model once on one random split, cross-validation evaluates it multiple times on different splits and averages the results. This gives a much more reliable estimate of how the model will actually perform in the wild.

---

## 📈 Why Recall Matters More Than Accuracy Here

| Metric | What it Measures | Priority |
|--------|-----------------|----------|
| **Accuracy** | Overall correct predictions | ❌ Misleading here |
| **Precision** | When model says "hate", how often is it right? | Moderate |
| **Recall** | Of all actual hate tweets, how many did we catch? | ✅ Highest |
| **F1-Score** | Harmonic mean of Precision and Recall | ✅ High |

In content moderation:
- A **false negative** (missed hate speech) = harm allowed to spread on platform
- A **false positive** (normal tweet flagged) = minor inconvenience, human can review

The asymmetry of consequences justifies optimizing for recall.

---

## 📋 Results Summary

| Model Version | Accuracy | Recall | F1-Score |
|--------------|----------|--------|----------|
| Baseline LR (no balancing) | ~95.6% | Low | ~0.55 |
| LR with `class_weight='balanced'` | Slightly lower | High | ~0.71 |
| GridSearch (F1-optimized) | Tuned | Tuned | Best F1 |
| **GridSearch (Recall-optimized) — Final** | Balanced | **Highest** | Good |

---

## 🚀 What Could Be Done Next

1. **Deep Learning Models:** Fine-tuning BERT or RoBERTa on this dataset would likely outperform TF-IDF + Logistic Regression significantly, since transformers understand context and word order.

2. **Oversampling (SMOTE):** Instead of adjusting class weights, synthetically generate new minority class samples using SMOTE (Synthetic Minority Oversampling Technique).

3. **Threshold Tuning:** Instead of using the default 0.5 probability threshold, plot a Precision-Recall curve and choose a threshold that maximizes recall above an acceptable precision floor.

4. **Ensemble Methods:** Random Forest or XGBoost with class balancing could provide better performance.

5. **Deployment:** Wrap the final model in a FastAPI endpoint so it can be called in real-time from a platform's moderation pipeline.

---

## ⚙️ How to Run This Project

### Prerequisites

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn jupyter
```

### Download NLTK Data

```python
import nltk
nltk.download('stopwords')
```

### Run the Notebook

```bash
jupyter notebook NLP_TWETTER_PROJECT.ipynb
```

Make sure `TwitterHate.csv` is in the same directory as the notebook.

---

## 👤 Author

**Tushar Parihar**  
MSc Bioinformatics, Savitribai Phule Pune University (SPPU)  
[GitHub](https://github.com/TusharParihar128)

---

## 📜 License

This project is part of an NLP coursework submission. Feel free to reference or adapt with attribution.
