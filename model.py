"""
NumPy Text Classifier from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - clean_text
# ── Step 001  clean_text (Strictly Alphabetical + Spaces) ──
def clean_text(text: str) -> str:
    # 1. Map only letters and spaces to lowercase; turn digits and symbols into spaces
    cleaned = "".join([c.lower() if c.isalpha() or c.isspace() else " " for c in text])
    
    # 2. Strip only outer leading/trailing spaces without touching internal gaps
    return cleaned.strip()

# Step 2 - tokenize
# ── Step 002  tokenize ──
def tokenize(text: str) -> list[str]:
    return clean_text(text).split()

# Step 3 - tokenize_corpus
# ── Step 003  tokenize_corpus ──
def tokenize_corpus(texts: list[str]) -> list[list[str]]:
    return [tokenize(t) for t in texts]

# Step 4 - split_train_val_test_indices
# ── Step 004  split_train_val_test_indices ──
def split_train_val_test_indices(n: int, val_fraction: float, test_fraction: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_val = int(n * val_fraction)
    n_test = int(n * test_fraction)
    
    val_idx = indices[:n_val]
    test_idx = indices[n_val:n_val + n_test]
    train_idx = indices[n_val + n_test:]
    return train_idx, val_idx, test_idx

# Step 5 - count_word_frequencies
# ── Step 005  count_word_frequencies ──
def count_word_frequencies(tokenized_corpus: list[list[str]]) -> dict[str, int]:
    counts = {}
    for doc in tokenized_corpus:
        for token in doc:
            counts[token] = counts.get(token, 0) + 1
    return counts

# Step 6 - build_vocabulary
# ── Step 006  build_vocabulary ──
def build_vocabulary(word_counts: dict[str, int], max_size: int) -> dict[str, int]:
    sorted_words = sorted(word_counts.items(), key=lambda item: (-item[1], item[0]))
    return {word: idx for idx, (word, _) in enumerate(sorted_words[:max_size])}

# Step 7 - tokens_to_bow
# ── Step 007  tokens_to_bow ──
def tokens_to_bow(tokens: list[str], vocab: dict[str, int]) -> np.ndarray:
    bow = np.zeros(len(vocab), dtype=float)
    for token in tokens:
        if token in vocab:
            bow[vocab[token]] += 1.0
    return bow

# Step 8 - corpus_to_bow_matrix
# ── Step 008  corpus_to_bow_matrix (Fixed Empty Corpus Case) ──
def corpus_to_bow_matrix(tokenized_corpus: list[list[str]], vocab: dict[str, int]) -> np.ndarray:
    # If the corpus is empty, explicitly return an empty 2D array of shape (0, V)
    if not tokenized_corpus:
        return np.empty((0, len(vocab)), dtype=float)
        
    return np.array([tokens_to_bow(tokens, vocab) for tokens in tokenized_corpus])

# Step 9 - compute_document_frequencies
# ── Step 009  compute_document_frequencies ──
def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    return np.sum(bow_matrix > 0, axis=0)

# Step 10 - compute_idf
# ── Step 010  compute_idf (Corrected Signature & Math) ──
def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    # 1. Scalar numerator with Laplace smoothing
    numerator = n_docs + 1
    
    # 2. Element-wise denominator array with Laplace smoothing
    denominator = df + 1
    
    # 3. Vectorized true division, natural log, and trailing baseline shift
    return np.log(numerator / denominator) + 1.0

# Step 11 - transform_tfidf
# ── Step 011  transform_tfidf (Corrected Vector Multiply) ──
def transform_tfidf(bow_matrix: np.ndarray, idf: np.ndarray) -> np.ndarray:
    # Scale term counts via standard column-wise broadcasting 
    # Do NOT L2-normalize rows or alter the output shapes/dtypes
    return bow_matrix * idf

# Step 12 - fit_tfidf
# ── Step 012  fit_tfidf ──
def fit_tfidf(bow_train_matrix: np.ndarray) -> np.ndarray:
    df = compute_document_frequencies(bow_train_matrix)
    return compute_idf(df, bow_train_matrix.shape[0])

# Step 13 - sigmoid
# ── Step 013  sigmoid ──
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

# Step 14 - logistic_predict_proba
# ── Step 014  logistic_predict_proba ──
def logistic_predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return sigmoid(np.dot(X, w) + b)

# Step 15 - binary_cross_entropy
# ── Step 015  binary_cross_entropy ──
def binary_cross_entropy(y_true: np.ndarray, y_proba: np.ndarray, w: np.ndarray, l2_lambda: float) -> float:
    m = len(y_true)
    epsilon = 1e-15
    y_proba = np.clip(y_proba, epsilon, 1.0 - epsilon)
    loss = -np.mean(y_true * np.log(y_proba) + (1.0 - y_true) * np.log(1.0 - y_proba))
    reg = 0.5 * l2_lambda * np.sum(w ** 2)
    return float(loss + reg)

# Step 16 - logistic_gradients
# ── Step 016  logistic_gradients ──
def logistic_gradients(X: np.ndarray, y_true: np.ndarray, y_proba: np.ndarray, w: np.ndarray, l2_lambda: float):
    m = X.shape[0]
    error = y_proba - y_true
    dw = (np.dot(X.T, error) / m) + (l2_lambda * w)
    db = float(np.mean(error))
    return (dw, db)

# Step 17 - initialize_logistic_params
# ── Step 017  initialize_logistic_params ──
def initialize_logistic_params(n_features: int):
    return np.zeros(n_features, dtype=float), 0.0

# Step 18 - gradient_descent_step
# ── Step 018  gradient_descent_step (Full Batch Update + Loss Chaining) ──
def gradient_descent_step(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lr: float, l2_lambda: float) -> tuple:
    # 1. Forward Pass: Compute probabilities using our stable sigmoid helper
    y_proba = logistic_predict_proba(X, w, b)
    
    # 2. Metric Collection: Calculate the regularised cross-entropy loss
    loss = binary_cross_entropy(y, y_proba, w, l2_lambda)
    
    # 3. Backward Pass: Compute gradients for weights and bias
    dw, db = logistic_gradients(X, y, y_proba, w, l2_lambda)
    
    # 4. Optimization: Adjust parameters against the gradient direction
    w_new = w - lr * dw
    b_new = b - lr * db
    
    # Return exactly the 3-element tuple expected by the harness
    return w_new, b_new, loss

# Step 19 - train_logistic_regression
# ── Step 019  train_logistic_regression (Synchronized) ──
def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float, l2_lambda: float, n_epochs: int) -> tuple:
    # 1. Safely initialize weights and bias vectors based on feature width
    w, b = initialize_logistic_params(X.shape[1])
    losses = []
    
    # 2. Iterate through epochs using our updated unified gradient descent function
    for _ in range(n_epochs):
        # Step 018 now internally computes proba, loss, gradients, and returns updated params
        w, b, loss = gradient_descent_step(X, y, w, b, lr, l2_lambda)
        losses.append(loss)
        
    # 3. Return the fully trained parameters along with the historical track
    return w, b, losses

# Step 20 - predict_labels
def predict_labels(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert predicted probabilities into hard binary labels.

    Args:
        proba: 1-D array of probabilities in [0, 1], shape (N,).
        threshold: Decision threshold; proba >= threshold maps to 1.

    Returns:
        Integer array of shape (N,) with values in {0, 1}.
    """
    # TODO: Convert probabilities to hard binary labels via the threshold...
    return (proba >= threshold).astype(int)
    pass

# Step 21 - confusion_counts
# ── Step 021  confusion_counts ──
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, tn, fn

# Step 22 - metrics_from_counts (not yet solved)
# TODO: implement

# Step 23 - tune_decision_threshold (not yet solved)
# TODO: implement

# Step 24 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 25 - vectorize_texts (not yet solved)
# TODO: implement

# Step 26 - predict_text (not yet solved)
# TODO: implement

# Step 27 - collect_prediction_errors (not yet solved)
# TODO: implement

