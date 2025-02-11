import numpy as np
import os
import re
from collections import Counter

class DatabaseLoader:
    BASE_DIR = os.path.dirname(os.path.abspath("/Users/rigas/Downloads/aclImdb"))
    DATA_DIR = os.path.join(BASE_DIR, "aclImdb_v1", "aclImdb")
    TRAINING_POSITIVE = os.path.join(DATA_DIR, "train", "pos")
    TRAINING_NEGATIVE = os.path.join(DATA_DIR, "train", "neg")
    TEST_POSITIVE = os.path.join(DATA_DIR, "test", "pos")
    TEST_NEGATIVE = os.path.join(DATA_DIR, "test", "neg")

    @staticmethod
    def load_reviews(directory: str) -> list:
        reviews = []
        for root, _, files in os.walk(directory):
            for file in files:
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    content = f.read()
                    reviews.append(re.sub(r'[^a-zA-Z ]', ' ', content).lower())
        return reviews

    @staticmethod
    def create_vocabulary(train_pos: list, train_neg: list, k: int, n: int, m: int) -> dict:
        stopwords = {"the", "is", "in", "for", "where", "when", "to", "at"}
        all_words = Counter()
        for review in train_pos + train_neg:
            words = set(review.split())
            all_words.update(word for word in words if word not in stopwords and len(word) > 1)
        sorted_words = all_words.most_common()[k:len(all_words) - n]
        vocabulary = {word: idx for idx, (word, _) in enumerate(sorted_words[:m])}
        return vocabulary

    @staticmethod
    def text_to_vector(text: str, vocabulary: dict) -> np.array:
        words = set(text.split())
        return np.array([1 if word in words else 0 for word in vocabulary])

class NaiveBayesClassifier:
    def __init__(self):
        self.vocabulary = None
        self.word_probs_pos = None
        self.word_probs_neg = None
        self.p_pos = None
        self.p_neg = None
    
    def train(self, train_pos: list, train_neg: list, vocabulary: dict, alpha: float = 1.0):
        self.vocabulary = vocabulary
        total_reviews = len(train_pos) + len(train_neg)
        self.p_pos = len(train_pos) / total_reviews
        self.p_neg = len(train_neg) / total_reviews
        pos_word_counts = Counter(word for review in train_pos for word in set(review.split()))
        neg_word_counts = Counter(word for review in train_neg for word in set(review.split()))
        self.word_probs_pos = {
            word: (pos_word_counts.get(word, 0) + alpha) / (len(train_pos) + 2 * alpha)
            for word in vocabulary
        }
        self.word_probs_neg = {
            word: (neg_word_counts.get(word, 0) + alpha) / (len(train_neg) + 2 * alpha)
            for word in vocabulary
        }
    
    def predict(self, text: str) -> str:
        words = set(text.split())
        log_prob_pos = np.log(self.p_pos)
        log_prob_neg = np.log(self.p_neg)
        for word in self.vocabulary:
            if word in words:
                log_prob_pos += np.log(self.word_probs_pos[word])
                log_prob_neg += np.log(self.word_probs_neg[word])
            else:
                log_prob_pos += np.log(1 - self.word_probs_pos[word])
                log_prob_neg += np.log(1 - self.word_probs_neg[word])
        return "positive" if log_prob_pos > log_prob_neg else "negative"
    
# Load a subset of the IMDB dataset
train_pos = DatabaseLoader.load_reviews(DatabaseLoader.TRAINING_POSITIVE)[:1000]
train_neg = DatabaseLoader.load_reviews(DatabaseLoader.TRAINING_NEGATIVE)[:1000]
test_pos = DatabaseLoader.load_reviews(DatabaseLoader.TEST_POSITIVE)[:500]
test_neg = DatabaseLoader.load_reviews(DatabaseLoader.TEST_NEGATIVE)[:500]

k = 50  # Exclude the 50 most common words
n = 50  # Exclude the 50 rarest words
m = 2000  # Select the 2000 most informative words

vocabulary = DatabaseLoader.create_vocabulary(train_pos, train_neg, k, n, m)

classifier = NaiveBayesClassifier()
classifier.train(train_pos, train_neg, vocabulary)

correct = 0
total = len(test_pos) + len(test_neg)

for review in test_pos:
    if classifier.predict(review) == "positive":
        correct += 1

for review in test_neg:
    if classifier.predict(review) == "negative":
        correct += 1

accuracy = correct / total
print(f"Accuracy: {accuracy * 100:.2f}%")


