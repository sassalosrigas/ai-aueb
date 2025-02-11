import numpy as np
import os
import re
from collections import Counter

class DatabaseLoader:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
    def create_vocabulary(train_pos: list, train_neg: list, k: int, n: int) -> dict:
        stopwords = {"the", "is", "in", "for", "where", "when", "to", "at"}
        all_words = Counter()
        for review in train_pos + train_neg:
            words = review.split()
            all_words.update(word for word in words if word not in stopwords and len(word) > 1)
        sorted_words = all_words.most_common()[k:len(all_words) - n]
        return {word: idx for idx, (word, _) in enumerate(sorted_words)}

    @staticmethod
    def finalize_vocabulary(vocabulary: dict, train_pos: list, train_neg: list, m: int) -> dict:
        information_gain = {}
        total_reviews = len(train_pos) + len(train_neg)
        p_pos = len(train_pos) / total_reviews
        p_neg = len(train_neg) / total_reviews
        h = DatabaseLoader.entropy(p_pos, p_neg)
        pos_word_counts = Counter(word for review in train_pos for word in review.split())
        neg_word_counts = Counter(word for review in train_neg for word in review.split())
        for word in vocabulary:
            reviews_with_word_pos = pos_word_counts.get(word, 0)
            reviews_with_word_neg = neg_word_counts.get(word, 0)
            reviews_without_word_pos = len(train_pos) - reviews_with_word_pos
            reviews_without_word_neg = len(train_neg) - reviews_with_word_neg
            reviews_with_word = reviews_with_word_pos + reviews_with_word_neg
            reviews_without_word = reviews_without_word_pos + reviews_without_word_neg
            p_word = reviews_with_word / total_reviews
            p_not_word = reviews_without_word / total_reviews
            h_word = DatabaseLoader.entropy(reviews_with_word_pos, reviews_with_word_neg)
            h_not_word = DatabaseLoader.entropy(reviews_without_word_pos, reviews_without_word_neg)
            information_gain[word] = h - p_word * h_word - p_not_word * h_not_word
        sorted_words = sorted(information_gain.items(), key=lambda x: x[1], reverse=True)
        return {word: idx for idx, (word, _) in enumerate(sorted_words[:m])}

    @staticmethod
    def entropy(positives: int, negatives: int) -> float:
        if positives + negatives == 0 or positives == 0 or negatives == 0:
            return 0
        p_pos = positives / (positives + negatives)
        p_neg = negatives / (positives + negatives)
        return -p_pos * np.log2(p_pos) - p_neg * np.log2(p_neg)

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
