from typing import Dict, List, Tuple
import numpy as np
import os
import re
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score

class DatabaseLoader:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # directory of this file
    DATA_DIR = os.path.join(BASE_DIR, "aclImdb_v1") # directory of the dataset
    TRAINING_POSITIVE = os.path.join(DATA_DIR, "train", "pos") # directory of the positive training reviews
    TRAINING_NEGATIVE = os.path.join(DATA_DIR, "train", "neg") # directory of the negative training reviews
    TEST_POSITIVE = os.path.join(DATA_DIR, "test", "pos") # directory of the positive test reviews
    TEST_NEGATIVE = os.path.join(DATA_DIR, "test", "neg") # directory of the negative test reviews

    @staticmethod
    def load_reviews(directory: str) -> List[str]: # load reviews from a directory
        """for every file in the directory, read the content, clean it up and append it to the reviews list"""
        reviews = []
        for root, _, files in os.walk(directory):
            for file in files:
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    content = f.read()
                    reviews.append(re.sub(r'[^a-zA-Z ]', ' ', content).lower())
        return reviews

    @staticmethod
    def create_vocabulary(train_pos: List[str], train_neg: List[str], k: int, n: int) -> Dict[str, int]:
        """create a vocabulary from the training data by counting the frequency of each word"""
        stopwords = {"the", "is", "in", "for", "where", "when", "to", "at"}
        all_words = Counter()

        for review in train_pos + train_neg:
            words = review.split()
            all_words.update(word for word in words if word not in stopwords and len(word) > 1)

        sorted_words = all_words.most_common()
        sorted_words = sorted_words[k:len(sorted_words) - n]

        vocabulary = {word: idx for idx, (word, _) in enumerate(sorted_words)}
        return vocabulary

    @staticmethod
    def finalize_vocabulary(vocabulary: Dict[str, int], train_pos: List[str], train_neg: List[str], m: int) -> Dict[str, int]:
        """select the top m words from the vocabulary using information gain"""
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
        final_vocabulary = {word: idx for idx, (word, _) in enumerate(sorted_words[:m])}
        return final_vocabulary

    @staticmethod
    def entropy(positives: int, negatives: int) -> float:
        """calculate the entropy of a classification"""
        if positives + negatives == 0 or positives == 0 or negatives == 0:
            return 0
        p_pos = positives / (positives + negatives)
        p_neg = negatives / (positives + negatives)
        return -p_pos * np.log2(p_pos) - p_neg * np.log2(p_neg)

class NaiveBayesClassifier:
    def __init__(self, vocabulary: Dict[str, int]):
        """initialize the classifier with the vocabulary"""
        self.vocabulary = vocabulary
        self.prior = {}
        self.conditional_probs = {}

    def train(self, train_pos: List[str], train_neg: List[str]):
        """train the classifier using the training data"""
        total_docs = len(train_pos) + len(train_neg)
        self.prior["pos"] = len(train_pos) / total_docs
        self.prior["neg"] = len(train_neg) / total_docs

        word_counts = {"pos": Counter(), "neg": Counter()}
        for review in train_pos:
            for word in set(review.split()):
                if word in self.vocabulary:
                    word_counts["pos"][word] += 1
        for review in train_neg:
            for word in set(review.split()):
                if word in self.vocabulary:
                    word_counts["neg"][word] += 1

        self.conditional_probs = {"pos": {}, "neg": {}}
        for word in self.vocabulary:
            self.conditional_probs["pos"][word] = (word_counts["pos"].get(word, 0) + 1) / (len(train_pos) + 2)
            self.conditional_probs["neg"][word] = (word_counts["neg"].get(word, 0) + 1) / (len(train_neg) + 2)

    def predict(self, text: str) -> str:
        """predict the class of a review"""
        words = set(text.split())
        log_prob_pos = np.log(self.prior["pos"])
        log_prob_neg = np.log(self.prior["neg"])

        for word in self.vocabulary:
            if word in words:
                log_prob_pos += np.log(self.conditional_probs["pos"].get(word, 1e-6))
                log_prob_neg += np.log(self.conditional_probs["neg"].get(word, 1e-6))
            else:
                log_prob_pos += np.log(1 - self.conditional_probs["pos"].get(word, 1e-6))
                log_prob_neg += np.log(1 - self.conditional_probs["neg"].get(word, 1e-6))

        return "1" if log_prob_pos > log_prob_neg else "0"

def evaluate(y_true, y_pred) -> Tuple[float, float, float, float, float, float, float]:
    """evaluate the classifier using precision, recall and F1 score"""
    TP_pos = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "1" and yp == "1")
    FP_pos = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "0" and yp == "1")
    FN_pos = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "1" and yp == "0")

    TP_neg = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "0" and yp == "0")
    FP_neg = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "1" and yp == "0")
    FN_neg = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "0" and yp == "1")

    precision_pos = TP_pos / (TP_pos + FP_pos) if TP_pos + FP_pos > 0 else 0
    recall_pos = TP_pos / (TP_pos + FN_pos) if TP_pos + FN_pos > 0 else 0
    f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos) if precision_pos + recall_pos > 0 else 0

    precision_neg = TP_neg / (TP_neg + FP_neg) if TP_neg + FP_neg > 0 else 0
    recall_neg = TP_neg / (TP_neg + FN_neg) if TP_neg + FN_neg > 0 else 0
    f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg) if precision_neg + recall_neg > 0 else 0

    # Micro-averaged metrics
    micro_precision = (TP_pos + TP_neg) / (TP_pos + TP_neg + FP_pos + FP_neg) if TP_pos + TP_neg + FP_pos + FP_neg > 0 else 0
    micro_recall = (TP_pos + TP_neg) / (TP_pos + TP_neg + FN_pos + FN_neg) if TP_pos + TP_neg + FN_pos + FN_neg > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if micro_precision + micro_recall > 0 else 0

    # Macro-averaged metrics
    macro_precision = (precision_pos + precision_neg) / 2
    macro_recall = (recall_pos + recall_neg) / 2
    macro_f1 = (f1_pos + f1_neg) / 2

    return precision_pos, recall_pos, f1_pos, precision_neg, recall_neg, f1_neg, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1

def plot_learning_curve(train_sizes, train_scores, dev_scores, metric_name):
    """plot the learning curve of the classifier"""
    plt.figure()
    plt.plot(train_sizes, train_scores, 'o-', label=f"Training {metric_name}")
    plt.plot(train_sizes, dev_scores, 'o-', label=f"Development {metric_name}")
    plt.xlabel("Training Examples")
    plt.ylabel(metric_name)
    plt.title(f"Learning Curve ({metric_name})")
    plt.legend(loc="best")
    plt.show()

def main():
    """main function. Load the reviews, create the vocabulary, train the classifier and evaluate it"""
    print("Loading reviews...")
    train_pos = DatabaseLoader.load_reviews(DatabaseLoader.TRAINING_POSITIVE)
    train_neg = DatabaseLoader.load_reviews(DatabaseLoader.TRAINING_NEGATIVE)
    test_pos = DatabaseLoader.load_reviews(DatabaseLoader.TEST_POSITIVE)
    test_neg = DatabaseLoader.load_reviews(DatabaseLoader.TEST_NEGATIVE)

    print("Creating vocabulary...")
    vocabulary = DatabaseLoader.create_vocabulary(train_pos, train_neg, k=50, n=50)

    print("Finalizing vocabulary...")
    vocabulary = DatabaseLoader.finalize_vocabulary(vocabulary, train_pos, train_neg, m=1000)

    classifier = NaiveBayesClassifier(vocabulary)

    # Learning curves
    train_sizes = [100, 500, 1000, 2000, 5000, 10000]
    train_precisions, train_recalls, train_f1s = [], [], []
    dev_precisions, dev_recalls, dev_f1s = [], [], []

    for size in train_sizes:
        print(f"Training with {size} examples...")
        classifier.train(train_pos[:size//2], train_neg[:size//2])

        # Evaluate on training data
        y_train_true = ["1"] * (size//2) + ["0"] * (size//2)
        y_train_pred = [classifier.predict(review) for review in train_pos[:size//2] + train_neg[:size//2]]
        precision_pos, recall_pos, f1_pos, precision_neg, recall_neg, f1_neg, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1 = evaluate(y_train_true, y_train_pred)
        train_precisions.append(macro_precision)
        train_recalls.append(macro_recall)
        train_f1s.append(macro_f1)

        # Evaluate on development data
        y_dev_true = ["1"] * len(test_pos) + ["0"] * len(test_neg)
        y_dev_pred = [classifier.predict(review) for review in test_pos + test_neg]
        precision_pos, recall_pos, f1_pos, precision_neg, recall_neg, f1_neg, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1 = evaluate(y_dev_true, y_dev_pred)
        dev_precisions.append(macro_precision)
        dev_recalls.append(macro_recall)
        dev_f1s.append(macro_f1)

    # Plot learning curves
    plot_learning_curve(train_sizes, train_precisions, dev_precisions, "Precision")
    plot_learning_curve(train_sizes, train_recalls, dev_recalls, "Recall")
    plot_learning_curve(train_sizes, train_f1s, dev_f1s, "F1 Score")

    # Evaluate on test data
    print("Evaluating on test data...")
    y_test_true = ["1"] * len(test_pos) + ["0"] * len(test_neg)
    y_test_pred = [classifier.predict(review) for review in test_pos + test_neg]
    precision_pos, recall_pos, f1_pos, precision_neg, recall_neg, f1_neg, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1 = evaluate(y_test_true, y_test_pred)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test_true, y_test_pred, target_names=["Negative (0)", "Positive (1)"]))

    # Print accuracy
    accuracy = accuracy_score(y_test_true, y_test_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Print macro-averaged metrics
    print(f"Macro-averaged Precision: {macro_precision:.4f}")
    print(f"Macro-averaged Recall: {macro_recall:.4f}")
    print(f"Macro-averaged F1 Score: {macro_f1:.4f}")

    # Print micro-averaged metrics
    print(f"Micro-averaged Precision: {micro_precision:.4f}")
    print(f"Micro-averaged Recall: {micro_recall:.4f}")
    print(f"Micro-averaged F1 Score: {micro_f1:.4f}")

if __name__ == "__main__":
    main()