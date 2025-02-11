import os
import re
import numpy as np
from collections import Counter
from typing import List, Dict

class DatabaseLoader:
  
    # Use relative paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    DATA_DIR = os.path.join(BASE_DIR, "aclImdb_v1", "aclImdb")  # Path to the data directory

    TRAINING_POSITIVE = os.path.join(DATA_DIR, "train", "pos")  # Relative path to positive training reviews
    TRAINING_NEGATIVE = os.path.join(DATA_DIR, "train", "neg")  # Relative path to negative training reviews
    TEST_POSITIVE = os.path.join(DATA_DIR, "test", "pos")       # Relative path to positive test reviews
    TEST_NEGATIVE = os.path.join(DATA_DIR, "test", "neg")       # Relative path to negative test reviews
    @staticmethod
    def load_reviews(directory: str) -> List[str]:
        """Load reviews from a directory and preprocess them."""
        reviews = []
        for root, _, files in os.walk(directory):
            print("thodoris")
            for file in files:
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    content = f.read()
                    reviews.append(re.sub(r'[^a-zA-Z ]', ' ', content).lower())
        return reviews

    @staticmethod
    def create_vocabulary(train_pos: List[str], train_neg: List[str], k: int, n: int) -> Dict[str, int]:
        """Create a vocabulary from training data, filtering out stopwords and rare/common words."""
        #all_words = defaultdict(int)
        stopwords = {"the", "is", "in", "for", "where", "when", "to", "at"}

        all_words = Counter()

        for review in train_pos + train_neg:
            words = review.split()
            all_words.update(word for word in words if word not in stopwords and len(word) > 1)

        #sorted_words = sorted(all_words.items(), key=lambda x: x[1])
        sorted_words = all_words.most_common()
        sorted_words = sorted_words[k:len(sorted_words) - n]

        vocabulary = {word: idx for idx, (word, _) in enumerate(sorted_words)}
        print(len(vocabulary))
        return vocabulary

    @staticmethod
    def finalize_vocabulary(vocabulary: Dict[str, int], train_pos: List[str], train_neg: List[str], m: int) -> Dict[str, int]:
        """Finalize vocabulary by selecting top `m` words based on information gain."""
        information_gain = {}
        total_reviews = len(train_pos) + len(train_neg)

        p_pos = len(train_pos) / total_reviews
        p_neg = len(train_neg) / total_reviews
        h = DatabaseLoader.entropy(p_pos, p_neg)

        pos_word_counts = Counter(word for review in train_pos for word in review.split())
        neg_word_counts = Counter(word for review in train_neg for word in review.split())


        for word in vocabulary:
            print(word)
            reviews_with_word_pos = pos_word_counts.get(word,0)
            reviews_with_word_neg = neg_word_counts.get(word,0)
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
        """Calculate entropy for a given set of positive and negative counts."""
        if positives + negatives == 0 or positives == 0 or negatives == 0:
            return 0
        p_pos = positives / (positives + negatives)
        p_neg = negatives / (positives + negatives)
        return -p_pos * np.log2(p_pos) - p_neg * np.log2(p_neg)