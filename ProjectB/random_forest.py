import os
import math
import random
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
from logisticregression import DatabaseLoader, LogisticRegression

@dataclass
class ID3Node:
    feature_idx: int = -1 #index tou split
    is_leaf: bool = False #einai/den einai fyllo
    category: int = -1  #kathgoria poy anhkei to review
    left: Optional['ID3Node'] = None   #aristero paidi
    right: Optional['ID3Node'] = None   #deksi paidi

class ID3:
    def __init__(self, num_features: int, max_features: Optional[int] = None, min_samples_split: int = 10):
        self.num_features = num_features #sinolikos arithmos features
        self.max_features = max_features if max_features else num_features  #features gia kathe split
        self.min_samples_split = min_samples_split #elaxista sample gia kathe split
        self.tree: Optional[ID3Node] = None #root dentrou

    @staticmethod
    def entropy(values: np.ndarray) -> float:
        """Ypologismos entropias"""
        if len(values) == 0:
            return 0.0
        counts = np.bincount(values)  #emfaniseis kathe label
        probabilities = counts / len(values)  #upologismos pithanothtas label
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))  #prosthiki mikrou arithmou gia apofygh log(0)

    def information_gain(self, y: np.ndarray, feature_vector: np.ndarray) -> float:
        """Ypologismos information gain"""
        total_entropy = self.entropy(y)
        
        
        left_mask = feature_vector == 1 #data aristerou split
        right_mask = ~left_mask     #data deksiou split

        left_entropy = self.entropy(y[left_mask])
        right_entropy = self.entropy(y[right_mask])

        left_weight = np.sum(left_mask) / len(y)
        right_weight = np.sum(right_mask) / len(y)

        weighted_entropy = left_weight * left_entropy + right_weight * right_entropy
        return total_entropy - weighted_entropy #information gain

    def fit(self, X: List[List[int]], y: List[int], max_depth: int) -> None:
        """Ekpaideush montelou ID3"""
        X = np.array(X)  
        y = np.array(y)
        self.tree = self._build_tree(X, y, list(range(self.num_features)), max_depth, 0)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, available_features: List[int], max_depth: int, depth: int) -> ID3Node:
        """Anadromikh klhsh dentrou apofashs"""
        if depth >= max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
            return ID3Node(is_leaf=True, category=self._most_common(y))
        print("tree")
    
        if self.max_features < len(available_features):
            selected_features = random.sample(available_features, self.max_features)  #tuxaia epilogh features
        else:
            selected_features = available_features

        
        best_feature = max(selected_features, key=lambda feat: self.information_gain(y, X[:, feat]), default=-1)

        if best_feature == -1:
            return ID3Node(is_leaf=True, category=self._most_common(y))

        node = ID3Node(feature_idx=best_feature)
        new_available = [f for f in available_features if f != best_feature]

        # Kane split ta dedomena
        left_mask = X[:, best_feature] == 1
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask]

        #Xtise anadromika ta upodentra
        node.left = self._build_tree(X_left, y_left, new_available, max_depth, depth + 1)
        node.right = self._build_tree(X_right, y_right, new_available, max_depth, depth + 1)

        return node

    @staticmethod
    def _most_common(y: np.ndarray) -> int:
        """Vriskei to pio sixno label"""
        return np.argmax(np.bincount(y)) if len(y) > 0 else 0

    def predict(self, X: List[List[int]]) -> List[int]:
        """Provlepsh label"""
        X = np.array(X)  
        return [self._predict_single(x, self.tree) for x in X]

    def _predict_single(self, x: np.ndarray, node: ID3Node) -> int:
        """Provlepsh label instance"""
        if node.is_leaf:
            return node.category
        return self._predict_single(x, node.left if x[node.feature_idx] == 1 else node.right)

class RandomForest:     #Random Forest me ID3 dentra
    def __init__(self, num_trees: int, num_features: int, max_features: Optional[int] = None, min_samples_split: int = 10, max_depth: int = 10):
        self.num_trees = num_trees #dentra sto dasos
        self.trees = []  #lista pou apothikeuei dentra
        self.num_features = num_features  #arithmos features 
        self.max_features = max_features if max_features else num_features
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def fit(self, X: List[List[int]], y: List[int]):
        """Ekpaideush random forest"""
        X = np.array(X)
        y = np.array(y)
        
        for _ in range(self.num_trees):
            bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            tree = ID3(num_features=self.num_features, max_features=self.max_features, min_samples_split=self.min_samples_split)
            tree.fit(X_bootstrap.tolist(), y_bootstrap.tolist(), max_depth=self.max_depth)
            self.trees.append(tree)

    def predict(self, X: List[List[int]]) -> List[int]:
        """Provlepsh me xrhsh majority votes"""
        X = np.array(X)
        all_predictions = np.array([tree.predict(X.tolist()) for tree in self.trees])
        majority_votes = [Counter(all_predictions[:, i]).most_common(1)[0][0] for i in range(len(X))]
        return majority_votes

    def evaluate(self, test_data: List[List[int]], test_labels: List[int]) -> None:
        """Aksiologhsh montelou kai eksagogh metrics"""
        test_preds = self.predict(test_data)

        precision = precision_score(test_labels, test_preds, average=None)
        recall = recall_score(test_labels, test_preds, average=None)
        f1 = f1_score(test_labels, test_preds, average=None)

        micro_precision = precision_score(test_labels, test_preds, average='micro')
        micro_recall = recall_score(test_labels, test_preds, average='micro')
        micro_f1 = f1_score(test_labels, test_preds, average='micro')

        macro_precision = precision_score(test_labels, test_preds, average='macro')
        macro_recall = recall_score(test_labels, test_preds, average='macro')
        macro_f1 = f1_score(test_labels, test_preds, average='macro')

        print("Test Metrics:")
        print(f"Precision (Class 0, Class 1): {precision}")
        print(f"Recall (Class 0, Class 1): {recall}")
        print(f"F1 Score (Class 0, Class 1): {f1}")
        print(f"Micro-Averaged Precision: {micro_precision}")
        print(f"Micro-Averaged Recall: {micro_recall}")
        print(f"Micro-Averaged F1: {micro_f1}")
        print(f"Macro-Averaged Precision: {macro_precision}")
        print(f"Macro-Averaged Recall: {macro_recall}")
        print(f"Macro-Averaged F1: {macro_f1}")

        print("\nClassification Report:")
        print(classification_report(test_labels, test_preds, target_names=["Class 0", "Class 1"]))

    def train_plot_learning_curves(self, X_train: List[List[int]], y_train: List[int], X_dev: List[List[int]], y_dev: List[int], log_interval: int = 4000) -> None:
        """Enarksh ekpaideushs kai plot twn aparaithtwn learning curves"""
        train_precision, train_recall, train_f1 = [], [], [] #arxikopoihsh pinakwn gia ta plots
        dev_precision, dev_recall, dev_f1 = [], [], []
        sample_counts = []

        for i in range(log_interval, len(X_train) + 1, log_interval):
            subset_X = X_train[:i]
            subset_y = y_train[:i]

            #epilogh subset kai ekpaideush se sigkekrimeno ariyhmo dedomenwn kathe fora
            self.fit(subset_X, subset_y)

            #append se arrays statistikwn gia to xtisimo kampulhs    
            train_preds = self.predict(subset_X)
            train_precision.append(precision_score(subset_y, train_preds, average='macro'))
            train_recall.append(recall_score(subset_y, train_preds, average='macro'))
            train_f1.append(f1_score(subset_y, train_preds, average='macro'))

        
            dev_preds = self.predict(X_dev)
            dev_precision.append(precision_score(y_dev, dev_preds, average='macro'))
            dev_recall.append(recall_score(y_dev, dev_preds, average='macro'))
            dev_f1.append(f1_score(y_dev, dev_preds, average='macro'))

            sample_counts.append(i)

        
        plt.figure(figsize=(18, 6))

        #plot gia precision
        plt.subplot(1, 3, 1)
        plt.plot(sample_counts, train_precision, label="Train Precision")
        plt.plot(sample_counts, dev_precision, label="Dev Precision")
        plt.xlabel("Number of Training Samples")
        plt.ylabel("Precision")
        plt.title("Precision over Training Samples")
        plt.legend()

        #plot gia recall
        plt.subplot(1, 3, 2)
        plt.plot(sample_counts, train_recall, label="Train Recall")
        plt.plot(sample_counts, dev_recall, label="Dev Recall")
        plt.xlabel("Number of Training Samples")
        plt.ylabel("Recall")
        plt.title("Recall over Training Samples")
        plt.legend()

        #plot gia f1
        plt.subplot(1, 3, 3)
        plt.plot(sample_counts, train_f1, label="Train F1")
        plt.plot(sample_counts, dev_f1, label="Dev F1")
        plt.xlabel("Number of Training Samples")
        plt.ylabel("F1 Score")
        plt.title("F1 Score over Training Samples")
        plt.legend()

        plt.tight_layout()
        

if __name__ == "__main__":
    print("Beginning ID3 training...")
    #load, preprocessing kai apothikeush dedomenwn
    train_pos = DatabaseLoader.load_reviews(DatabaseLoader.TRAINING_POSITIVE)
    train_neg = DatabaseLoader.load_reviews(DatabaseLoader.TRAINING_NEGATIVE)
    test_pos = DatabaseLoader.load_reviews(DatabaseLoader.TEST_POSITIVE)
    test_neg = DatabaseLoader.load_reviews(DatabaseLoader.TEST_NEGATIVE)

    
    temp_vocabulary = DatabaseLoader.create_vocabulary(train_pos, train_neg, 250, 750)
    vocabulary = DatabaseLoader.finalize_vocabulary(temp_vocabulary, train_pos, train_neg, 7500)

    training_data = [LogisticRegression.convert_to_vector(review, vocabulary) for review in train_pos + train_neg]
    training_labels = [1] * len(train_pos) + [0] * len(train_neg)

    test_data = [LogisticRegression.convert_to_vector(review, vocabulary) for review in test_pos + test_neg]
    test_labels = [1] * len(test_pos) + [0] * len(test_neg)

    print("Shuffling training data...")
    training_data = np.array(training_data)
    training_labels = np.array(training_labels)
    shuffle_indices = np.random.permutation(len(training_data))
    training_data = training_data[shuffle_indices]
    training_labels = training_labels[shuffle_indices]

    split_idx = int(0.8 * len(training_data))
    dev_data, dev_labels = training_data[split_idx:], training_labels[split_idx:]
    training_data, training_labels = training_data[:split_idx], training_labels[:split_idx]
    #telos epeksergasias dedomenwn

    num_trees = 5  #Arithmos dentrwn sto dasos
    rf_model = RandomForest(num_trees=num_trees, num_features=len(vocabulary), max_features=1000, min_samples_split=5, max_depth=11)
    
    #Train montelou kai emfanish kampulwn
    rf_model.train_plot_learning_curves(training_data.tolist(), training_labels.tolist(), dev_data.tolist(), dev_labels.tolist())
    
    #Aksiologhsh sta test data meta apo train
    print("Evaluating Random Forest...")
    rf_model.evaluate(test_data, test_labels)
    plt.show()