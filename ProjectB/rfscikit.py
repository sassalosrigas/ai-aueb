import os
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from logisticregression import DatabaseLoader

def main():
    print("Loading reviews...")
    train_pos = DatabaseLoader.load_reviews(DatabaseLoader.TRAINING_POSITIVE)
    train_neg = DatabaseLoader.load_reviews(DatabaseLoader.TRAINING_NEGATIVE)
    test_pos = DatabaseLoader.load_reviews(DatabaseLoader.TEST_POSITIVE)
    test_neg = DatabaseLoader.load_reviews(DatabaseLoader.TEST_NEGATIVE)

    print("Creating vocabulary...")
    temp_vocabulary = DatabaseLoader.create_vocabulary(train_pos, train_neg, k=250, n=750)
    vocabulary = DatabaseLoader.finalize_vocabulary(temp_vocabulary, train_pos, train_neg, m=7500)
    vocab_list = list(vocabulary.keys())

    X_train = train_pos + train_neg
    y_train = [1] * len(train_pos) + [0] * len(train_neg)
    X_test = test_pos + test_neg
    y_test = [1] * len(test_pos) + [0] * len(test_neg)

    combined = list(zip(X_train, y_train))
    random.shuffle(combined)
    X_train_shuffled, y_train_shuffled = zip(*combined)
    
    vectorizer = CountVectorizer(vocabulary=vocab_list, binary=True)
    X_train_vec = vectorizer.transform(X_train_shuffled)
    X_test_vec = vectorizer.transform(X_test)

    split_idx = int(0.8 * len(X_train_shuffled))
    X_dev_vec, y_dev = X_train_vec[split_idx:], y_train_shuffled[split_idx:]
    X_train_vec, y_train_shuffled = X_train_vec[:split_idx], y_train_shuffled[:split_idx]

    params = {
        'n_estimators': 5,
        'max_features': 1000,
        'min_samples_split': 5,
        'max_depth': 11,
        'criterion': 'entropy',
        'bootstrap': True,
        'random_state': 42
    }

    sklearn_rf = RandomForestClassifier(**params)
    print("\nTraining Scikit-Learn Random Forest...")
    sklearn_rf.fit(X_train_vec, y_train_shuffled)

    def print_metrics(y_true, y_pred, label):
        print(f"\n{label} Evaluation:")
        print(classification_report(y_true, y_pred, target_names=["Negative (0)", "Positive (1)"]))
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"Macro Precision: {precision_score(y_true, y_pred, average='macro'):.4f}")
        print(f"Macro Recall: {recall_score(y_true, y_pred, average='macro'):.4f}")
        print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
        print(f"Micro Precision: {precision_score(y_true, y_pred, average='micro'):.4f}")
        print(f"Micro Recall: {recall_score(y_true, y_pred, average='micro'):.4f}")
        print(f"Micro F1: {f1_score(y_true, y_pred, average='micro'):.4f}")

    print_metrics(y_test, sklearn_rf.predict(X_test_vec), "Sklearn RF Test")

    train_sizes = [3000, 6000, 10000, 13000, 16000, 20000]
    train_precision, train_recall, train_f1 = [], [], []
    dev_precision, dev_recall, dev_f1 = [], [], []

    for size in train_sizes:
        subset_X, subset_y = X_train_vec[:size], y_train_shuffled[:size]
        sklearn_subset = RandomForestClassifier(**params)
        sklearn_subset.fit(subset_X, subset_y)

        train_preds = sklearn_subset.predict(subset_X)
        dev_preds = sklearn_subset.predict(X_dev_vec)

        train_precision.append(precision_score(subset_y, train_preds, average='macro'))
        train_recall.append(recall_score(subset_y, train_preds, average='macro'))
        train_f1.append(f1_score(subset_y, train_preds, average='macro'))

        dev_precision.append(precision_score(y_dev, dev_preds, average='macro'))
        dev_recall.append(recall_score(y_dev, dev_preds, average='macro'))
        dev_f1.append(f1_score(y_dev, dev_preds, average='macro'))

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(train_sizes, train_precision, label="Train Precision")
    plt.plot(train_sizes, dev_precision, label="Dev Precision")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Precision")
    plt.title("Precision over Training Samples")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_sizes, train_recall, label="Train Recall")
    plt.plot(train_sizes, dev_recall, label="Dev Recall")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Recall")
    plt.title("Recall over Training Samples")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(train_sizes, train_f1, label="Train F1")
    plt.plot(train_sizes, dev_f1, label="Dev F1")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("F1 Score")
    plt.title("F1 Score over Training Samples")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
