import os
import re
from collections import Counter
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from naiveBayes import DatabaseLoader, plot_learning_curve

def main():
    """Main function. Load the reviews, create the vocabulary, train the scikit-learn classifier, and evaluate it."""
    print("Loading reviews...")
    train_pos = DatabaseLoader.load_reviews(DatabaseLoader.TRAINING_POSITIVE)
    train_neg = DatabaseLoader.load_reviews(DatabaseLoader.TRAINING_NEGATIVE)
    test_pos = DatabaseLoader.load_reviews(DatabaseLoader.TEST_POSITIVE)
    test_neg = DatabaseLoader.load_reviews(DatabaseLoader.TEST_NEGATIVE)

    print("Creating vocabulary...")
    vocabulary = DatabaseLoader.create_vocabulary(train_pos, train_neg, k=100, n=100)

    print("Finalizing vocabulary...")
    vocabulary = DatabaseLoader.finalize_vocabulary(vocabulary, train_pos, train_neg, m=5000)

    # Combine the data
    X_train = train_pos + train_neg
    y_train = ["1"] * len(train_pos) + ["0"] * len(train_neg)
    X_test = test_pos + test_neg
    y_test = ["1"] * len(test_pos) + ["0"] * len(test_neg)

    # Vectorize the data
    vectorizer = CountVectorizer(vocabulary=vocabulary.keys(), binary=True)
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train BernoulliNB classifier
    print("Training scikit-learn's Bernoulli Naive Bayes classifier...")
    classifier = BernoulliNB()
    classifier.fit(X_train_vec, y_train)

    # Evaluate classifier
    print("Evaluating scikit-learn's classifier...")
    y_test_pred_sk = classifier.predict(X_test_vec)

    # Print classification report
    print("\nBernoulliNB Classification Report:")
    print(classification_report(y_test, y_test_pred_sk, target_names=["Negative (0)", "Positive (1)"]))

    # Print accuracy
    accuracy_sk = accuracy_score(y_test, y_test_pred_sk)
    print(f"Accuracy: {accuracy_sk:.4f}")

    # Calculate macro-averaged and micro-averaged metrics
    macro_precision = precision_score(y_test, y_test_pred_sk, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, y_test_pred_sk, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_test_pred_sk, average="macro", zero_division=1)

    micro_precision = precision_score(y_test, y_test_pred_sk, average="micro")
    micro_recall = recall_score(y_test, y_test_pred_sk, average="micro")
    micro_f1 = f1_score(y_test, y_test_pred_sk, average="micro")

    # Print macro-averaged metrics
    print(f"Macro-averaged Precision: {macro_precision:.4f}")
    print(f"Macro-averaged Recall: {macro_recall:.4f}")
    print(f"Macro-averaged F1 Score: {macro_f1:.4f}")

    # Print micro-averaged metrics
    print(f"Micro-averaged Precision: {micro_precision:.4f}")
    print(f"Micro-averaged Recall: {micro_recall:.4f}")
    print(f"Micro-averaged F1 Score: {micro_f1:.4f}")

    # Learning curves
    train_sizes = [100, 500, 1000, 2000, 5000, 10000]
    train_precisions, train_recalls, train_f1s = [], [], []
    dev_precisions, dev_recalls, dev_f1s = [], [], []

    for size in train_sizes:
        print(f"Training with {size} examples...")
        # Subset the training data
        X_train_subset = train_pos[:size//2] + train_neg[:size//2]
        y_train_subset = ["1"] * (size//2) + ["0"] * (size//2)

        # Train on the subset
        X_train_subset_vec = vectorizer.transform(X_train_subset)
        classifier.fit(X_train_subset_vec, y_train_subset)

        # Evaluate on training data
        y_train_pred = classifier.predict(X_train_subset_vec)
        
        macro_precision_train = precision_score(y_train_subset, y_train_pred, average="macro", zero_division=0)
        macro_recall_train = recall_score(y_train_subset, y_train_pred, average="macro", zero_division=0)
        macro_f1_train = f1_score(y_train_subset, y_train_pred, average="macro", zero_division=0)
        train_precisions.append(macro_precision_train)
        train_recalls.append(macro_recall_train)
        train_f1s.append(macro_f1_train)

        # Evaluate on test data
        y_test_pred = classifier.predict(X_test_vec)
        macro_precision_test = precision_score(y_test, y_test_pred, average="macro", zero_division=0)
        macro_recall_test = recall_score(y_test, y_test_pred, average="macro", zero_division=0)
        macro_f1_test = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
        dev_precisions.append(macro_precision_test)
        dev_recalls.append(macro_recall_test)
        dev_f1s.append(macro_f1_test)

    # Plot learning curves
    plot_learning_curve(train_sizes, train_precisions, dev_precisions, "Precision")
    plot_learning_curve(train_sizes, train_recalls, dev_recalls, "Recall")
    plot_learning_curve(train_sizes, train_f1s, dev_f1s, "F1 Score")

if __name__ == "__main__":
    main()