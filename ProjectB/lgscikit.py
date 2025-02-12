from logisticregression import DatabaseLoader, LogisticRegression
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
    
def main():
    """eksagogh apotelesmatwn me xrhsh tou etoimou algoriyhmou tou scikit"""
    print("Beginning")
    train_pos = DatabaseLoader.load_reviews(DatabaseLoader.TRAINING_POSITIVE)
    print(1)
    train_neg = DatabaseLoader.load_reviews(DatabaseLoader.TRAINING_NEGATIVE)
    print(2)
    test_pos = DatabaseLoader.load_reviews(DatabaseLoader.TEST_POSITIVE)
    print(3)
    test_neg = DatabaseLoader.load_reviews(DatabaseLoader.TEST_NEGATIVE)
    print("End")

    temp_vocabulary = DatabaseLoader.create_vocabulary(train_pos, train_neg, 750, 1000)
    print("Temp vocabulary ready")
    vocabulary = DatabaseLoader.finalize_vocabulary(temp_vocabulary, train_pos, train_neg, 30000)
    print("Final vocabulary ready")

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
    #mexri edw to kommati idio me thn dikia mas ulopoihsh wste na akolouthithei akrivwss h idia diadikasia proepeksegasias
    print("Training model...")
    sklearn_model = SklearnLogisticRegression(C = 0.001, max_iter=1, penalty = "l2", warm_start=True )  #arxikopoihsh etoimou logistic regression
    
    train_precision, train_recall, train_f1 = [], [], []
    dev_precision, dev_recall, dev_f1 = [], [], []
    epochs = 20
    for i in range(epochs):
        print("ei")                                             
        sklearn_model.fit(training_data, training_labels)  #train gia ena iteration tou algorithmou th fora
        train_preds = sklearn_model.predict(training_data)         #append katallhlwn statistikwn gia ta plots
        train_precision.append(precision_score(training_labels, train_preds, pos_label=1, zero_division=0))
        train_recall.append(recall_score(training_labels, train_preds, pos_label=1, zero_division=0))
        train_f1.append(f1_score(training_labels, train_preds, pos_label=1, zero_division=0))
        dev_preds = sklearn_model.predict(dev_data)
        dev_precision.append(precision_score(dev_labels, dev_preds, pos_label=1, zero_division=0))
        dev_recall.append(recall_score(dev_labels, dev_preds, pos_label=1, zero_division=0))
        dev_f1.append(f1_score(dev_labels, dev_preds, pos_label=1, zero_division=0))

    plt.figure(figsize=(12, 6))     #set up twn kampulwn
    plt.plot(range(1, epochs + 1), train_precision, label="Train Precision")
    plt.plot(range(1, epochs + 1), dev_precision, label="Dev Precision")
    plt.plot(range(1, epochs + 1), train_recall, label="Train Recall")
    plt.plot(range(1, epochs + 1), dev_recall, label="Dev Recall")
    plt.plot(range(1, epochs + 1), train_f1, label="Train F1")
    plt.plot(range(1, epochs + 1), dev_f1, label="Dev F1")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("Score Trajectory")
    print("Evaluating model...")
    predictions = sklearn_model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)

    print(f"Test Accuracy: {accuracy:.4f}")
    print("Classification Report:") 
    print(report)       #print upoloipwn statistikwn

    plt.legend()
    plt.show()      #emfanish plot
    

if __name__ == "__main__":
    main()