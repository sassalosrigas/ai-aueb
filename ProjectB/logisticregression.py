import os
import re
import numpy as np
from collections import defaultdict,Counter
from typing import List, Dict, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
import random
import matplotlib.pyplot as plt

class DatabaseLoader:
   
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  #to direvtory sto opoio vriskomaste
    DATA_DIR = os.path.join(BASE_DIR, "aclImdb_v1", "aclImdb")  # to directory rwn dedomenwn

    TRAINING_POSITIVE = os.path.join(DATA_DIR, "train", "pos")  #antistoixo directory gia kathe paradeigma pou exoume
    TRAINING_NEGATIVE = os.path.join(DATA_DIR, "train", "neg")  
    TEST_POSITIVE = os.path.join(DATA_DIR, "test", "pos")       
    TEST_NEGATIVE = os.path.join(DATA_DIR, "test", "neg")       

    @staticmethod
    def load_reviews(directory: str) -> List[str]:
        """Gia kathe typo folder(training/test, positive/negative) antlei ta review kai ta katharizei vgazontas eidikous xarakthres, arithmous klp"""
        reviews = []
        for root, _, files in os.walk(directory):
            print("thodoris")
            for file in files:
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    content = f.read()
                    reviews.append(re.sub(r'[^a-zA-Z ]', ' ', content).lower())   #katharisma review
        return reviews

    @staticmethod
    def create_vocabulary(train_pos: List[str], train_neg: List[str], k: int, n: int) -> Dict[str, int]:
        """Dhmiourgia prwths morfhs vocabulary kai kopsimo spaniwn/sixnwn leksewn"""

        stopwords = {"the", "is", "in", "for", "where", "when", "to", "at"}  #lekseis opws arthra pou den prosdidoun nohma oi opoies prospernountai
        all_words = Counter()     #metrhths gia tis lekseis pou tha vroume

        for review in train_pos + train_neg:    #gia kathe review xwrise to se lekseis kai dialekse an tha mpei sto vocabulary
            words = review.split()
            all_words.update(word for word in words if word not in stopwords and len(word) > 1)

        #sorted_words = sorted(all_words.items(), key=lambda x: x[1])
        sorted_words = all_words.most_common()    #taksinomhsh twn leksevn
        sorted_words = sorted_words[k:len(sorted_words) - n]        #kopsimo k pio sixnwn kai n pio spaniwn leksewn

        vocabulary = {word: idx for idx, (word, _) in enumerate(sorted_words)}  #xtisimo vocabulary apo oti apemeine
        print(len(vocabulary))
        return vocabulary

    @staticmethod
    def finalize_vocabulary(vocabulary: Dict[str, int], train_pos: List[str], train_neg: List[str], m: int) -> Dict[str, int]:
        """Epeksergasia leksewn pou mikan sto vocabulary kai epeksergasia gia thn telikh morfh pou tha parei"""
        information_gain = {}    #domh pou periexei to information gain kathe lekshs
        total_reviews = len(train_pos) + len(train_neg)

        p_pos = len(train_pos) / total_reviews  #pososto thetikwn api tis sinolikes kritikes
        p_neg = len(train_neg) / total_reviews  #pososo arnhtikwn apo tis sinolikes kritikes
        h = DatabaseLoader.entropy(p_pos, p_neg)        #sinoliki entropia

        pos_word_counts = Counter(word for review in train_pos for word in review.split())  #count leksewn pou vriskontai se thetikes kritikes
        neg_word_counts = Counter(word for review in train_neg for word in review.split())  #count leksewn pou vriskontai se arnhtikes kritikes


        for word in vocabulary:         #gia kathe leksh sto leksiko
            print(word)         
            reviews_with_word_pos = pos_word_counts.get(word,0)  #thetikes kritikes pou periexoun th leksh        
            reviews_with_word_neg = neg_word_counts.get(word,0)  #arnhtikes kritikes pou periexoun th leksh
            reviews_without_word_pos = len(train_pos) - reviews_with_word_pos   #thetikes kritikes pou den periexoun th leksh
            reviews_without_word_neg = len(train_neg) - reviews_with_word_neg   #arnhtikes kritikes pou den periexoun th leksh

            reviews_with_word = reviews_with_word_pos + reviews_with_word_neg       #sinolikes kritikes pou periexoun th leksh
            reviews_without_word = reviews_without_word_pos + reviews_without_word_neg  #sinolikes kritikes pou den periexoun th leksh

            p_word = reviews_with_word / total_reviews      #pososto review pou periexoun th leksh
            p_not_word = reviews_without_word / total_reviews   #pososto review pou den periexoun th leksh

            h_word = DatabaseLoader.entropy(reviews_with_word_pos, reviews_with_word_neg)   #entropia review pou periexoun th leksh
            h_not_word = DatabaseLoader.entropy(reviews_without_word_pos, reviews_without_word_neg)     #entropia review pou den periexoun th leksh

            information_gain[word] = h - p_word * h_word - p_not_word * h_not_word      #information gain ths trexousas lekshs

        sorted_words = sorted(information_gain.items(), key=lambda x: x[1], reverse=True)   #sort by information gain
        final_vocabulary = {word: idx for idx, (word, _) in enumerate(sorted_words[:m])}    #kopsimo kai diathrhsh leksewn me to m megalitero information gain
        return final_vocabulary

    @staticmethod
    def entropy(positives: int, negatives: int) -> float:
        """Sinarthsh ypologismou entropias mias lekshs"""
        if positives + negatives == 0 or positives == 0 or negatives == 0:  #apokleismos senariwn pou dinoun sfalma/lathos apotelesma 
            return 0
        p_pos = positives / (positives + negatives)
        p_neg = negatives / (positives + negatives)
        return -p_pos * np.log2(p_pos) - p_neg * np.log2(p_neg)  #tupos entropias


class LogisticRegression:
    def __init__(self, learning_rate: float = 0.001, lambda_: float = 0.001, epochs: int = 80):
        """Arxikopoihsh parametrwn tou algorithmou"""
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.epochs = epochs
        self.weights = None

    def sigmoid(self, t: float) -> float:
        """Typos ths sigmoeidhs sunarthshs"""
        return 1 / (1 + np.exp(-t))

    def initialize_weights(self, size: int) -> np.ndarray:
        """Arxikopoihsh varwn gia kathe leksh me tuxaio tropo"""
        return np.random.rand(size) * 0.01

    def train(self, training_data: List[np.ndarray], training_labels: List[int], dev_data: List[np.ndarray], dev_labels: List[int]) -> None:
        """Ekpaideush tou algoritmou gia aukshsh akriveias provlepsewn"""
        self.weights = self.initialize_weights(len(training_data[0]))
        train_precision, train_recall, train_f1 = [], [], []
        dev_precision, dev_recall, dev_f1 = [], [], []

        for epoch in range(self.epochs):    #epanalhpsh diadikasias gia ton arithmo twn epoxwn
            total_loss = 0          #sinolikh apwleia tou algoritmo
            data_with_labels = list(zip(training_data, training_labels)) #zip data me labels gia na mhn xanetai h antistoixia review-label
            random.shuffle(data_with_labels)            #shuffle dedomenwn
            shuffled_data, shuffled_labels = zip(*data_with_labels)    

            for x, y in zip(shuffled_data, shuffled_labels):   #gia kathe zeugari dedomenwn
                t = np.dot(self.weights, x)
                y_pred = self.sigmoid(t)   #provlepsh kathgorias 
                example_loss = -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)  #apwleia paradeigmatos
                total_loss += example_loss
                self.weights += self.learning_rate * (y - y_pred) * x - self.learning_rate * self.lambda_ * self.weights  #enhmerwsh vaarous

            regularization_term = (self.lambda_ / 2) * np.sum(self.weights ** 2)    #prosthikh omalopoihshs
            total_loss += regularization_term
            print(f"Epoch {epoch + 1} - Avg Loss: {total_loss / len(training_data)}")

            train_preds = [self.predict(x) for x in training_data]
            train_precision.append(precision_score(training_labels, train_preds, zero_division=0))
            train_recall.append(recall_score(training_labels, train_preds, zero_division=0))
            train_f1.append(f1_score(training_labels, train_preds, zero_division=0))

            dev_preds = [self.predict(x) for x in dev_data]
            dev_precision.append(precision_score(dev_labels, dev_preds, zero_division=0))
            dev_recall.append(recall_score(dev_labels, dev_preds, zero_division=0))
            dev_f1.append(f1_score(dev_labels, dev_preds, zero_division=0))

        plt.figure(figsize=(12, 6))
        plt.plot(range(1, self.epochs + 1), train_precision, label="Train Precision")
        plt.plot(range(1, self.epochs + 1), dev_precision, label="Dev Precision")
        plt.plot(range(1, self.epochs + 1), train_recall, label="Train Recall")
        plt.plot(range(1, self.epochs + 1), dev_recall, label="Dev Recall")
        plt.plot(range(1, self.epochs + 1), train_f1, label="Train F1")
        plt.plot(range(1, self.epochs + 1), dev_f1, label="Dev F1")
        plt.xlabel("Epochs")
        plt.ylabel("Score")
        plt.title("Learning Curves")

    def predict(self, x: np.ndarray) -> int:
        """Methodos provlepshs kathgorias"""
        z = np.dot(self.weights, x)
        return 1 if self.sigmoid(z) >= 0.5 else 0

    def evaluate(self, test_data: List[np.ndarray], test_labels: List[str]) -> None:
        test_preds = [self.predict(x) for x in test_data]

        # Compute precision, recall, and F1 score for each class
        precision = precision_score(test_labels, test_preds, average=None)  # For both classes
        recall = recall_score(test_labels, test_preds, average=None)        # For both classes
        f1 = f1_score(test_labels, test_preds, average=None)                # For both classes

        # Compute micro-averaged and macro-averaged metrics
        micro_precision = precision_score(test_labels, test_preds, average='micro')
        micro_recall = recall_score(test_labels, test_preds, average='micro')
        micro_f1 = f1_score(test_labels, test_preds, average='micro')

        macro_precision = precision_score(test_labels, test_preds, average='macro')
        macro_recall = recall_score(test_labels, test_preds, average='macro')
        macro_f1 = f1_score(test_labels, test_preds, average='macro')

        # Print results
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

        plt.legend()
        plt.show()


    @staticmethod
    def convert_to_vector(review: str, vocabulary: Dict[str, int]) -> np.ndarray:
        """Metatroph review se vector"""
        words = review.split()  #xorisma review se lekseis
        vector = np.zeros(len(vocabulary)) #megethos vector iso me vocabulary arxika kathe thesh 0
        for word in words:
            if word in vocabulary:
                vector[vocabulary[word]] = 1  #gia kathe leksh tou vocabulary an periexetai sto review kane thn timh autou tou index 1
        return vector


def main():
    print("beggining")
    train_pos = DatabaseLoader.load_reviews(DatabaseLoader.TRAINING_POSITIVE) #fortosh dedomenwn kathe periptwshs apo to database
    print(1)
    train_neg = DatabaseLoader.load_reviews(DatabaseLoader.TRAINING_NEGATIVE)
    print(2)
    test_pos = DatabaseLoader.load_reviews(DatabaseLoader.TEST_POSITIVE)
    print(3)
    test_neg = DatabaseLoader.load_reviews(DatabaseLoader.TEST_NEGATIVE)
    print("end") 
    # Create vocabulary
    temp_vocabulary = DatabaseLoader.create_vocabulary(train_pos, train_neg, 750,1000) #prwth morfh vocabulary
    print("temp")
    vocabulary = DatabaseLoader.finalize_vocabulary(temp_vocabulary, train_pos, train_neg, 30000)   #telikh morfh vocabulary
    print("vocab ready")
    # Prepare training data
    training_data = [LogisticRegression.convert_to_vector(review, vocabulary) for review in train_pos + train_neg] #metatroph se vector olwn twn data tou training
    training_labels = [1] * len(train_pos) + [0] * len(train_neg)  #label timh 1 sta thetika kai 0 sta arnhtika
    print("begin training")
    # Train model
    test_data = [LogisticRegression.convert_to_vector(review, vocabulary) for review in test_pos + test_neg]
    test_labels = [1] * len(test_pos) + [0] * len(test_neg)


    sklearn_model = SklearnLogisticRegression()
    sklearn_model.fit(training_data, training_labels)

    # Evaluate your mode
    # Evaluate Scikit-learn's model
    sklearn_preds = sklearn_model.predict(test_data)
    sklearn_precision = precision_score(test_labels, sklearn_preds, average='binary', pos_label=1)
    sklearn_recall = recall_score(test_labels, sklearn_preds, average='binary', pos_label=1)
    sklearn_f1 = f1_score(test_labels, sklearn_preds, average='binary', pos_label=1)

    print("Scikit-learn's Logistic Regression:")
    print(f"Precision: {sklearn_precision:.4f}, Recall: {sklearn_recall:.4f}, F1: {sklearn_f1:.4f}")
    print("Shuffling training data...")
    training_data = np.array(training_data)
    training_labels = np.array(training_labels)
    shuffle_indices = np.random.permutation(len(training_data))  # Generate random indices for shuffling
    training_data = training_data[shuffle_indices]  # Shuffle data
    training_labels = training_labels[shuffle_indices]  # Shuffle labels

    split_idx = int(0.8 * len(training_data))  # 80% training, 20% development
    dev_data, dev_labels = training_data[split_idx:], training_labels[split_idx:]
    training_data, training_labels = training_data[:split_idx], training_labels[:split_idx]
    # Train model
    print("Training model...")
    model = LogisticRegression()
    model.train(training_data, training_labels, dev_data, dev_labels)

    # Evaluate model
    print("Evaluating model...")
    model.evaluate(test_data, test_labels)

if __name__ == "__main__":
    main()