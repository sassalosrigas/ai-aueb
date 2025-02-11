from typing import Dict, List
import numpy as np
import os
import re
from collections import Counter

class DatabaseLoader:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #to directory sto opoio vriskomaste
    DATA_DIR = os.path.join(BASE_DIR, "aclImdb_v1") # to directory rwn dedomenwn
    TRAINING_POSITIVE = os.path.join(DATA_DIR, "train", "pos") #antistoixo directory gia kathe paradeigma pou exoume
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

class NaiveBayesClassifier:
    def __init__(self, vocabulary: Dict[str, int]):
        self.vocabulary = vocabulary
        self.prior = {}
        self.conditional_probs = {}

    def train(self, train_pos: List[str], train_neg: List[str]):
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

class main:
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
    classifier.train(train_pos, train_neg)

    print("Predicting...")
    for review in test_pos + test_neg:
        print(classifier.predict(review))
    print("Predictions done.")