#/usr/bin/python3
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import os
from string import punctuation 
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline, FeatureUnion
import numpy as np
import time

import nltk
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from collections import defaultdict
from nltk.tag import pos_tag
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB

import math as m
import pandas as pd
from sklearn.svm import SVC

# nltk.download()
file_names = []
DATA_DIR = ['positive_polarity', 'negative_polarity']
sub_dirs = []
for sub_dir in DATA_DIR:
    sub_dirs = sub_dirs + [os.path.join(sub_dir, f) for f in os.listdir(sub_dir)  if f.startswith('.') is False]
tmp = []
for sub_dir in sub_dirs:
    tmp = tmp + [os.path.join(sub_dir, f) for f in os.listdir(sub_dir)  if f.startswith('.') is False]
sub_dirs = tmp
for sub_dir in sub_dirs:
    file_names = file_names + [os.path.join(sub_dir, f) for f in os.listdir(sub_dir)  if f.startswith('.') is False]

# print(sub_dirs)
print(len(file_names))


class Email(object):
    def __init__(self, data_path=None):
        self.type = 0
        self.words = 0
        self.sentences = []
        self.lemmatizer = Lemmatization()
        if data_path is not None:
            self.load_from_file(data_path)

    def remove_stopwords(self, email):
        sentence = email.split(' ')
        stop_words = set(stopwords.words('english'))
        new_email = []
        for word in sentence:
            for char in word:
                if ord(char) < 97 or ord(char) > 122:
                    word = word.replace(char, '')
            if word not in stop_words and len(word) > 0: 
                new_email.append(word)
        return new_email

    def load_from_file(self, data_path):
        data_path_name = data_path.split('\\') # How you split might differ depending on your programming software/machine.
        # print(data_path)
        data_path_name_type = data_path_name[1][0]
        # 1 represents spam email
        self.type = 1 if data_path_name_type == 'd' else 0
        with open(data_path) as f:
            for line in f.readlines():
                email = line.lower()     
                email = self.remove_stopwords(email)
                for punc in punctuation:
                    if punc in email:
                        if punc != '-': email = email.replace(punc, '')  
                        else: email = email.replace(punc, ' ');         
                new_email = self.lemmatizer.get_lemmatized_email(email)
                self.sentences = new_email
                self.words = len(self.sentences)
        return self.sentences

class Lemmatization(object):
    def __init__(self):
        pass

    def get_pos_tag(self, nltk_tag):  
        if nltk_tag.startswith('J') or nltk_tag.startswith('A'):
            return wordnet.ADJ
        elif nltk_tag.startswith('S'):
            return wordnet.ADJ_SAT
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:          
            return None
    
    def get_lemmatized_email(self, email):
        lemma = WordNetLemmatizer()
        # Split each line into a list of tokens
        sentence = email
        # Creates a list of tuples. First element is the word, second is the pos_tag
        pos_tags = nltk.pos_tag(sentence)
        # Iterate through each tuple in the list.
        new_email = []
        for tag in pos_tags: 
            word = tag[0]
            pos_tag = self.get_pos_tag(tag[1]) or wordnet.NOUN
            new_email.append(lemma.lemmatize(word, pos_tag))
        return new_email

"""
class Matrix(object):
    def __init__(self):
        self.dictionary = []
        self.tf_list = []
        self.df = Counter()
        self.idf = Counter()
        self.tf_idf = Counter()
    
    def computeTF(self, file):
        tf = Counter()
        for word in file:
            tf[word] += 1
        keys = tf.keys()
        for word in keys:
            tf[word] /= len(file)
        self.tf_list.append(tf)

    def computeDF(self, cnt, file):
        for word in file:
            if (cnt[word] == 0):
                self.df[word] += 1
            cnt[word] += 1

    def createFreqs(self, filenames):
        for file in filenames:
            self.computeTF(file)
            self.computeDF(Counter(), file)

    def computeIDF(self, filenames):
        keys = self.df.keys()
        tf_idf = Counter()
        for key in keys:
            self.idf[key] = m.log(len(filenames)/(self.df[key] + 1), 2)

    def createMatrix(self, index):
        tf_idf = Counter()
        for word in self.tf_list[index]:
            tf_idf[word] = self.tf_list[index][word] * self.idf[word]
        self.dictionary.append(tf_idf)

    def createDictionary(self, filenames):
        for index in range(len(filenames)):
            self.createMatrix(index)

    def getDictionary(self):
            return self.dictionary
    """
class Dataset():
    def __init__(self, file_names = None):
        self.file_names = file_names
        
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = file_names[idx]
        email = Email(file_name)
        sentences = email.sentences
        label = email.type
        num_words = email.words
        #return  sentences, label, num_words
        return sentences, label
    
    def getTextLength(self, data):
         return np.array([len(text) for text in data]).reshape(-1, 1)

def main():
    data = []
    total_sentences = ""
    dataset = Dataset(file_names)
    X = []
    y = []
    for file in file_names:
        email = Email(file)
        data.append(email.sentences)
        y.append(email.type)

    for sentence in data: 
        X.append(" ".join(sentence))
        total_sentences += " ".join(sentence)
    #1st index is class labels
    train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True)
    Encoder = LabelEncoder()
    train_labels = Encoder.fit_transform(train_labels)
    test_labels = Encoder.fit_transform(test_labels)
    pipeline = Pipeline([('vectorizer', CountVectorizer(max_features=5000,   
                              ngram_range=(1, 2))),
                              ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True))])
    train_data = pipeline.fit_transform(train_data, train_labels)
    test_data = pipeline.fit_transform(test_data, test_labels)
    
    """
    classifier = Pipeline([
        ('length', FunctionTransformer(dataset.getTextLength(total_sentences), validate=False)),
        ('SVM', svm.SVC())])
    """
    # Creates best approximations for hyperparameters
    """
    param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['linear']} 
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
    grid.fit(train_data, train_labels)
    print(grid.best_params_)
    print(grid.best_estimator_)
    """
    classifier = svm.SVC(C=1, kernel="rbf", gamma=1)
    trainStart = time.time()
    classifier.fit(train_data, train_labels)
    trainEnd = time.time()
    predict_train = classifier.predict(train_data)
    predict_test = classifier.predict(test_data)
    trainTime = trainEnd-trainStart
    print("SVM Training Accuracy Score: " , accuracy_score(predict_train, train_labels)*100, " Training time for train data: ", trainTime, "s")
    print("SVM Testing Accuracy Score: " , accuracy_score(predict_test, test_labels)*100, " Training time for test data: ", trainTime, "s")

    # model = Word2Vec(total_sentences, size=100, min_count=1, workers=4, sg = 1)
    # vector = model.wv['bag']
    # print('the vector representation of bag:', vector) 






if __name__ == '__main__':
    main()







