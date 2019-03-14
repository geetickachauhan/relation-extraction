# -*- coding: utf-8 -*-
from itertools import combinations

from nltk import ngrams
#from sklearn.cross_validation import train_test_split
#from sklearn.metrics import classification_report
import numpy as np

dataset_dictionary = None
top_word_pair_features = None
top_syntactic_grammar_list = None

trained_model_pickle_file = 'trained_model.pkl'


def get_empty_vector(n):
    return [0 for _ in range(n)]


def get_top_word_dataset_dictionary():
    from feature_vector import get_dataset_dictionary

    global dataset_dictionary
    if dataset_dictionary is None:
        dataset_dictionary = get_dataset_dictionary()
    return dataset_dictionary


def get_top_word_pair_features():
    from feature_vector import extract_top_word_pair_features

    global top_word_pair_features
    if top_word_pair_features is None:
        top_word_pair_features = extract_top_word_pair_features()
    return top_word_pair_features


def get_top_syntactic_grammar_list():
    from feature_vector import extract_top_syntactic_grammar_trio

    global top_syntactic_grammar_list
    if top_syntactic_grammar_list is None:
        top_syntactic_grammar_list = extract_top_syntactic_grammar_trio()
    return top_syntactic_grammar_list


def get_word_feature(normalized_sentence):
    unique_tokens = set(word for word in normalized_sentence.split())
    # exclude duplicates in same line and sort to ensure one word is always before other
    bi_grams = set(ngrams(normalized_sentence.split(), 2))
    words = unique_tokens | bi_grams
    dataset_dictionary = get_top_word_dataset_dictionary()
    X = [i if j in words else 0 for i, j in enumerate(dataset_dictionary)]
    return X


def get_frequent_word_pair_feature(normalized_sentence):
    unique_tokens = sorted(set(word for word in normalized_sentence.split()))
    # exclude duplicates in same line and sort to ensure one word is always before other
    combos = combinations(unique_tokens, 2)
    top_word_pair_features = get_top_word_pair_features()
    X = [i if j in combos else 0 for i, j in enumerate(top_word_pair_features)]
    return X


def get_syntactic_grammar_feature(sentence_text):
    from feature_vector import extract_syntactic_grammar
    trigrams_list = extract_syntactic_grammar(sentence_text)
    top_syntactic_grammar_list = get_top_syntactic_grammar_list()
    X = [i if j in trigrams_list else 0 for i, j in enumerate(top_syntactic_grammar_list)]
    return X


def make_feature_vector(row):
    normalized_sentence = row.normalized_sentence
    sentence = row.sentence_text

    word_feature = get_word_feature(normalized_sentence)
    frequent_word_feature = get_frequent_word_pair_feature(normalized_sentence)
    syntactic_grammar_feature = get_syntactic_grammar_feature(sentence)

    features = word_feature
    features.extend(frequent_word_feature)
    features.extend(syntactic_grammar_feature)
    return features


#def main():
#    from dataset.read_dataset import get_dataset_dataframe
#    df = get_dataset_dataframe()
#    X, Y = extract_training_data_from_dataframe(df)
#    from sklearn.svm import SVC
#    X_train, X_test, y_train, y_test = \
#        train_test_split(X, Y, test_size=.2, random_state=42)
#
#    print(df.head())
#    print('X: ', (X.shape), 'Y : ', np.array(Y.shape))
#    model = SVC(kernel='linear')
#    model.fit(X_train, y_train)
#    score = model.score(X_test, y_test)
#    import pandas as pd
#
#    pd.to_pickle(model, trained_model_pickle_file)
#    # classification_report()
#    print('Score : ', score)
#    y_pred = model.predict(X_test)
#    print(classification_report(y_test, y_pred))


def extract_data_from_dataframe(df):
    from read_dataset import get_training_label

    X = df.apply(make_feature_vector, axis=1)
    Y = df.apply(get_training_label, axis=1)
    X = np.array(X.tolist())
    Y = np.array(Y.tolist())
    return X, Y
