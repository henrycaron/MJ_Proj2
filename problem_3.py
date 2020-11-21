import pandas as pd
import numpy as np
import os
import sklearn
import sklearn.tree
import sklearn.ensemble
from pprint import pprint 
from load_word_embeddings import load_word_embeddings
import nltk
from nltk.tokenize import word_tokenize # or use some other tokenizer
import random
# from keras.preprocessing.sequence import pad_sequences


def tokenize_text(raw_text):
    ''' Transform a plain-text string into a list of tokens
    
    We assume that *whitespace* divides tokens.
    
    Args
    ----
    raw_text : string
    
    Returns
    -------
    list_of_tokens : list of strings
        Each element is one token in the provided text
    '''
    list_of_tokens = raw_text.split() # split method divides on whitespace by default
    for pp in range(len(list_of_tokens)):
        cur_token = list_of_tokens[pp]
        # Remove punctuation
        for punc in ['?', '!', '_', '.', ',', '"', '/', "'", "(", ")", "--", ":", "-"]:
            cur_token = cur_token.replace(punc, "")
        # Turn to lower case
        clean_token = cur_token.lower()
        # Replace the cleaned token into the original list
        list_of_tokens[pp] = clean_token
    return list_of_tokens

def most_common_words(tr_text_list):
    tok_count_dict = dict()

    for line in tr_text_list:
        tok_list = tokenize_text(line)
        for tok in tok_list:
            if tok in tok_count_dict:
                tok_count_dict[tok] += 1
            else:
                tok_count_dict[tok] = 1
    return tok_count_dict
    
def cleaned_text_list(reviews, word2vec, top_tokens): # converts review string to list of strings with irrelevant words removed
    tr_text_list = reviews['text'].values.tolist()
    
    ret_list = list()
    non_words = set()
    for line in range(len(tr_text_list)):
        str_list = tokenize_text(tr_text_list[line])
        vec_list = list()
        for word in str_list:
            # print(word)
            if word in word2vec and word not in top_tokens:
                vec_list.append(word2vec[word])
            else:
                non_words.add(word)
        ret_list.append(np.asarray(vec_list))
    # print(non_words)
    return np.asarray(ret_list)

def preprocess(vector):
    N, = vector.shape
    output = np.zeros([N, 50])
    for i in range(len(vector)):
        output[i] = np.asarray(np.mean(vector[i], axis = 0))
    return output
    
def data_fetch_and_clean():
    data_dir = 'data_reviews'
    x_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    x_df = x_df['text'].values.tolist()
    random.shuffle(x_df, 1)
    y_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    y_df = y_df['is_positive_sentiment'].values.tolist()
    random.shuffle(y_df, 1)
    for line in range(len(x_df)):
        x_df[line] = tokenize_text(x_df[line])

    
    x_train_df = x_df[:2000]
    x_test_df = x_df[2000:]
   
    y_train_df = y_df[:2000]
    y_test_df = y_df[2000:]
    
    # print(y_train_df)
    # for line in range(len(x_test_df)):
    #     x_test_df[line] = tokenize_text(x_test_df[line])
    training_set = [(i,j) for i,j in zip(x_train_df, y_train_df)]
    print(training_set[0:20])
    # print(training_set[:20])
    all_words = set()
    for line in x_train_df:
        for word in line:
            all_words.add(word)
    
    print(x_test_df[0])

    tr   = [({word: (word in x[0]) for word in all_words}, x[1]) for x in training_set]
    test = [{word: (word in x)     for word in all_words}        for x in x_test_df]
    print(len(test))
    # pprint(t[0])
    return tr, test, x_test_df, y_test_df
    
if __name__ == "__main__":
    training_set, x_test_df, x_test_orig_df, y_test_df = data_fetch_and_clean()
    # labeled_names = ([(name, 'male') for name in names.words('male.txt')] + \
    # [(name, 'female') for name in names.words('female.txt')])
    # featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
    # train_set, test_set = featuresets[500:], featuresets[:500]
    # pprint(train_set)
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    classifier.show_most_informative_features(15)

    f = open("predictions_prob_3.txt", "w")

    y_hat_test_df = []
    for i in range(len(x_test_df)):
        prediction = (classifier.classify(x_test_df[i]))
        y_hat_test_df.append(prediction)
        # print(x_test_orig_df[i], "pred: ", prediction, "actual: ", y_test_df[i]) 
        # f.write(str(prediction.prob(1)))
        # f.write("\n")
        # print("%s: %f" % ("1", prediction.prob(1)))
        
    print(sklearn.metrics.accuracy_score(y_test_df, y_hat_test_df))
    f.close()

    