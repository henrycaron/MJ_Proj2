import pandas as pd
import numpy as np
import os
import sklearn
import sklearn.tree
from load_word_embeddings import load_word_embeddings

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
        for punc in ['?', '!', '_', '.', ',', '"', '/']:
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
    
def cleaned_text_list(reviews, word2vec): # converts review string to list of strings with irrelevant words removed
    tr_text_list = reviews['text'].values.tolist()
    # tok_count_dict = most_common_words(tr_text_list)
    ret_list = list()
    for line in range(len(tr_text_list)):
        str_list = tokenize_text(tr_text_list[line])
        vec_list = list()
        for word in str_list:
            if word in word2vec:
                vec_list.append(word2vec[word])
        ret_list.append(np.asarray(vec_list))
    return np.asarray(ret_list)

def preprocess(vector):
    for i in range(len(vector)):
        vector[i] = np.asarray(np.mean(vector[i], axis = 0))
    return vector
    
def data_fetch_and_clean(word2vec):
    data_dir = 'data_reviews'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))
    
    x_train_N_ = cleaned_text_list(x_train_df, word2vec)
    x_test_N_ = cleaned_text_list(x_test_df, word2vec)
    
    x_train_NF = preprocess(x_train_N_)
    x_test_NF = preprocess(x_test_N_)
    print(x_train_NF.shape)
    
    print(x_test_NF[:20])
    print(x_test_NF.shape)
    
    return x_train_df, y_train_df, x_test_df

if __name__ == "__main__":
    word2vec = load_word_embeddings()
    x_train_df, y_train_df, x_test_df = data_fetch_and_clean(word2vec)
    