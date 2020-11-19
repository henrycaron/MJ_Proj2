import pandas as pd
import numpy as np
import os
import sklearn
import sklearn.tree
import sklearn.ensemble
from pprint import pprint 
from load_word_embeddings import load_word_embeddings
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
    
def data_fetch_and_clean(word2vec):
    data_dir = 'data_reviews'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

    top_tokens = ("the", "and", "i", "a", "is", "this", "it", "of", "to", 
                  "was", "for", "that", "in", "was", "for", "that", "my", "on",
                  "as", "its", "are", "have", "at", "be")
    x_train_N_ = cleaned_text_list(x_train_df, word2vec, top_tokens)
    x_test_N_ = cleaned_text_list(x_test_df, word2vec, top_tokens)
    
    x_train_NF = preprocess(x_train_N_)
    x_test_NF = preprocess(x_test_N_)
    # print(x_train_NF.shape)
    
    # print(x_test_NF[:20])
    # print(x_test_NF.shape)
    
    return x_train_NF, y_train_df, x_test_NF
    
if __name__ == "__main__":
    word2vec = load_word_embeddings()
    x_train_df, y_train_df, x_test_df = data_fetch_and_clean(word2vec)
    
        # inp = input()
        # print(inp, np.linalg.norm(word2vec[inp] - word2vec["bad"]))

    #tree model
    # tree = sklearn.tree.DecisionTreeClassifier(
    #     criterion='gini', min_samples_split=2, min_samples_leaf=1)
    # hyperparameter_grid_by_name = dict(
    #     max_depth=[32, 128, 256],
    #     min_samples_leaf=[1, 3, 9],
    #     )
    # grid = sklearn.model_selection.GridSearchCV(
    #     tree,
    #     hyperparameter_grid_by_name,
    #     scoring='balanced_accuracy',
    #     cv=7,
    #     return_train_score=True)
    
    # grid.fit(x_train_df, y_train_df['is_positive_sentiment'])
    # tree_search_results_df = pd.DataFrame(grid.cv_results_).copy()
    # print(grid.best_params_)
    # print(grid.best_score_)
    
    # forest model
    forest_hyperparameter_grid_by_name = dict(
        max_features=[3, 10, 20, 35, 49],
        max_depth=[16, 32],
        min_samples_leaf=[1],
        n_estimators=[125],
        random_state=[101],
    )
    forest = sklearn.ensemble.RandomForestClassifier(
        n_estimators=125,
        criterion='gini',
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1)
    
    forest_searcher = sklearn.model_selection.GridSearchCV(
        forest,
        forest_hyperparameter_grid_by_name,
        scoring='balanced_accuracy',
        cv=7,
        return_train_score=True,
        refit=False)
        
    forest_searcher.fit(x_train_df, y_train_df['is_positive_sentiment'])
    
    forest_search_results = pd.DataFrame(forest_searcher.cv_results_).copy()
    print(forest_searcher.best_params_)
    print(forest_searcher.best_score_)
    
    # lasso model
    # lasso = sklearn.linear_model.LogisticRegression(
    #     penalty='l1', solver='saga', random_state=101)
    # lasso_hyperparameter_grid_by_name = dict(
    #     C=np.logspace(-4, 4, 9),
    #     max_iter=[20, 40], # sneaky way to do "early stopping" 
    #                        # we'll take either iter 20 or iter 40 in training process, by best valid performance
    #     )
        
    # lasso_searcher = sklearn.model_selection.GridSearchCV(
    #     lasso,
    #     lasso_hyperparameter_grid_by_name,
    #     scoring='balanced_accuracy',
    #     cv=7,
    #     return_train_score=True,
    #     refit=False)
    
    # lasso_searcher.fit(x_train_df, y_train_df['is_positive_sentiment'])
    
    # lasso_search_results_df = pd.DataFrame(lasso_searcher.cv_results_).copy()
    # print(lasso_searcher.best_params_)
    # print(lasso_searcher.best_score_)