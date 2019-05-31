'''
Author: Geeticka Chauhan
Accompanies the notebooks/Data-Exploration ipynb notebooks
'''

import pandas as pd
from collections import Counter

# Get the entity pair dictionary with the dataframe
def get_entity_pair_dict_with_df(df, dict_of_e1_e2):
    def combine_e1_e2(row):
        e1 = row.e1.lower()
        e2 = row.e2.lower()
        pair = (e1, e2)
        return pair
    df['e1_and_e2'] = df.apply(combine_e1_e2, axis=1)
    unique_pairs = list(df['e1_and_e2'])

    for pair in unique_pairs:
        if pair in dict_of_e1_e2:
            dict_of_e1_e2[pair] += 1
        elif (pair[1], pair[0]) in dict_of_e1_e2:
            dict_of_e1_e2[(pair[1], pair[0])] += 1
        else:
            dict_of_e1_e2[pair] += 1
    return df, dict_of_e1_e2

# get the dictionary for the entity pair
def convert_pair_to_dict(row, needed_dict):
    pair = row['e1_and_e2']
    if pair in needed_dict:
        pair_idx = needed_dict[pair]
    elif (pair[1], pair[0]) in needed_dict:
        pair_idx = needed_dict[(pair[1], pair[0])]
    else:
        print('This scenario should not have happened')
    return pair_idx

# Get the entity dictionary and update the dataframe with the pair map referring to the entity
def get_entity_dict_df_pair_map(df_train, df_test):
    dict_of_e1_e2 = Counter()
    df_train, dict_of_e1_and_e2 = get_entity_pair_dict_with_df(df_train, dict_of_e1_e2)
    df_test, dict_of_e1_and_e2 = get_entity_pair_dict_with_df(df_test, dict_of_e1_e2)
    ls = dict_of_e1_e2.most_common()
    needed_dict = {w[0]: index for (index, w) in enumerate(ls)}
    df_train['pair_map'] = df_train.apply(convert_pair_to_dict, args=(needed_dict,), axis=1)
    df_test['pair_map'] = df_test.apply(convert_pair_to_dict, args=(needed_dict,), axis=1)
    return needed_dict, df_train, df_test

# calculate the length of the context
def length_of_context(row):
    e1 = row.metadata['e1']['word_index'][0][1]
    e2 = row.metadata['e2']['word_index'][0][1]
    distance = abs(int(e1) - int(e2))
    return distance

# calculate the length of the sentence
def length_of_sentence(row):
    tokenized_sentence = row.tokenized_sentence.split()
    return len(tokenized_sentence)
