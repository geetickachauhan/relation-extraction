import os, pandas as pd, numpy as np
import nltk
import spacy
from spacy.tokens import Doc
nlp = spacy.load('en')

# important global variables for identifying the location of entities
entity1 = 'E'
entity2 = 'EOTHER'
entity_either = 'EEITHER'

'''
The methods below are for the preprocessing type 1 
'''
# separate the indexes of entity 1 and entity 2 by what is intersecting 
# and what is not
def get_common_and_separate_entities(e1_indexes, e2_indexes):
    e1_indexes = set(e1_indexes)
    e2_indexes = set(e2_indexes)
    common_indexes = e1_indexes.intersection(e2_indexes)
    only_e1_indexes = list(e1_indexes.difference(common_indexes))
    only_e2_indexes = list(e2_indexes.difference(common_indexes))
    
    return only_e1_indexes, only_e2_indexes, list(common_indexes)


# given an entity replacement dictionary like {'0:0': 'entity1'} 
# provide more information related to the location of the entity
def entity_replacement_dict_with_entity_location(entity_replacement_dict, 
                                                 only_e1_indexes, only_e2_indexes, common_indexes):
    def update_dict_with_indexes(new_entity_replacement_dict, only_indexes, start, end):
        for i in only_indexes:
            key = str(i[0]) + ':' + str(i[-1])
            new_entity_replacement_dict[key]['start'] = start
            new_entity_replacement_dict[key]['end'] = end
        return new_entity_replacement_dict
        
    new_entity_replacement_dict = {} 
    # below is just for initialization purposes, when start and end is none, means we are not 
    # inserting anything before or after those words in the sentence
    for key in entity_replacement_dict.keys():
        new_entity_replacement_dict[key] = {'replace_by': entity_replacement_dict[key], 
                                      'start': None, 'end': None}
    new_entity_replacement_dict = update_dict_with_indexes(new_entity_replacement_dict, only_e1_indexes,
                                                          entity1 + 'START', entity1 + 'END')
    new_entity_replacement_dict = update_dict_with_indexes(new_entity_replacement_dict, only_e2_indexes,
                                                           entity2 + 'START', entity2 + 'END')
    new_entity_replacement_dict = update_dict_with_indexes(new_entity_replacement_dict, common_indexes,
                                                           entity_either + 'START', entity_either + 'END')
    return new_entity_replacement_dict

###
### Helper functions
###
#given string 12:30, return 12, 30 as a tuple of ints
def parse_position(position):
    positions = position.split(':')
    return int(positions[0]), int(positions[1])

def sort_position_keys(entity_replacement_dict):
    positions = list(entity_replacement_dict.keys())
    sorted_positions = sorted(positions, key=lambda x: int(x.split(':')[0]))
    return sorted_positions

# remove any additional whitespace within a line
def remove_whitespace(line):
    return str(" ".join(line.split()).strip())

def list_to_string(sentence):
    return " ".join(sentence)

# adapted from tag_sentence method in converter_ddi
# note that white spaces are added in the new sentence on purpose
def replace_with_concept(row):
    sentence = row.tokenized_sentence.split()
    e1_indexes = row.metadata['e1']['word_index']
    e2_indexes = row.metadata['e2']['word_index'] # assuming that within the same entity indexes, no overlap
    new_sentence = ''
    only_e1_indexes, only_e2_indexes, common_indexes = \
    get_common_and_separate_entities(e1_indexes, e2_indexes)
    
    entity_replacement_dict = row.metadata['entity_replacement'] # assuming no overlaps in replacement
    
    new_entity_replacement_dict = entity_replacement_dict_with_entity_location(entity_replacement_dict, 
                                                                               only_e1_indexes, only_e2_indexes, 
                                                                               common_indexes)
    repl_dict = new_entity_replacement_dict # just using proxy because names are long
    sorted_positions = sort_position_keys(new_entity_replacement_dict)
    for i in range(len(sorted_positions)):
        curr_pos = sorted_positions[i]
        curr_start_pos, curr_end_pos = parse_position(curr_pos)
        start_replace = '' if repl_dict[curr_pos]['start'] is None else repl_dict[curr_pos]['start'].upper()
        end_replace = '' if repl_dict[curr_pos]['end'] is None else repl_dict[curr_pos]['end'].upper()
        between_replace = repl_dict[curr_pos]['replace_by'].upper() # between the entity replacement
        if i == 0:
            new_sentence += list_to_string(sentence[:curr_start_pos]) + ' ' + start_replace + ' ' + \
            between_replace + ' ' + end_replace + ' '
        else:
            prev_pos = sorted_positions[i-1]
            _, prev_end_pos = parse_position(prev_pos)
            middle = list_to_string(sentence[prev_end_pos+1 : curr_start_pos]) # refers to middle between prev segment and the 
            # current segment
            if middle == '':
                middle = ' '
            new_sentence += middle + ' ' + start_replace + ' ' + between_replace + ' ' + end_replace + ' '
            if i == len(sorted_positions) - 1 and curr_end_pos < len(sentence) - 1:
                new_sentence += ' ' + list_to_string(sentence[curr_end_pos+1:])
    new_sentence = remove_whitespace(new_sentence)
    return new_sentence


'''
Preprocessing Type 2: Removal of stop words, punctuations and the replacement of digits
'''
# gives a dictionary signifying the location of the different entities in the sentence
def get_entity_location_dict(only_e1_indexes, only_e2_indexes, common_indexes):
    entity_location_dict = {}
    def update_dict_with_indexes(entity_location_dict, only_indexes, start, end):
        for i in only_indexes:
            key = str(i[0]) + ':' + str(i[-1])
            entity_location_dict[key] = {'start': start, 'end': end}
        return entity_location_dict
    entity_location_dict = update_dict_with_indexes(entity_location_dict, only_e1_indexes, 
                                                    entity1 + 'START', entity1 + 'END')
    entity_location_dict = update_dict_with_indexes(entity_location_dict, only_e2_indexes, 
                                                    entity2 + 'START', entity2 + 'END')
    entity_location_dict = update_dict_with_indexes(entity_location_dict, common_indexes, 
                                                    entity_either + 'START', entity_either + 'END')
    return entity_location_dict

# given the index information of the entities, return the sentence with 
# tags ESTART EEND etc to signify the location of the entities
def get_new_sentence_with_entity_replacement(sentence, e1_indexes, e2_indexes):
    new_sentence = ''
    only_e1_indexes, only_e2_indexes, common_indexes = \
        get_common_and_separate_entities(e1_indexes, e2_indexes)
    entity_loc_dict = get_entity_location_dict(only_e1_indexes, only_e2_indexes, common_indexes)
    sorted_positions = sort_position_keys(entity_loc_dict)
    for i in range(len(sorted_positions)):
        curr_pos = sorted_positions[i]
        curr_start_pos, curr_end_pos = parse_position(curr_pos)
        start_replace = entity_loc_dict[curr_pos]['start']
        end_replace = entity_loc_dict[curr_pos]['end']
        if i == 0:
            new_sentence += list_to_string(sentence[:curr_start_pos]) + ' ' + start_replace + ' ' + \
            list_to_string(sentence[curr_start_pos : curr_end_pos + 1]) + ' ' + end_replace + ' '
        else:
            prev_pos = sorted_positions[i-1]
            _, prev_end_pos = parse_position(prev_pos)
            middle = list_to_string(sentence[prev_end_pos+1 : curr_start_pos])
            if middle == '':
                middle = ' '
            new_sentence += middle + ' ' + start_replace + ' ' + \
                    list_to_string(sentence[curr_start_pos: curr_end_pos+1]) + ' ' + end_replace + ' '
            if i == len(sorted_positions) - 1 and curr_end_pos < len(sentence) - 1:
                new_sentence += ' ' + list_to_string(sentence[curr_end_pos+1:])
    new_sentence = remove_whitespace(new_sentence)
    # TODO write some code to do the replacement
    return new_sentence

# preprocessing 2: remove the stop words and punctuation from the data
# and replace all digits
# TODO: might be nice to give an option to specify whether to remove the stop words or not 
# this is a low priority part though
def replace_digit_punctuation_stop_word(row, stop_word_removal=True):
    sentence = row.tokenized_sentence.split()
    e1_indexes = row.metadata['e1']['word_index']
    e2_indexes = row.metadata['e2']['word_index']
    sentence = get_new_sentence_with_entity_replacement(sentence, e1_indexes, e2_indexes)
    
    # detection of stop words, punctuations and digits
    index_to_keep_dict = {} # index: {keep that token or not, replace_with}
    tokenizedSentence = sentence.lower().split()
    doc = Doc(nlp.vocab, words=tokenizedSentence)
    nlp.tagger(doc)
    nlp.parser(doc)
    for token in doc:
        word_index = token.i
        stop_word = token.is_stop
        punct = token.is_punct
        num = token.like_num
        
        if (stop_word_removal and (stop_word or punct)) or (not stop_word_removal and punct):
            index_to_keep_dict[word_index] = {'keep': False, 'replace_with': None}
        elif num:
            index_to_keep_dict[word_index] = {'keep': True, 'replace_with': 'NUMBER'}
        else:
            index_to_keep_dict[word_index] = {'keep': True, 'replace_with': None}
    
    # generation of the new sentence based on the above findings
    sentence = sentence.split()
    new_sentence = []
    for i in range(len(sentence)):
        word = sentence[i]
        if word.endswith('END') or word.endswith('START'):
            new_sentence.append(word)
            continue
        if not index_to_keep_dict[i]['keep']:
            continue # don't append when it is a stop word or punctuation
        if index_to_keep_dict[i]['replace_with'] is not None:
            new_sentence.append(index_to_keep_dict[i]['replace_with'])
            continue
        new_sentence.append(word)
    return list_to_string(new_sentence)


'''
Below methods do entity detection from the tagged sentences, i.e. a sentence that contains 
ESTART, EEND etc, use that to detect the locations of the respective entities and remove the tags
from the sentence to return something clean
'''
# below is taken directly from the ddi converter and 
# removes the first occurence of the start and end, and tells of their location
def get_entity_start_and_end(entity_start, entity_end, tokens):
    e_start = tokens.index(entity_start)
    e_end = tokens.index(entity_end) - 2 # 2 tags will be eliminated

    # only eliminate the first occurence of the entity_start and entity_end
    new_tokens = []
    entity_start_seen = 0
    entity_end_seen = 0
    for x in tokens:
        if x == entity_start:
            entity_start_seen += 1
        if x == entity_end:
            entity_end_seen += 1
        if x == entity_start and entity_start_seen == 1:
            continue
        if x == entity_end and entity_end_seen == 1:
            continue
        new_tokens.append(x)
    return (e_start, e_end), new_tokens


# based upon the method in converter for DDI, this will do removal of the entity tags and keep 
# track of where they are located in the sentence
def get_entity_positions_and_replacement_sentence(tokens):
    e1_idx = []
    e2_idx = []
    
    tokens_for_indexing = tokens
    for token in tokens:
        if token.endswith('START'):
            ending_token = token[:-5] + 'END'
            e_idx, tokens_for_indexing = get_entity_start_and_end(token, ending_token, tokens_for_indexing)

            if token == entity1 + 'START' or token == entity_either + 'START':
                e1_idx.append(e_idx)
            if token == entity2 + 'START' or token == entity_either + 'START':
                e2_idx.append(e_idx)
    return e1_idx, e2_idx, tokens_for_indexing


#TODO unify the preprocessing code with actually writing to a dataframe so that experiments can be started
# Read the original dataframe, generate the replacement sentence and then from that, you should just 
# call the get_entity_positions_and_replacement_sentence 
# might be good to just have one method to do this because it seems like the tasks are kinda similar
# just different methods to call for preprocessing 1 vs 2
'''
Returns the dataframe after doing the preprocessing 
'''

# update the metadata and the sentence with the preprocessed version
def update_metadata_sentence(row):
    tagged_sentence = row.tagged_sentence
    e1_idx, e2_idx, tokens_for_indexing = get_entity_positions_and_replacement_sentence(tagged_sentence.split())
    
    new_sentence = list_to_string(tokens_for_indexing)
    metadata = row.metadata
    metadata['e1']['word_index'] = e1_idx
    metadata['e2']['word_index'] = e2_idx
    metadata.pop('entity_replacement', None) # remove the entity replacement dictionary from metadata
    row.tokenized_sentence = new_sentence
    row.metadata = metadata
    return row

# give this preprocessing function a method to read the dataframe, and the location of the original 
# dataframe to read so that it can do the preprocessing
# whether to do type 1 vs type 2 of the preprocessing
# 1: replace with all concepts in the sentence, 2: replace the stop words, punctuations and digits
# 3: replace only punctuations and digits
def preprocess(read_dataframe, df_directory, type_to_do=1):
    df = read_dataframe(df_directory)
    if type_to_do == 1:
        df['tagged_sentence'] = df.apply(replace_with_concept, axis=1) # along the column axis
    elif type_to_do == 2:
        df['tagged_sentence'] = df.apply(replace_digit_punctuation_stop_word, args=(True,), axis=1)
    elif type_to_do == 3:
        df['tagged_sentence'] = df.apply(replace_digit_punctuation_stop_word, args=(False,), axis=1)
    df = df.apply(update_metadata_sentence, axis=1)
    #df = df.rename({'tokenized_sentence': 'preprocessed_sentence'}, axis=1)
    df = df.drop(['tagged_sentence'], axis=1)
    return df
