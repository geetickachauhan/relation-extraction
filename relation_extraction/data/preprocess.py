'''
Author: Geeticka Chauhan
Performs pre-processing on a csv file independent of the dataset (once converters have been applied). 
Refer to notebooks/Data-Preprocessing for more details. The methods are specifically used in the non
_original notebooks for all datasets.
'''

import os, pandas as pd, numpy as np
import nltk
import spacy
from spacy.tokens import Doc

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
    nlp = spacy.load('en_core_web_lg')
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
Preprocessing Type 3 part 1: NER
'''

# a method to check for overlap between the ner_dict that is created
def check_for_overlap(ner_dict):
    def expand_key(string): # a string that looks like '2:2' to [2]
        start = int(string.split(':')[0])
        end = int(string.split(':')[1])
        return list(range(start, end+1))
    expanded_keys = [expand_key(key) for key in ner_dict.keys()]
    for i1, item in enumerate(expanded_keys):
        for i2 in range(i1 + 1, len(expanded_keys)):
            if set(item).intersection(expanded_keys[i2]):
                return True # overlap is true
        for i2 in range(0, i1):
            if set(item).intersection(expanded_keys[i2]):
                return True
    return False


###
### Helper functions for the NER replacement
###
def overlap_index(index1, index2):
    def expand(index):
        start = int(index[0])
        end = int(index[1])
        return list(range(start, end+1))
    expand_index1 = expand(index1)
    expand_index2 = expand(index2)
    if set(expand_index1).intersection(set(expand_index2)):
        return True
    else: return False
    
# for indexes that look like (1,1) and (2,2) check if the left is fully included in the right
def fully_included(index1, index2):
    if int(index1[0]) >= int(index2[0]) and int(index1[1]) <= int(index2[1]): return True
    else: return False

def beginning_overlap(index1, index2): # this is tricky when (1,1) and (2,2) are there
    if int(index1[0]) < int(index2[0]) and int(index1[1]) <= int(index2[1]): return True
    else: return False

def end_overlap(index1, index2): # this is tricky
    if int(index1[0]) >= int(index2[0]) and int(index1[1]) > int(index2[1]): return True
    else: return False
    
def beginning_and_end_overlap(index1, index2):
    if int(index1[0]) < int(index2[0]) and int(index1[1]) > int(index2[1]): return True
    else:
        return False
#else there is no overlap

# taken from https://stackoverflow.com/questions/46548902/converting-elements-of-list-of-nested-lists-from-string-to-integer-in-python
def list_to_int(lists):
  return [int(el) if not isinstance(el,list) else convert_to_int(el) for el in lists]

def correct_entity_indexes_with_ner(ner_dict, e_index):
    new_e_index = []
    for i in range(len(e_index)): # we are reading tuples here
        for key in ner_dict.keys():
            indexes = e_index[i]
            index2 = indexes
            index1 = parse_position(key) # checking if ner is fully included etc
            if not overlap_index(index1, index2): # don't do below if there is no overlap
                continue
            if beginning_overlap(index1, index2):
                e_index[i] = (index1[0], e_index[i][1])
            elif end_overlap(index1, index2):
                e_index[i] = (e_index[i][0], index1[1])
            elif beginning_and_end_overlap(index1, index2):
                e_index[i] = (index1[0], index1[1]) # else you don't change or do anything
    return e_index
            

# given all of these dictionaries, return the ner replacement dictionary
def get_ner_replacement_dictionary(only_e1_index, only_e2_index, common_indexes, ner_dict):
    def update_dict_with_entity(e_index, ner_repl_dict, entity_name):
        for indexes in e_index:
            key1 = str(indexes[0]) + ':' + str(indexes[0]) + ':' + entity_name + 'START'
            ner_repl_dict[key1] = {'replace_by': None, 'insert': entity_name + 'START'}
            key2 = str(int(indexes[-1]) + 1) + ':' + str(int(indexes[-1]) + 1) + ':' + entity_name + 'END'
            ner_repl_dict[key2] = {'replace_by': None, 'insert': entity_name + 'END'}
        return ner_repl_dict
    # we are going to do something different: only spans for NER will be counted, but
    # for the ENTITYSTART and ENTITYEND, we will keep the span as what token to insert before
    ner_repl_dict = {}
    for key in ner_dict:
        ner_repl_dict[key] = {'replace_by': ner_dict[key], 'insert': None}
    ner_repl_dict = update_dict_with_entity(only_e1_index, ner_repl_dict, entity1)
    ner_repl_dict = update_dict_with_entity(only_e2_index, ner_repl_dict, entity2)
    ner_repl_dict = update_dict_with_entity(common_indexes, ner_repl_dict, entity_either)
    return ner_repl_dict

# this function is different from the sort_position_keys because
# we care about sorting not just by the beginning token, but also by the length that the span contains
def ner_sort_position_keys(ner_repl_dict): # this can potentially replace sort_position_keys
    # but only if the application of this function does not change the preprocessed CSVs generated
    def len_key(key):
        pos = parse_position(key)
        return pos[1] - pos[0] + 1
    def start_or_end(key): 
        # handle the case where the ending tag of the entity is in the same place as the
        #starting tag of another entity - this happens when two entities are next to each other
        if len(key.split(':')) <= 2: # means that this is a named entity
            return 3
        start_or_end = key.split(':')[2]
        if start_or_end.endswith('END'): # ending spans should get priority
            return 1
        elif start_or_end.endswith('START'):
            return 2
    positions = list(ner_repl_dict.keys())
    sorted_positions = sorted(positions, key=lambda x: (parse_position(x)[0], len_key(x), start_or_end(x)))
    return sorted_positions

# given a splitted sentence - make sure that the sentence is in list form
def get_ner_dict(sentence, nlp):
    #nlp = spacy.load(spacy_model_name)
    tokenizedSentence = sentence # in this case lowercasing is not helpful
    doc = Doc(nlp.vocab, words=tokenizedSentence)
    nlp.tagger(doc)
    nlp.parser(doc)
    nlp.entity(doc) # run NER
    ner_dict = {} # first test for overlaps within ner
    for ent in doc.ents:
        key = str(ent.start) + ':' + str(ent.end - 1)
        ner_dict[key] = ent.label_
    return ner_dict

def convert_indexes_to_int(e_idx):
    new_e_idx = []
    for indexes in e_idx:
        t = (int(indexes[0]), int(indexes[1]))
        new_e_idx.append(t)
    return new_e_idx

def replace_ner(row, nlp, check_ner_overlap=False): # similar to concept_replace, with some caveats
    sentence = row.tokenized_sentence.split()
    e1_indexes = row.metadata['e1']['word_index']
    e2_indexes = row.metadata['e2']['word_index']
    e1_indexes = convert_indexes_to_int(e1_indexes)
    e2_indexes = convert_indexes_to_int(e2_indexes)
    only_e1_indexes, only_e2_indexes, common_indexes = \
    get_common_and_separate_entities(e1_indexes, e2_indexes)
    ner_dict = get_ner_dict(sentence, nlp)
    if check_ner_overlap and check_for_overlap(ner_dict):
        print("There is overlap", ner_dict) # only need to check this once
    #Below code works only if there isn't overlap within ner_dict, so make sure that there isn't overlap
    
    # overlaps between ner label and e1 and e2 indexes are a problem
    # And they can be of two types
        # Type 1: NER overlaps with e1 or e2 in the beginning or end
        # Here we want to keep the NER link the same but extend e1 or e2 index to the beginning or end of the
        # NER 

        #Type 2: NER is inside of the entity completely: At this point it should be simply ok to mention at what 
        # token to insert ENTITYstart and ENTITYend
        # Type 1 is a problem, but Type 2 is easy to handle while the new sentence is being created
        
    only_e1_indexes = correct_entity_indexes_with_ner(ner_dict, only_e1_indexes)
    only_e2_indexes = correct_entity_indexes_with_ner(ner_dict, only_e2_indexes)
    common_indexes = correct_entity_indexes_with_ner(ner_dict, common_indexes)

    # below needs to be done in case there was again a shift that might have caused both e1 and e2 to have
    # the same spans
    only_e1_indexes, only_e2_indexes, common_indexes2 = \
            get_common_and_separate_entities(only_e1_indexes, only_e2_indexes)
    common_indexes.extend(common_indexes2)

    ner_repl_dict = get_ner_replacement_dictionary(only_e1_indexes, only_e2_indexes, common_indexes,
                                                  ner_dict)
    sorted_positions = ner_sort_position_keys(ner_repl_dict)
    new_sentence = '' # this below part is buggy, shouldn't be too bad to fix 
    for i in range(len(sorted_positions)):
        curr_pos = sorted_positions[i]
        curr_start_pos, curr_end_pos = parse_position(curr_pos)
        curr_dict = ner_repl_dict[curr_pos]
        start_insert = '' if curr_dict['insert'] is None else curr_dict['insert'].upper()
        between_replace = '' if curr_dict['replace_by'] is None else curr_dict['replace_by']
        if i == 0:
            new_sentence += list_to_string(sentence[:curr_start_pos]) + ' ' + start_insert + ' ' + \
            between_replace + ' '
        else:
            prev_pos = sorted_positions[i-1]
            _, prev_end_pos = parse_position(prev_pos)
            if ner_repl_dict[prev_pos]['insert'] is None: # means middle will be starting from prev_pos + 1
                middle = list_to_string(sentence[prev_end_pos+1 : curr_start_pos])
            else: # means middle needs to start from the prev_pos
                middle = list_to_string(sentence[prev_end_pos: curr_start_pos])
            if middle == '':
                middle = ' '
            new_sentence += middle + ' ' + start_insert + ' ' + between_replace + ' '
            if i == len(sorted_positions) - 1 and curr_end_pos < len(sentence) - 1:
                position = curr_end_pos + 1 if curr_dict['insert'] is None else curr_end_pos
                new_sentence += ' ' + list_to_string(sentence[position:])
    new_sentence = remove_whitespace(new_sentence)
    return new_sentence

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

    between_tags = 0
    for index in range(e_start + 1, e_end + 2): 
        # we want to check between the start and end for occurence of other tags
        if tokens[index].endswith('START') or tokens[index].endswith('END'):
            between_tags += 1
    e_end -= between_tags

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
def preprocess(read_dataframe, df_directory, nlp, type_to_do=1):
    df = read_dataframe(df_directory)
    if type_to_do == 1:
        df['tagged_sentence'] = df.apply(replace_with_concept, axis=1) # along the column axis
    elif type_to_do == 2:
        df['tagged_sentence'] = df.apply(replace_digit_punctuation_stop_word, args=(True,), axis=1)
    elif type_to_do == 3:
        df['tagged_sentence'] = df.apply(replace_digit_punctuation_stop_word, args=(False,), axis=1)
    elif type_to_do == 4:
        df['tagged_sentence'] = df.apply(replace_ner, args=(nlp, False), axis=1)
    df = df.apply(update_metadata_sentence, axis=1)
    #df = df.rename({'tokenized_sentence': 'preprocessed_sentence'}, axis=1)
    df = df.drop(['tagged_sentence'], axis=1)
    return df
