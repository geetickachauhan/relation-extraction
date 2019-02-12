import os, pandas as pd, numpy as np
import sys
sys.path.append('../../../')
from relation_extraction.data import utils
import nltk
from ast import literal_eval
import itertools
RESOURCE_PATH = "/data/medg/misc/geeticka/relation_extraction/ddi"
outdir = 'pre-processed/original/'
def res(path): return os.path.join(RESOURCE_PATH, path)
from relation_extraction.data.converters.converter_i2b2 import write_dataframe, read_dataframe,\
check_equality_of_written_and_read_df, write_into_txt, combine

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
                                                          'E1START', 'E1END')
    new_entity_replacement_dict = update_dict_with_indexes(new_entity_replacement_dict, only_e2_indexes,
                                                           'E2START', 'E2END')
    new_entity_replacement_dict = update_dict_with_indexes(new_entity_replacement_dict, common_indexes,
                                                           'EITHERSTART', 'EITHEREND')
    return new_entity_replacement_dict

'''
Helper functions
'''
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
    length_sorted_positions = len(sorted_positions)
    for i in range(length_sorted_positions):
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

#TODO write a method to do entity detection from when you write E1START, EITHERSTART, E2START and their
# corresponding ends

#TODO write code for preprocessing 2 i.e. removal of punctuations, first can just do E1START, EITHERSTART, 
# E2START insertions

#TODO unify the preprocessing code with actually writing to a dataframe so that experiments can be started

