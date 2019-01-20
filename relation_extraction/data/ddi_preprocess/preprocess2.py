# -*- coding: utf-8 -*-
import glob
import os
from pyexpat import ExpatError
from xml.dom import minidom
import re

import pandas as pd
from nltk.corpus import stopwords
from tqdm import tqdm


import spacy
parser = spacy.load('en')
# pd.set_option('display.width', 1000)
dataset_csv_file = 'dataset_dataframe.csv'

training_dataset_dataframe = None


def get_entity_dict(sentence_dom):
    entities = sentence_dom.getElementsByTagName('entity')
    entity_dict = {}
    for entity in entities:
        id = entity.getAttribute('id')
        word = entity.getAttribute('text')
        type = entity.getAttribute('type')
        charOffset = entity.getAttribute('charOffset')
        charOffset = charOffset.split(';') # charOffset can either be length 1 or 2 
        # because we have cases like loop, potassium-sparing diuretics
        # where loop diuretics and potassium-sparing diuretics is the entity
        entity_dict[id] = {'id': id, 'word': word, 'charOffset': charOffset, 'type': type}
    return entity_dict


# given the metadata, get the individual positions in the sentence and know what to replace them by
def create_positions_dict(metadata):
    position_dict = {}
    e1 = metadata['e1']
    e2 = metadata['e2']
    other_entities = metadata['other_entities']
    common_position_e1_e2 = list(set(e1['charOffset']).intersection(e2['charOffset']))
    if common_position_e1_e2: # there are commonalities between e1 and e2
        for pos in common_position_e1_e2:
            position_dict[pos] = 'EITHERDRUG'
    for pos in e1['charOffset']:
        if pos not in position_dict:
            position_dict[pos] = 'DRUG'
    for pos in e2['charOffset']:
        if pos not in position_dict:
            position_dict[pos] = 'OTHERDRUG'
    #for other_ent in other_entities:
    #    for pos in other_ent['charOffset']:
    #        if pos not in position_dict:
    #            position_dict[pos] = 'UNRELATEDDRUG'
    return position_dict

#given string 12-30, return 12, 30 as a tuple of ints
def parse_position(position):
    positions = position.split('-')
    return int(positions[0]), int(positions[1])
    #if metadata['e1']['charOffset'] and metadata['e2']['charOffset'] have something in common

# given position dictionary, sort the positions from ascending order. Assumes no overlap. 
# will be messed up if there is overlap
# can also check for overlap but not right now
def sort_position_keys(position_dict):
    positions = list(position_dict.keys())
    sorted_positions = sorted(positions, key=lambda x: int(x.split('-')[0]))
    return sorted_positions

# We will replace e1 by DRUG, e2 by OTHERDRUG, common between e1 and e2 as EITHERDRUG and other drugs as 
# UNRELATEDDRUG
def normalize_sentence(row):
    sentence = row.sentence_text
    e1 = row.e1
    e2 = row.e2
    metadata = row.metadata
    position_dict = create_positions_dict(metadata)
    sorted_positions = sort_position_keys(position_dict)
    new_sentence = ''
# TODO (geeticka): check for cases when previous ending position and next starting position 
# are equal or next to each other. Add a space in that case
    for i in range(len(sorted_positions)):
        curr_pos = sorted_positions[i]
        curr_start_pos, curr_end_pos = parse_position(curr_pos)
        if i == 0:
            new_sentence += sentence[:curr_start_pos] + ' ' + position_dict[curr_pos]
        else:
            prev_pos = sorted_positions[i-1]
            _, prev_end_pos = parse_position(prev_pos)
            middle = sentence[prev_end_pos+1 : curr_start_pos]
            if middle == '':
                middle = ' '
            new_sentence += middle + ' ' + position_dict[curr_pos] 
            if i == len(sorted_positions) - 1 and curr_end_pos < len(sentence) - 1:
                new_sentence += ' ' + sentence[curr_end_pos+1:]
                
    new_sentence = new_sentence.replace('.', ' . ')
    new_sentence = new_sentence.replace(',', ' , ')
    new_sentence = new_sentence.replace('-',' ')
    new_sentence = new_sentence.replace('/',' / ')
    new_sentence = re.sub('\d', "number", new_sentence)
    #new_sentence = re.sub(' +', ' ', new_sentence) # remove unnecessary spaces
    return new_sentence

            

# normalizing removes stop words and replaces the entities with different words whereas in our case
# we do not want to do that. 
def tokenize_sentence(row):
    sentence = row.normalized_sentence
    return tokenize(sentence)

def tokenize(e):
    parsedData = parser(e)
    tokenized = []
    for span in parsedData.sents:
        sent = [parsedData[i] for i in range(span.start, span.end)]
        for token in sent:
            tokenized.append(token.orth_)
    return " ".join(tokenized).strip()

def get_entity_number(row):
    tokenized_sentence = row.tokenized_sentence
    e1 = 'DRUG'
    e2 = 'OTHERDRUG'
    # also need to look for 'EITHERDRUG'
    
    e1_limit = []
    e2_limit = []
    

    idx_e1 = 0
    e1_found = False

    idx_e2 = 0
    e2_found = False

    idx_sent = 0
    for word in tokenized_sentence.split():
        if word.startswith('DRUG'):
            e1_limit.append(idx_sent)
            e1_found = True

        if word.startswith('OTHERDRUG'):
            e2_limit.append(idx_sent)
            e2_found = True

        if word.startswith('EITHERDRUG'):
            e1_limit.append(idx_sent)
            e2_limit.append(idx_sent)
        idx_sent += 1
    return e1_limit, e2_limit

# return the entities in the sentence except those in the pair
def get_other_entities(entity_dict, e1, e2):
    blacklisted_set = [e1, e2]
    return [value for key, value in entity_dict.items() if key not in blacklisted_set]

def get_dataset_dataframe(directory=None, relation_extraction=True):
    '''
    If relation_extraction is True, then we don't care whether the ddi flag is true or false
    '''
    global training_dataset_dataframe, dataset_csv_file

    if training_dataset_dataframe:
        return training_dataset_dataframe

    #if directory is None:
    #    print("You must provide a directory!")
    if directory is None:
        directory = os.path.expanduser('/data/medg/misc/semeval_2010/medical-data/DDICorpus/Train/DrugBank/')

    dataset_csv_file_prefix = str(directory.split('/')[-3]).lower() + '_'

    dataset_csv_file = dataset_csv_file_prefix + dataset_csv_file
    if os.path.isfile(dataset_csv_file):
        df = pd.read_csv(dataset_csv_file)
        return df

    lol = []
    total_files_to_read = glob.glob(directory + '*.xml')
    print('total_files_to_read:' , len(total_files_to_read) , ' from dir: ' , directory)
    for file in tqdm(total_files_to_read):
        try:
            DOMTree = minidom.parse(file)
            sentences = DOMTree.getElementsByTagName('sentence')

            for sentence_dom in sentences:
                entity_dict = get_entity_dict(sentence_dom)

                pairs = sentence_dom.getElementsByTagName('pair')
                sentence_text = sentence_dom.getAttribute('text')
                for pair in pairs:
                    ddi_flag = pair.getAttribute('ddi')
                    e1_id = pair.getAttribute('e1')
                    e2_id = pair.getAttribute('e2')
                    other_entities = get_other_entities(entity_dict, e1_id, e2_id)
                    
                    e1_data = entity_dict[e1_id]
                    e2_data = entity_dict[e2_id]
                    metadata = {'e1': e1_data, 'e2': e2_data,
                                'other_entities': other_entities}

                    #TODO (geeticka) calculate the other entities here
                    # print(pair.attributes().items())
                    if relation_extraction is True and ddi_flag == 'false':
                        relation_type = 'none'
                        lol.append([sentence_text, e1_data['word'], e2_data['word'],
                            relation_type, metadata])
                    if ddi_flag == 'true':
                        relation_type = pair.getAttribute('type')
                        lol.append([sentence_text, e1_data['word'], e2_data['word'],
                            relation_type, metadata])
        except ExpatError:
            pass

    df = pd.DataFrame(lol, columns='sentence_text,e1,e2,relation_type,metadata'.split(','))
    df['normalized_sentence'] = df.apply(normalize_sentence, axis=1) 
    df['tokenized_sentence'] = df.apply(tokenize_sentence, axis=1)
    df['entity_number'] = df.apply(get_entity_number, axis=1)
    #df.to_csv(dataset_csv_file)
    #df = pd.read_csv(dataset_csv_file)
    return df


