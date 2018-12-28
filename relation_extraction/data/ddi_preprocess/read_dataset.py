# -*- coding: utf-8 -*-
import glob
import os
from pyexpat import ExpatError
from xml.dom import minidom

import pandas as pd
from nltk.corpus import stopwords
from tqdm import tqdm

STOP_WORDS = set(stopwords.words('english')) | set('the')

import spacy
parser = spacy.load('en')
# pd.set_option('display.width', 1000)
dataset_csv_file = 'dataset_dataframe.csv'
types = set()

training_dataset_dataframe = None


def get_entity_dict(sentence_dom):
    entities = sentence_dom.getElementsByTagName('entity')
    entity_dict = {}
    for entity in entities:
        id = entity.getAttribute('id')
        word = entity.getAttribute('text')
        entity_dict[id] = word
    return entity_dict


def normalize_sentence(row):
    sentence = row.sentence_text.replace('.', ' . ')
    sentence = sentence.replace(',', ' , ')
    e1 = row.e1
    e2 = row.e2
    new_sentence_tokenized = []
    i = 0
    for word in sentence.split():
        if word in STOP_WORDS:
            continue
        if word.lower() == e1.lower():
            new_sentence_tokenized.append('DRUG')
            i += 1
        elif word.lower() == e2.lower():
            new_sentence_tokenized.append('OTHER_DRUG')
            i += 1
        elif i == 0:
            new_sentence_tokenized.append(word + '_bf')
        elif i == 1:
            new_sentence_tokenized.append(word + '_be')
        else:
            new_sentence_tokenized.append(word + '_af')
    normalized_sentence = ' '.join(new_sentence_tokenized).strip()
    # print(e1, e2, ' :  sentence :', sentence, 'new_sentence', normalized_sentence, '\n\n')
    return normalized_sentence

# normalizing removes stop words and replaces the entities with different words whereas in our case
# we do not want to do that. 
def tokenize_sentence(row):
    sentence = row.sentence_text
    return tokenize(sentence)
    #parsedData = parser(sentence)
    #new_sentence_tokenized = []
    #for span in parsedData.sents:
    #    sent = [parsedData[i] for i in range(span.start, span.end)]
    #    for token in sent:
    #        new_sentence_tokenized.append(token.orth_)
    #return " ".join(new_sentence_tokenized).strip()

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
    e1 = row.e1
    e2 = row.e2
    
    e1_limit = []
    e2_limit = []
    
    e1 = tokenize(e1)
    e2 = tokenize(e2)

    e1 = e1.split()
    idx_e1 = 0
    e1_found = False

    e2 = e2.split()
    idx_e2 = 0
    e2_found = False

    idx_sent = 0
    for word in tokenized_sentence.split():
        if idx_e1 < len(e1) and e1[idx_e1].lower() == word.lower():
            e1_limit.append(idx_sent)
            idx_e1 += 1
            e1_found = True
        elif idx_e1 < len(e1) and e1_found is True and e1[idx_e1].lower() != word.lower():
            idx_e1 = 0
            e1_found = False
            e1_limit = []

        # need to handle the case where the first letter of the e1 matched, but all the letters did not match
        if idx_e2 < len(e2) and e2[idx_e2].lower() == word.lower():
            e2_limit.append(idx_sent)
            idx_e2 += 1
            e2_found = True
        elif idx_e2 < len(e2) and e2_found is True and e2[idx_e2].lower() != word.lower():
            idx_e2 = 0
            e2_found = False
            e2_limit = []
        idx_sent += 1
    return e1_limit, e2_limit

def get_dataset_dataframe(directory=None, relation_extraction=False):
    '''
    If relation_extraction is True, then we don't care whether the ddi flag is true or false
    '''
    global training_dataset_dataframe, dataset_csv_file

    if training_dataset_dataframe:
        return training_dataset_dataframe
    global types

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
                    # print(pair.attributes().items())
                    if not os.path.isfile('types'):
                        types.add(pair.getAttribute('type'))
                    if relation_extraction is True and ddi_flag == 'false':
                        e1 = pair.getAttribute('e1')
                        e2 = pair.getAttribute('e2')
                        relation_type = 'none'
                        lol.append([sentence_text, entity_dict[e1], entity_dict[e2], relation_type])
                    if ddi_flag == 'true':
                        e1 = pair.getAttribute('e1')
                        e2 = pair.getAttribute('e2')
                        relation_type = pair.getAttribute('type')
                        lol.append([sentence_text, entity_dict[e1], entity_dict[e2], relation_type])
        except ExpatError:
            pass

    #pd.to_pickle(types, 'types')
    df = pd.DataFrame(lol, columns='sentence_text,e1,e2,relation_type'.split(','))
    df['normalized_sentence'] = df.apply(normalize_sentence, axis=1)
    df['tokenized_sentence'] = df.apply(tokenize_sentence, axis=1)
    df['entity_number'] = df.apply(get_entity_number, axis=1)
    #df.to_csv(dataset_csv_file)
    #df = pd.read_csv(dataset_csv_file)
    return df


def get_training_label(row):
    global types

    types = pd.read_pickle('types')
    types = [t for t in types if t]
    type_list = list(types)
    relation_type = row.relation_type
    X = [i for i, t in enumerate(type_list) if relation_type == t]
    # s = np.sum(X)
    if X:
        return X[0]
    else:
        return 1
