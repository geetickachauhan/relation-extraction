"""
    Author: Geeticka Chauhan
    Based upon code here https://github.com/pratapbhanu/CRCNN/blob/master/dataio.py and 
    https://github.com/roomylee/cnn-relation-extraction/blob/master/data_helpers.py

    This accompanies the notebook: notebook/Semeval_preprocess/Semeval_preprocess_original
"""
import spacy
from ast import literal_eval
import pandas as pd
parser = spacy.load('en')


relation_dict = {0:'Component-Whole(e2,e1)', 1:'Instrument-Agency(e2,e1)', 2:'Member-Collection(e1,e2)',
3:'Cause-Effect(e2,e1)', 4:'Entity-Destination(e1,e2)', 5:'Content-Container(e1,e2)',
6:'Message-Topic(e1,e2)', 7:'Product-Producer(e2,e1)', 8:'Member-Collection(e2,e1)',
9:'Entity-Origin(e1,e2)', 10:'Cause-Effect(e1,e2)', 11:'Component-Whole(e1,e2)',
12:'Message-Topic(e2,e1)', 13:'Product-Producer(e1,e2)', 14:'Entity-Origin(e2,e1)',
15:'Content-Container(e2,e1)', 16:'Instrument-Agency(e1,e2)', 17:'Entity-Destination(e2,e1)',
18:'Other'}
rev_relation_dict = {val: key for key, val in relation_dict.items()}

def tokenize(sentence):
    doc = parser(sentence)
    tokenized = []
    for token in doc:
        tokenized.append(token.text)
    return tokenized

# get the start and end of the entities 
def get_entity_start_and_end(entity_start, entity_end, tokens):
    e_start = tokens.index(entity_start)
    e_end = tokens.index(entity_end) - 2 # because 2 tags will be eliminated
    tokens = [x for x in tokens if x != entity_start and x != entity_end]
    return [(e_start, e_end)], tokens

# given the entity starting and ending word index, and entity replacement dictionary, 
# update the dictionary to inform of the replace_by string for eg ENTITY
def get_entity_replacement_dictionary(e_idx, entity_replacement, replace_by):
    key = str(e_idx[0][0]) + ":" + str(e_idx[0][1])
    entity_replacement[key] = replace_by
    return entity_replacement

# remove any additional whitespace within a line
def remove_whitespace(line):
    return str(" ".join(line.split()).strip())

# given a sentence that contains the ENITTYSTART, ENTITYEND etc, replace them by ""
def get_original_sentence(sent):
    entity_tags = ['ENTITYSTART', 'ENTITYEND', 'ENTITYOTHERSTART', 'ENTITYOTHEREND']
    original_sentence = sent
    for tag in entity_tags:
        original_sentence = original_sentence.replace(tag, "")
    return remove_whitespace(original_sentence)

# provide the directory where the file is located, along with the name of the file
def get_dataset_dataframe(directory):
    data = []
    with open(directory, 'r') as file:
        for line in file:
            id, sent = line.split('\t')
            rel = next(file).strip()
            next(file) # comment
            next(file) # blankline

            sent = sent.strip()
            if sent[0] == '"':
                sent = sent[1:]
            if sent[-1] == '"':
                sent = sent[:-1]
            sent = sent.replace('<e1>', ' ENTITYSTART ')
            sent = sent.replace('</e1>', ' ENTITYEND ')
            sent = sent.replace('<e2>', ' ENTITYOTHERSTART ')
            sent = sent.replace('</e2>', ' ENTITYOTHEREND ')
            sent = remove_whitespace(sent) # to get rid of additional white space

            tokens = tokenize(sent)
            start_with_e1 = True
            for token in tokens:
                if token == 'ENTITYSTART':
                    break
                if token == 'ENTITYOTHERSTART':
                    start_with_e1 = False
                    print("In sentence with ID %d sentence starts with e2"%id)
                    break
            
            if start_with_e1:
                e1_idx, tokens = get_entity_start_and_end('ENTITYSTART', 'ENTITYEND', tokens)
                e2_idx, tokens = get_entity_start_and_end('ENTITYOTHERSTART', 'ENTITYOTHEREND', tokens)
            else:
                e2_idx, tokens = get_entity_start_and_end('ENTITYOTHERSTART', 'ENTITYOTHEREND', tokens)
                e1_idx, tokens = get_entity_start_and_end('ENTITYSTART', 'ENTITYEND', tokens)

            e1 = str(" ".join(tokens[e1_idx[0][0] : e1_idx[0][1] + 1]).strip())
            e2 = str(" ".join(tokens[e2_idx[0][0] : e2_idx[0][1] + 1]).strip())
            
            entity_replacement = {}
            entity_replacement = get_entity_replacement_dictionary(e1_idx, entity_replacement, 'ENTITY')
            entity_replacement = get_entity_replacement_dictionary(e2_idx, entity_replacement, 'ENTITYOTHER')

            metadata = {'e1': {'word': e1, 'word_index': e1_idx}, # to indicate that this is word level idx
                        'e2': {'word': e2, 'word_index': e2_idx}, 
                        'entity_replacement': entity_replacement,
                        'sentence_id': id}

            tokenized_sent = " ".join(tokens)
            original_sentence = get_original_sentence(sent) # just to write into the dataframe, sent is manipulated
            data.append([original_sentence, e1, e2, rel, metadata, tokenized_sent])

        df = pd.DataFrame(data,
                columns='original_sentence,e1,e2,relation_type,metadata,tokenized_sentence'.split(','))
        return df

# to streamline the writing of the dataframe
def write_dataframe(df, directory):
    df.to_csv(directory, sep='\t', encoding='utf-8', index=False)

# to streamline the reading of the dataframe
def read_dataframe(directory):
    df = pd.read_csv(directory, sep='\t')
    def literal_eval_metadata(row):
        metadata = row.metadata
        metadata = literal_eval(metadata)
        return metadata
    df['metadata'] = df.apply(literal_eval_metadata, axis=1)
    # metadata is a dictionary which is written into the csv format as a string
    # but in order to be treated as a dictionary it needs to be evaluated
    return df

# The goal here is to make sure that the df that is written into memory is the same one that is read
def check_equality_of_written_and_read_df(df, df_copy):
    bool_equality = df.equals(df_copy)
    # to double check, we want to check with every column
    bool_every_column = True
    for idx in range(len(df)):
        row1 = df.iloc[idx]
        row2 = df_copy.iloc[idx]
        if row1['original_sentence'] != row2['original_sentence'] or row1['e1'] != row2['e1'] or \
                row1['relation_type'] != row2['relation_type'] or \
                row1['tokenized_sentence'] != row2['tokenized_sentence'] or \
                row1['metadata'] != row2['metadata']: 
                    bool_every_column = False
                    break
    return bool_equality, bool_every_column

# write the dataframe into the text format accepted by the cnn model
def write_into_txt(df, directory):
    print("Unique relations: \t", df['relation_type'].unique())
    null_row = df[df["relation_type"].isnull()]
    if null_row.empty:
        idx_null_row = None
    else:
        idx_null_row = null_row.index.values[0]
    with open(directory, 'w') as outfile:
        for i in range(0, len(df)):
            if idx_null_row is not None and i == idx_null_row:
                continue
            row = df.iloc[i]
            relation = rev_relation_dict[row.relation_type]
            metadata = row.metadata
            e1 = metadata['e1']['word_index'][0]
            e2 = metadata['e2']['word_index'][0]
            tokenized_sentence = row.tokenized_sentence
            outfile.write(str(relation) + " " + str(e1[0]) + " " + str(e1[-1]) + " " + 
                          str(e2[0]) + " " + str(e2[-1]) + " " + tokenized_sentence + "\n")
        outfile.close()
