"""
    Author: Geeticka Chauhan
    Might be useful to also look at the code by https://github.com/yuanluo/seg_cnn
    This file accompanies the notebook:
    notebooks/Data-Preprocessing/i2b2-preprocessing/i2b2-processing-original.ipynb
    Use the above notebook to create the original dataframe first, because the other 
    Notebooks in the i2b2-processing folder rely on the csv file generated in this notebook
"""
import glob
import os
import re
import pandas as pd
from tqdm import tqdm
from ast import literal_eval

relation_dict = {0: 'TrIP', 1: 'TrWP', 2: 'TrCP', 3: 'TrAP', 4: 'TrNAP', 5: 'TeRP', 6: 'TeCP', 7: 'PIP'}
rev_relation_dict = {val: key for key, val in relation_dict.items()}

# given a file path, just get the name of the file
def get_filename_with_extension(path):
    return os.path.basename(path)

# given the file name with an extension like filename.con, return the filename 
# without the extension i.e. filename
def get_filename_without_extension(path):
    filename_with_extension = os.path.basename(path)
    return os.path.splitext(filename_with_extension)[0]

# given a string that looks like c="concept" extract the concept
def extract_concept_from_string(fullstring):
    return re.match(r'^c=\"(?P<concept>.*)\"$', fullstring).group('concept')

# given a string that looks like t="type" extract the type
def extract_concept_type_from_string(fullstring):
    return re.match(r'^t=\"(?P<type>.*)\"$', fullstring).group('type')

# given a string that looks like r="TrAP" extract the relation
def extract_relation_from_string(fullstring):
    return re.match(r'^r=\"(?P<relation>.*)\"$', fullstring).group('relation')

# given a concept that looks like c="his home regimen" 111:8 111:10, return the components
def get_concept_subparts(concept):
    concept_name = " ".join(concept.split(' ')[:-2])
    concept_name = extract_concept_from_string(concept_name)

    concept_pos1 = concept.split(' ')[-2]
    concept_pos2 = concept.split(' ')[-1]
    return concept_name, concept_pos1, concept_pos2

# given a position like 111:8 return the line number and word number
def get_line_number_and_word_number(position):
    split = position.split(':')
    return split[0], split[1]

# given a specific concept file path, generate a concept dictionary
def get_concept_dictionary(file_path):
    concept_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            concept = line.split('||')[0] # line splitting
            type_of_concept = line.split('||')[1]
            
            type_of_concept = extract_concept_type_from_string(type_of_concept) # getting useful info
            concept_name, concept_pos1, concept_pos2 = get_concept_subparts(concept)

            line1, _ = get_line_number_and_word_number(concept_pos1)
            line2, _ = get_line_number_and_word_number(concept_pos2)
            if line1 != line2:
                print("There is a problem! Concept spans multiple lines")
            
            from_to_positions = concept_pos1 + ";" + concept_pos2
            concept_dict[from_to_positions] = {
                    'fromto': from_to_positions, 'word': concept_name, 'type': type_of_concept}
    return concept_dict

# given a line number and the concept dictionary, return all the concepts from the 
# particular line #
def get_entity_replacement_dictionary(linenum, concept_dict):
    entity_replacement = {}
    for key, val in concept_dict.items():
        dict_linenum = key.split(';')[0].split(':')[0]
        if dict_linenum == linenum:
            fromword = key.split(';')[0].split(':')[1]
            toword = key.split(';')[1].split(':')[1]
            ent_repl_key = str(fromword) + ':' + str(toword)
            entity_replacement[ent_repl_key] = val['type']
    return entity_replacement # returns a list of dictionaries i.e. from-to, word, type

# given a line in the relation file, return the concept1 word, spans, relation and concept 2 word, spans
def read_rel_line(rel_line):
    line = rel_line.strip()
    concept1 = line.split('||')[0]
    relation = line.split('||')[1]
    concept2 = line.split('||')[2]

    concept1_name, concept1_pos1, concept1_pos2 = get_concept_subparts(concept1)
    concept2_name, concept2_pos1, concept2_pos2 = get_concept_subparts(concept2)
    relation = extract_relation_from_string(relation)

    line1_concept1, from_word_concept1 = get_line_number_and_word_number(concept1_pos1)
    line2_concept1, to_word_concept1  = get_line_number_and_word_number(concept1_pos2)

    line1_concept2, from_word_concept2 = get_line_number_and_word_number(concept2_pos1)
    line2_concept2, to_word_concept2 = get_line_number_and_word_number(concept2_pos2)

    if line1_concept1 != line2_concept1 or line1_concept2 != line2_concept2 or \
            line1_concept1 != line1_concept2:
                print("Concepts are in two different lines")
    # assuming that all the lines are the same
    return {'e1_word': concept1_name, 'e1_from': from_word_concept1, 'e1_to': to_word_concept1,
            'e2_word': concept2_name, 'e2_from': from_word_concept2, 'e2_to': to_word_concept2, 
            'line_num': line1_concept1, 'relation': relation}

# give it a directory with res(directory + 'concept/')
def get_dataset_dataframe(concept_directory, rel_directory, txt_directory):
    data = []
    total_rel_files_to_read = glob.glob(os.path.join(rel_directory, '*'))
    
    for rel_file_path in tqdm(total_rel_files_to_read):
        with open(rel_file_path, 'r') as rel_file:
            base_filename = get_filename_without_extension(rel_file_path)
            concept_file_path = os.path.join(concept_directory, base_filename +".con")
            concept_dictionary = get_concept_dictionary(concept_file_path)
            
            text_file_path = os.path.join(txt_directory, base_filename +".txt")
            text_file = open(text_file_path, 'r').readlines() 

            for rel_line in rel_file:
                rel_dict = read_rel_line(rel_line)
                tokenized_sentence = text_file[int(rel_dict['line_num']) - 1].strip()
                sentence_text = tokenized_sentence
                e1 = rel_dict['e1_word']
                e2 = rel_dict['e2_word']
                relation_type = rel_dict['relation']
                linenum = rel_dict['line_num']
                entity_replacement_dict = get_entity_replacement_dictionary(linenum, concept_dictionary)
                
                e1_idx = [(rel_dict['e1_from'], rel_dict['e1_to'])]
                e2_idx = [(rel_dict['e2_from'], rel_dict['e2_to'])]

                metadata = {'e1': {'word': str(e1), 'word_index': e1_idx},
                            'e2': {'word': str(e2), 'word_index': e2_idx},
                            'entity_replacement': entity_replacement_dict,
                            'sentence_id': str(linenum), # numbering starts from 1
                            'filename': str(base_filename)}
                data.append([str(sentence_text), str(e1), str(e2), str(relation_type), metadata,
                    str(tokenized_sentence)])

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
# TODO: need to change below in order to contain a sorted list of the positions
# metadata['e1']['word_index'] returns a list of tuples
            e1 = metadata['e1']['word_index'][0]
            e2 = metadata['e2']['word_index'][0]
            tokenized_sentence = row.tokenized_sentence
            outfile.write(str(relation) + " " + str(e1[0]) + " " + str(e1[-1]) + " " + 
                          str(e2[0]) + " " + str(e2[-1]) + " " + tokenized_sentence + "\n")
        outfile.close()


# combine txt files of beth and partners
def combine(res, outdir, file1, file2, outfilename):
    outfile = outdir + outfilename
    # https://stackoverflow.com/questions/13613336/python-concatenate-text-files
    filenames = [res(outdir + file1+'.txt'), res(outdir + file2+'.txt')]
    with open(res(outfile), 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
