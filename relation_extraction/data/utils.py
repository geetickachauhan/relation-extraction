import numpy as np
import logging
from collections import Counter
import pickle
import re
import networkx as nx
import spacy
from spacy.tokens import Doc
import nltk
nltk.download('wordnet')
from nltk import wordnet as wn
import random
import h5py # conda install -c conda-forge h5py
from spacy.lang.en.stop_words import STOP_WORDS as stop_words

#TODO (geeticka) need to clean up utils based upon the methods that are
# not directly used by the script anymore
# to get the dataset from the cross validation splits
TRAIN, DEV, TEST = 0, 1, 2
class Dataset():
    def __init__(self, relations_split_file):
        with open(relations_split_file, mode='rb') as f: self.relations_splits = pickle.load(f)
        self.K = len(self.relations_splits)

    def get_data_for_fold(self, fold_num, data_type=TRAIN, mode='normal'): # mode can also be elmo
        assert fold_num < self.K
        data = self.relations_splits[fold_num][data_type]
        if mode == 'elmo':
            return data['sentences'].tolist(), data['relations'].tolist(), data['e1_pos'].tolist(), \
                   data['e2_pos'].tolist(), data['elmo_embeddings'].tolist()
        return data['sentences'].tolist(), data['relations'].tolist(), data['e1_pos'].tolist(), \
               data['e2_pos'].tolist()
        # we need it in list format

    def get_full_data(self):
        data = pd.concat([self.relations_splits[0][t] for t in [DEV, TEST, TRAIN]])
        return data.values.tolist()

    # when reporting the scores for the paper, will merge dev and train set and will grab 0th fold of test
    def get_train_dev_data(self):
        data = pd.concate([self.relations_splits[0][t] for t in [TRAIN, DEV]])
        return data.values.tolist()

# given a string that looks like a list, parse it into an actual list
def argument_to_list(argument):
    return list(map(float, argument.strip('[]').split(',')))

# Given a string like "word_1" return "word"
# basically the word ends with _number and we want to split that up
def get_only_word(string):
    # below is a regex to get the group(1) of the string which means it just grabs whatever is
    # before the _
    return ''.join(re.findall("^(.*)_\d+$", string))
    #return " ".join(re.findall("[a-zA-Z]+", string))

def get_only_number(string):
    return ''.join(re.findall("^.*_(\d+)$", string))
    #return " ".join(re.findall("[0-9]+", string))


def stringify_tokenized(tokenizedSentence):
    return " ".join(tokenizedSentence)

# given a tokenized and splitted sentence
def sentence_replace(sentence, positions, string_update):
    return sentence[:positions[0]] + [string_update] + sentence[positions[1]+1:]

# sentence is the sentence to update and entity positions is a list of entity positions
def per_sentence_replacement_ddi(sentence, entity_positions):
    # if entity position is updated, then all positions after it also have to be updated
    
    e0_pos = entity_positions[0]
    sentence = sentence_replace(sentence, e0_pos, 'DRUG1')
    new_e0_pos = (e0_pos[0], e0_pos[0])
   
    entity_positions[0] = new_e0_pos
    diff = e0_pos[1] - e0_pos[0] # if the entity is 2 word, then move every other e_pos down by 1
    if entity_positions[0] == entity_positions[1]: # if both entities are the same
        entity_positions[1] = new_e0_pos
        return sentence, entity_positions
    if diff > 0:
        for i in range(1, len(entity_positions)):
            e_pos = entity_positions[i]
            if e_pos[0] > e0_pos[1]:
                entity_positions[i] = (entity_positions[i][0] - diff, entity_positions[i][1] - diff)
     
    e1_pos = entity_positions[1]
    sentence = sentence_replace(sentence, e1_pos, 'DRUG2')
    new_e1_pos = (e1_pos[0], e1_pos[0])
    
    entity_positions[1] = new_e1_pos
    diff = e1_pos[1] - e1_pos[0]
    if diff > 0 and len(entity_positions) > 2:
        for i in range(2, len(entity_positions)):
            e_pos = entity_positions[i]
            if e_pos[0] > e1_pos[1]:
                entity_positions[i] = (entity_positions[i][0] - diff, entity_positions[i][1] - diff)
    # then should handle for the case when there are more than entity 1 and entity 2 i.e. drug0 (any other drug)
    return sentence, entity_positions

# replace by DRUG1, DRUG2
def replace_by_drug_ddi(data):
    sentences, relations, e1_pos, e2_pos = data
    new_sentences = []
    new_e1_pos = []
    new_e2_pos = []
    for (sent, pos1, pos2) in zip(sentences, e1_pos, e2_pos):
        new_sent, new_positions = per_sentence_replacement_ddi(sent, [pos1, pos2])
        new_sentences.append(new_sent)
        new_e1_pos.append(new_positions[0])
        new_e2_pos.append(new_positions[1])
    return new_sentences, relations, new_e1_pos, new_e2_pos

def load_data(file_list):
        sentences = []
        relations = []
        e1_pos = []
        e2_pos = []

        for file in file_list:
                with open(file, 'r') as f:
                        for line in f.readlines():
                                line = line.strip().lower().split()
                                relations.append(int(line[0]))
                                e1_pos.append( (int(line[1]), int(line[2])) ) # (start_pos, end_pos)
                                e2_pos.append( (int(line[3]), int(line[4])) ) # (start_pos, end_pos)
                                sentences.append(line[5:])

        return sentences, relations, e1_pos, e2_pos

#stores the words with an index in the corpus organized from largest frequency to lowest frequency
def build_dict(sentences, low_freq_thresh=0, remove_stop_words=False):
    word_count = Counter()
    for sent in sentences:
        if sent is not None:
            for w in sent:
                if remove_stop_words is True and w in stop_words:
                    # first make sure to put stop words at the end so that they don't leave
                    # holes in the indexing
                    word_count[w] = -1
                    # make sure that they come after the words with frequency 1
                else:
                    word_count[w] += 1

    # the words from the low_freq_thresh wouldn't leave holes in the indexing because every word with
    # an index higher than them will be mapped to 0
    ls = word_count.most_common()
    # above organizes the words by most common and less common words; at this point we have the counts

    dictionary = {}
    for index, word_and_count in enumerate(ls):
        word = word_and_count[0]
        if remove_stop_words is True and word in stop_words:
            dictionary[word] = 0 #giving it a zero index
        elif low_freq_thresh > 0 and word_count[word] <= low_freq_thresh:
            dictionary[word] = 0
        else:
            dictionary[word] = index + 1
    return dictionary
        #basically every word with a count below that number should be sent to 0
    # leave 0 to pad
    # need to add conditions for when the remove stop words is True and low frequency words are there
    #return {word_and_count[0]: index + 1 for (index, word_and_count) in enumerate(ls)}
    # above is basically just creating a dictionary with key as the word and the value as the index of the word. Now index of 1 is for the highest frequency
    # whereas index of 2 is for lower frequency.

## stores the words with an index in the corpus organized from largest frequency to lowest
## frequency
#def build_dict(sentences):
#    word_count = Counter()
#    word_count[""] = -1
#    for sent in sentences:
#        for w in sent:
#            word_count[w] += 1
#
#    ls = word_count.most_common()
#    # above organizes the words by most common and less common words; at this point we have the counts
#
#    # leave 0 to PAD or for ""; in this case index always starts with "" being 0 so it is safe
#    # to keep the index as index
#    return {w[0]: index for (index, w) in enumerate(ls)}
#    # above is basically just creating a dictionary with key as the word nad the value as the index of the
#    # word. Now index of 1 is for the highest frequency whereas index of 2 is for lower frequency. 0 is for
#    # unknown words or "" which is used in the case of hypernyms


def load_embedding_senna(config, word_dict, normalize=False):
        emb_file = config.embedding_file
        emb_vocab = config.embedding_vocab

        vocab = {}
        with open(emb_vocab, 'r') as f:
                for id, w in enumerate(f.readlines()):
                        w = w.strip().lower()
                        vocab[w] = id

        f = open(emb_file, 'r')
        embed = f.readlines()

        dim = len(embed[0].split())
        num_words = len(word_dict) + 1
        embeddings = np.random.RandomState(seed=config.seed).uniform(-0.1, 0.1, size=(num_words, dim))
        config.embedding_size = dim

        pre_trained = 0
        for w in vocab.keys():
                if w in word_dict:
                        embeddings[word_dict[w]] = [float(x) for x in embed[vocab[w]].split()]
                        pre_trained += 1
        embeddings[0] = np.zeros((dim))

        logging.info('embeddings: %.2f%%(pre_trained) unknown: %d' %(pre_trained/float(num_words)*100, num_words-pre_trained))

        f.close()

        if normalize:
                        embeddings = embeddings * 0.1 / np.std(embeddings)

        return embeddings.astype(np.float32)

def load_embedding(config, word_dict, normalize=False):
                emb_file = config.embedding_file

                f = open(emb_file, 'r')
                f.readline()
                embed = f.readlines()

                dim = len(embed[0].strip().split()) - 1
                num_words = len(word_dict) + 1
                embeddings = np.random.RandomState(seed=config.seed).uniform(-0.1, 0.1, size=(num_words, dim))

                pre_trained = 0
                for line in embed:
                   line = line.strip().split()
                   word = line[0]
                   if word in word_dict:
                       try: embeddings[word_dict[word]] = list(map(float, line[1:]))
                       except: pass
                       else: pre_trained += 1
                # basically checking that if the word from embeddings file exists in our corpus isreversed_dictionary
                # then we will store it into a matrix called embeddings. Embeddings is going to be initialized by some random values
                # between -0.1 and 0.1 in a uniform manner. Now, whichever words are intersecting between embeddings file and dictionary will have
                # embeddings from the file whereas the words that are in the dictionary but not present in the embedding file will have a random embedding value.
                logging.info('embeddings: %.2f%%(pre_trained) unknown: %d' %(pre_trained/float(num_words)*100, num_words-pre_trained))
                f.close()

                if normalize:
                        embeddings = embeddings * 0.1 / np.std(embeddings)

                return embeddings.astype(np.float32)


# given the total data length, max sentence length, position of entities, compute the the
# relative distances of everything with respect to it
# TODO: (geeticka) think about whether it makes sense to use pos function in exactly the same way for the
# case when the sentences are trimmed short
def relative_distance(num_data, max_sen_len, e1_pos, e2_pos):
    dist1 = np.zeros((num_data, max_sen_len), dtype=int)
    dist2 = np.zeros((num_data, max_sen_len), dtype=int)
    # compute relative distance
    #TODO: (geeticka) think about what to do for the cases when e1_pos and e2_pos is None
    for sent_idx in range(num_data):
        for word_idx in range(max_sen_len):
            if e1_pos[sent_idx] is None or e2_pos[sent_idx] is None:
                continue
            if word_idx < e1_pos[sent_idx][0]:
                    dist1[sent_idx, word_idx] = pos(e1_pos[sent_idx][0] - word_idx)
            # in the above the word is behind the e1's word
            elif word_idx > e1_pos[sent_idx][1]:
                    dist1[sent_idx, word_idx] = pos(e1_pos[sent_idx][1] - word_idx)
            # the word is after the e1
            else:
                    dist1[sent_idx, word_idx] = pos(0)
            # the word is within the entity

            if word_idx < e2_pos[sent_idx][0]:
                    dist2[sent_idx, word_idx] = pos(e2_pos[sent_idx][0] - word_idx)
            elif word_idx > e2_pos[sent_idx][1]:
                    dist2[sent_idx, word_idx] = pos(e2_pos[sent_idx][1] - word_idx)
            else:
                    dist2[sent_idx, word_idx] = pos(0)

    num_pos = max(np.amax(dist1), np.amax(dist2)) - min(np.amin(dist1), np.amin(dist2))
    return dist1, dist2, num_pos

def pad_elmo_embedding(max_len, elmo_embeddings):
    new_elmo_embeddings = []
    for i in range(0, len(elmo_embeddings)):
        sentence = elmo_embeddings[i]
        num_of_words_to_pad = max_len - sentence.shape[1]
        array_to_pad = np.zeros(shape=(sentence.shape[0], num_of_words_to_pad, sentence.shape[2]),
        dtype='float32')
        appended_array = np.append(sentence, array_to_pad, axis=1)
        new_elmo_embeddings.append(appended_array)
    return new_elmo_embeddings

def vectorize(config, data, word_dict):
    def assign_splits(pos1, pos2):
        if pos1[1] < pos2[1]:
            return pos1[1], pos2[1]
        elif pos1[1] > pos2[1]:
            return pos2[1], pos1[1]
        elif config.use_piecewise_pool is True:
            raise Exception("Entity positions cannot end at the same position for piecewise splitting")
        else:
            return pos1[1], pos2[1] # this is not going to be used, but also cannot be none
            # I anticipate the above to be a problem for NER blinding, where there are 
            # overlaps between the entity pairs because the existence of the NER label extends the
            # entity pairs

    if config.use_elmo is True: sentences, relations, e1_pos, e2_pos, elmo_embeddings = data
    else: sentences, relations, e1_pos, e2_pos = data
    max_sen_len = config.max_len
    max_e1_len = config.max_e1_len
    max_e2_len = config.max_e2_len
    num_data = len(sentences)
    local_max_e1_len = max(list(map(lambda x: x[1]-x[0]+1, e1_pos)))
    local_max_e2_len = max(list(map(lambda x: x[1]-x[0]+1, e2_pos)))
    print('max sen len: {}, local max e1 len: {}, local max e2 len: {}'.format(max_sen_len, local_max_e1_len, local_max_e2_len))

    if config.use_elmo is True: padded_elmo_embeddings = pad_elmo_embedding(max_sen_len, elmo_embeddings)
    
    # maximum values needed to decide the dimensionality of the vector
    sents_vec = np.zeros((num_data, max_sen_len), dtype=int)
    e1_vec = np.zeros((num_data, max_e1_len), dtype=int)
    e2_vec = np.zeros((num_data, max_e2_len), dtype=int)
    # dist1 and dist2 are defined in the compute distance function
    
    position1 = [] # need to populate this way because want to make sure that the splits are in order
    position2 = []
    for idx, (sent, pos1, pos2) in enumerate(zip(sentences, e1_pos, e2_pos)):
        # all unseen words are mapped to the index 0
        vec = [word_dict[w] if w in word_dict else 0 for w in sent]
        sents_vec[idx, :len(vec)] = vec
        
        split1, split2 = assign_splits(pos1, pos2)
        position1.append(split1)
        position2.append(split2)

        # for the particular sentence marked by idx, set the entry as the vector gotten from above
        # which is basically just a list of the indexes of the words
        for ii in range(max_e1_len):
            if ii < (pos1[1]-pos1[0]+1):
                    e1_vec[idx, ii] = vec[range(pos1[0], pos1[1]+1)[ii]]
                    # this is assigning the particular sentence's e1 val to have the index of the corresponding word
            else:
                    e1_vec[idx, ii] = vec[pos1[-1]]
                    # in the above case it is grabbing the last word in the entity and padding with that

        for ii in range(max_e2_len):
            if ii < (pos2[1]-pos2[0]+1):
                    e2_vec[idx, ii] = vec[range(pos2[0], pos2[1]+1)[ii]]
            else:
                    e2_vec[idx, ii] = vec[pos2[-1]]

    dist1, dist2, num_pos = relative_distance(num_data, max_sen_len, e1_pos, e2_pos)
    
    if config.use_elmo is True: 
        return sents_vec, np.array(relations).astype(np.int64), e1_vec, e2_vec, dist1, dist2, \
    padded_elmo_embeddings, position1, position2
    return sents_vec, np.array(relations).astype(np.int64), e1_vec, e2_vec, dist1, dist2, position1, position2
    # we are also returning the ending positions of the entity 1 and entity 2

def pos(x):
        '''
        map the relative distance between [0, 123)
        '''
        if x < -60:
                        return 0
        if x >= -60 and x <= 60:
                        return x + 61
        if x > 60:
                        return 122

def batch_iter(seed, data, batch_size, shuffle=True):
        """
        Generates batches for the NN input feed.

        Returns a generator (yield) as the datasets are expected to be huge.
        """
        data = np.array(data)
        data_size = len(data)

        batches_per_epoch = int(np.ceil(data_size/float(batch_size)))

        # logging.info("Generating batches.. Total # of batches %d" % batches_per_epoch)

        if shuffle:
                indices = np.random.RandomState(seed=seed).permutation(np.arange(data_size))
                # refer to https://stackoverflow.com/questions/47742622/np-random-permutation-with-seed
                shuffled_data = data[indices]
        else:
                shuffled_data = data
        for batch_num in range(batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

def test_pred_writer(preds, relation_dict, save_path):
                sent_idx = 8000
                with open(save_path, 'w') as file:
                                for pred in preds:
                                                sent_idx += 1
                                                file.write('{}\t{}\n'.format(sent_idx, relation_dict[pred]))
                print('Answer writting done!')

def pred_writer(data, preds, relation_dict, save_path, fold_num):
        with open(save_path+'_fold{}'.format(fold_num), 'w') as file:
                for sent, relation, e1_pos, e2_pos, pred in zip(*(data + (preds,))):
                        file.write('Sentence:\t{}\nEntity 1:\t{}\nEntity 2:\t{}\nGround Truth:\t{}\tPrediction:\t{}\n\n'.format(' '.join(sent),
                                                ' '.join([sent[idx] for idx in range(e1_pos[0], e1_pos[1]+1)]),
                                                ' '.join([sent[idx] for idx in range(e2_pos[0], e2_pos[1]+1)]),
                                                relation_dict[relation], relation_dict[pred]))
        print('Answer writting done!')

def convert_labels(read_file, save_file):
        with open(save_file, 'w') as outfile:
                with open(read_file, 'r') as infile:
                        for line in infile:
                                line = line.strip().split()
                                label = int(line[0])
                                if label == 1:
                                        label = 18
                                elif label > 1:
                                        label -= 1
                                line = [str(label)] + line[1:] + ['\n']
                                outfile.write(' '.join(line))

# read the elmo embeddings for the train and the test file
def get_elmo_embeddings(filename):
    h5py_file = h5py.File(filename, 'r')
    elmo_embeddings = []
    # the h5py file contains one extra index for a new line character so must ignore that
    for i in range(0, len(h5py_file) - 1):
        embedding = h5py_file.get(str(i))
        elmo_embeddings.append(np.array(embedding))
    return (elmo_embeddings, )

# this function first split the line of data into relation, entities and sentence
# then cut the sentence according to the required border size
# if border size is -1, means using the full sentence, if it is 0, means only using
# the sentence between two entities (inclusive)
def split_data_cut_sentence(data, border_size=-1):
        sentences = []
        relations = []
        e1_pos = []
        e2_pos = []
        # is_reversed = []

        # In the parsed data: Num1 num2 num3 num4 num5 sentence
        # Num1 - relation number
        # Num2 - left entity start (starts the numbering from 0)
        # Num3 - left entity end
        # Num4 - right entity start
        # Num5 - right entity end
        if border_size < 0:
                for line in data:
                        line = line.strip().lower().split()
                        left_start_pos = int(line[1])
                        right_end_pos = int(line[4])
                        if left_start_pos < right_end_pos:
                                relations.append(int(line[0]))
                                e1_pos.append( (int(line[1]), int(line[2])) ) # (start_pos, end_pos)
                                e2_pos.append( (int(line[3]), int(line[4])) ) # (start_pos, end_pos)
                                # is_reversed.append( float(isreversed_dictionary[int(line[0])]) )
                                sentences.append(line[5:])
        else:
                for line in data:
                        line = line.strip().lower().split()
                        left_start_pos = int(line[1])
                        right_end_pos = int(line[4])
                        if left_start_pos < right_end_pos:
                                relations.append(int(line[0]))
                                # is_reversed.append( float(isreversed_dictionary[int(line[0])]) )
                                sentence = line[5:]
                                len_sen = len(sentence)
                                if left_start_pos >= border_size:
                                        left_border_size = border_size
                                else:
                                        left_border_size = left_start_pos
                                e1_pos.append( (left_border_size, int(line[2])-left_start_pos+left_border_size) ) # (start_pos, end_pos)
                                e2_pos.append((int(line[3])-left_start_pos+left_border_size, int(line[4])-left_start_pos+left_border_size)) # (start_pos, end_pos)
                                sentences.append(sentence[(left_start_pos-left_border_size):min(right_end_pos+border_size+1, len_sen)])

        return sentences, relations, e1_pos, e2_pos

def graph_file_reader(f_file, dim):
    """
    Give this function a file, and then read the individual files
    """
    f = open(f_file, 'r')
    num = sum(1 for line in open(f_file))

    vector = np.zeros((num, dim), dtype=float)
    i = 0
    for line in f:
        vec = line.split('\t')
        vec = vec[:-1]
        vec = [float(x) for x in vec]
        vector[i] = vec
        i+=1
    return num, vector


def graph_reader(entity_file, relation_file, dim):
    """
    Function returns entity and relation embeddings of Freebase formed using DKRL
    Taken from the code of "Learning beyond datasets"
    """
    ent_num, ent = graph_file_reader(entity_file, dim)
    rel_num, rel = graph_file_reader(relation_file, dim)
    print("Number of entities %d"%ent_num)
    print("Number of relations %d"%rel_num)

    return ent, rel

if __name__ == '__main__':
        # for item in load_data('data/test.txt'):
        #       print(item)
        convert_labels('data/train.txt', 'data/train_new.txt')
        convert_labels('data/test.txt', 'data/test_new.txt')
