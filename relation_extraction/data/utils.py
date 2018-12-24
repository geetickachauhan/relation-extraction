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
nlp = spacy.load('en')

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

def get_indiv_pos(position, words_num_dict, words_new_num_dict):
    if str(position) not in words_num_dict:
        return None
    word = words_num_dict[str(position)]
    return words_new_num_dict[word]

def get_new_entity_position(words, e1_pos, e2_pos, numbers):
    d1 = {numbers[i]: word for i, word in enumerate(words)} # words_num_dict
    d2 = {word: i for i, word in enumerate(words)} # words_new_num_dict
    # TODO: new_position may include just one word inside of the entities
    # which means that some word between the start and end might be in the path
    #i.e. we need to check for every entity word if None is returned by get_indiv_pos
    # because exactly one word's val returned will not be None
    for i in range(e1_pos[0], e1_pos[1] + 1):
        new_pos = get_indiv_pos(i, d1, d2)
        if new_pos is not None:
            new_e1_pos = (new_pos, new_pos)
    for i in range(e2_pos[0], e2_pos[1] + 1):
        new_pos = get_indiv_pos(i, d1, d2)
        if new_pos is not None:
            new_e2_pos = (new_pos, new_pos)
    return new_e1_pos, new_e2_pos

# 0 and 11 are opposites
# 1 and 16 are opposites
# 2 and 8 are opposites
# 3 and 10 are opposites
# 4 and 17 are opposites
# 5 and 15 are opposites
# 6 and 12 are opposites
# 7 and 13 are opposites
# 9 and 14 are opposites
# 18 does not have an opposite
# given a relation as a number, return the number of the opposite relation
def give_reverse_relation(relation):
    reverse_dict = {0: 11, 11:0, 1:16, 16:1, 2:8, 8:2, 3:10, 10:3, 4:17,
                    17:4, 5:15, 15:5, 6:12, 12:6, 7:13, 13:7, 9:14, 14:9, 18:18}
    return reverse_dict[relation]

# get the length of the shortest path between two entities
# in a graph. If path does not exist, returns None
def get_path_length(entity1, entity2, graph):
    # sometimes the entities don't exist in the graph altogether because maybe they have no
    # connections with the other words in the sentence
    if entity1 not in graph:
        return None
    if entity2 not in graph:
        return None
    if nx.has_path(graph, entity1, entity2):
        length = nx.shortest_path_length(graph, source=entity1, target=entity2)
        return length
    else:
        return None

# given the shortest path and graph, get the path with the dependency
# features i.e. for a path ['name_2', 'is_3'], the returned value will be
# ['name_2', 'ROOT', 'is_3']
# This isn't used by data augmentation, but is going to be used by the CNN model
# to featurize the dependency labels
def get_path_with_edge_name(graph, path):
    edge_attributes = nx.get_edge_attributes(graph, 'name')
    path_with_edge_names = []
    path_with_edge_names_onlywords = [] # to have the option to get the words without the nums
    for i in range(0, len(path)-1):
        path_with_edge_names.append(path[i])
        path_with_edge_names_onlywords.append(get_only_word(path[i]))
        w1 = path[i]
        w2 = path[i+1]
        edge = edge_attributes[(w1,w2) if (w1,w2) in edge_attributes else (w2,w1)]\
                + "_" + (str(0) if (w1,w2) in edge_attributes else str(1))
        path_with_edge_names.append(edge)
        path_with_edge_names_onlywords.append(edge)
    path_with_edge_names.append(path[-1])
    path_with_edge_names_onlywords.append(get_only_word(path[-1]))
    return path_with_edge_names, path_with_edge_names_onlywords

def stringify_tokenized(tokenizedSentence):
    return " ".join(tokenizedSentence)

# given a tokenized sentence eg: ["The", "bear", "ran", "home"] and
# positions of the two entities, get the shortest dependency path between
# both entities
# in case of multi word entities, grabs the min path length between all possible
# words in the entities. It is possible to have no shortest dependency path
# between two entities, in which case None, None is returned
def get_shortest_dependency_path(tokenizedSentence, e1_pos, e2_pos):
    #sentence = stringify_tokenized(tokenizedSentence)
    #parsedData = parser(sentence)
    # specify a pre-tokenized sentence to spacy and run parts of the pipeline
    doc = Doc(nlp.vocab, words=tokenizedSentence)
    nlp.tagger(doc)
    nlp.parser(doc)
    edges = []
    #print("Tokenized Sentence, e1, e2", tokenizedSentence, e1_pos, e2_pos)
    # it is possible that spacy is giving extra tokens to the sentence than are needed
    # which means at this stage we need to make sure that we remove those from the
    # parsedData before we tokenize everything or just adjust the indexes
    for token in doc:
        # FYI https://spacy.io/docs/api/token
        for child in token.children:
            edges.append(('{0}_{1}'.format(token.lower_,token.i), #head
                          '{0}_{1}'.format(child.lower_,child.i), #child
                         {'name': token.dep_})) #name of the dependency
            #print("appending edges", token.lower_, token.i, child.lower_, child.i, token.dep_)

    graph = nx.Graph(edges)
    entity1 = [tokenizedSentence[e1_pos[i]]+ "_" + str(e1_pos[i]) for i in range(0, len(e1_pos))]
    entity2 = [tokenizedSentence[e2_pos[i]]+ "_" + str(e2_pos[i]) for i in range(0, len(e2_pos))]
    path_length = []
    for e1 in entity1:
        for e2 in entity2:
            length = get_path_length(e1, e2, graph)
            if length is not None:
                path_length.append({'len': length, 'e1': e1, 'e2': e2})
    if not path_length: # means list is empty which means that path didnt exist at all
        return None, None, None
    minItem = min(path_length, key=lambda x: x['len'])
    path = nx.shortest_path(graph, source=minItem['e1'], target=minItem['e2'])
    path_with_edge_names, path_with_edge_names_onlywords = get_path_with_edge_name(graph, path)
    return path, path_with_edge_names, path_with_edge_names_onlywords

# given splitted training/ testing data, return the paths, paths_e1_pos etc for the data
def shortest_dependency_path_full_data(data):
    sentences = data[0]
    relations = data[1]
    e1_pos = data[2]
    e2_pos = data[3]
    paths = []
    paths_with_edge_names = []
    paths_e1_pos = []
    paths_e2_pos = []
    for idx, (sent, pos1, pos2, rel) in enumerate(zip(sentences, e1_pos, e2_pos, relations)):
        path, _, path_with_edge_names = get_shortest_dependency_path(sent, pos1, pos2)
        if path is not None:
            oldnums = [get_only_number(word) for word in path]
            path = [get_only_word(word) for word in path]
            path_pos1, path_pos2 = get_new_entity_position(path, pos1, pos2, oldnums)
        else:
            path = None
            path_pos1 = None
            path_pos2 = None
        paths.append(path)
        paths_e1_pos.append(path_pos1)
        paths_e2_pos.append(path_pos2)
        paths_with_edge_names.append(path_with_edge_names)
    return paths, paths_e1_pos, paths_e2_pos, paths_with_edge_names

# given splitted training data, extend it by reversing the sentences and give the relevant pos embeddings
# if simple is false, augment the data with the shortest dependency path between the entities
def augment_data(data, simple=True):
    sentences = data[0]
    relations = data[1]
    e1_pos = data[2]
    e2_pos = data[3]
    paths = data[4]
    paths_e1_pos = data[5]
    paths_e2_pos = data[6]
    augmented_sentences = []
    augmented_relations = []
    augmented_e1_pos = []
    augmented_e2_pos = []
    augmented_relations = []
    for idx, (sent, pos1, pos2, rel, path, path_pos1, path_pos2) in enumerate(zip(sentences,\
            e1_pos, e2_pos, relations, paths, paths_e1_pos, paths_e2_pos)):
        if simple is True: # in this case everything is reversed
            augmented_sent = list(reversed(sent))
            augmented_rel = give_reverse_relation(rel)
            augmented_pos1_first = len(sent) - (pos2[1] + 1)
            augmented_pos1_second = len(sent) - (pos2[0] + 1)
            augmented_pos2_first = len(sent) - (pos1[1] + 1)
            augmented_pos2_second = len(sent) - (pos1[0] + 1)
            augmented_pos1 = (reversed_pos1_first, reversed_pos1_second)
            augmented_pos2 = (reversed_pos2_first, reversed_pos2_second)
        else: # in this case only the shortest dependency path is considered
            #print("Sentence", sent)
            #path = get_shortest_dependency_path(sent, pos1, pos2)[0]
            if path is None:
                continue
            #print("Path", path)
            #oldnums = [get_only_number(word) for word in path]
            #print("Old indexing", oldnums)
            #augmented_sent = [get_only_word(word) for word in path]
            #augmented_pos1, augmented_pos2 = get_new_entity_position(augmented_sent, pos1, pos2, oldnums)
            #augmented_rel = rel
            augmented_sent = path
            augmented_pos1 = path_pos1
            augmented_pos2 = path_pos2
            augmented_rel = rel
        augmented_sentences.append(augmented_sent)
        augmented_relations.append(augmented_rel)
        augmented_e1_pos.append(augmented_pos1)
        augmented_e2_pos.append(augmented_pos2)
    sentences = sentences + augmented_sentences
    relations = relations + augmented_relations
    e1_pos = e1_pos + augmented_e1_pos
    e2_pos = e2_pos + augmented_e2_pos

    return sentences, relations, e1_pos, e2_pos


#input must be a tokenized sentence in the form ['There', 'is', 'a', 'dog', '.']
def get_hypernyms_per_sentence(tokenizedSentence):
    hypernyms = []
    for word in tokenizedSentence:
        hypernym_perword = []
        if len(wn.wordnet.synsets(word)) == 0:
            hypernyms.append("")
            continue
        foundhypernym = 0
        for synset in wn.wordnet.synsets(word):
            if len(synset.hypernyms()) > 0:
                hypernym = synset.hypernyms()[0]
                foundhypernym += 1
                hypernyms.append(hypernym.name())
                break
        if foundhypernym == 0:
            hypernyms.append("")
    return hypernyms


# This function generates the hypernyms of a tokenized sentence and splits out only the words
# it has a flag to indicate whether to give only the first word or all of them
def get_hypernyms_per_sentence_onlywords(tokenizedSentence, onlyFirstWord=False):
    #print("Only getting the hypernyms for the words")
    hypernyms = get_hypernyms_per_sentence(tokenizedSentence)
    hypernyms_onlywords = []
    for hypernym in hypernyms:
        if hypernym == '':
            hypernyms_onlywords.append(hypernym)
            continue
        word = " ".join(re.findall("[a-zA-Z]+", hypernym))
        last_character = word[len(word)-1:]
        if(last_character == 'v' or last_character == 'n' or last_character == 'a' or last_character == 'r'):
            word = word[:len(word)-1]
            word = word.strip()
        if onlyFirstWord == True:
            #print("Only the first word is being extracted")
            word = word.split()[0]
        hypernyms_onlywords.append(word)
    return hypernyms_onlywords

#given data in the form of split_data_cut_sentences output, return the hypernyms of the
# entities, onlyWordsAndFirstWord is a flag to indicate whether you want only the words of the hypernyms
# and if you want only the first word from that eg: in travel_rapidly.v.01, when both flags are true,
# you would get 'travel' as the hypernym
def get_hypernyms(data, onlyWordsAndFirstWord=[False, False]):
    sentences = data[0]
    e1_pos = data[2]
    e2_pos = data[3]
    num_data = len(sentences)

    e1_hypernym = []
    e2_hypernym = []
    for idx, (sent, pos1, pos2) in enumerate(zip(sentences, e1_pos, e2_pos)):
        e1 = [sent[x] for x in range(pos1[0], pos1[1]+1)]
        e2 = [sent[x] for x in range(pos2[0], pos2[1]+1)]
        if onlyWordsAndFirstWord[0] == True:
            hyp1 = get_hypernyms_per_sentence_onlywords(e1, onlyWordsAndFirstWord[1])
            hyp2 = get_hypernyms_per_sentence_onlywords(e2, onlyWordsAndFirstWord[1])
        else:
            hyp1 = get_hypernyms_per_sentence(e1)
            hyp2 = get_hypernyms_per_sentence(e2)
        #hyp1 = " ".join(map(str, hyp1))
        #e1_hypernym[idx, :len(hyp1)] = hyp1
        e1_hypernym.append(hyp1)
        #hyp2 = " ".join(map(str, hyp2))
        e2_hypernym.append(hyp2)
        #e2_hypernym[idx, :len(hyp2)] = get_hypernyms_per_sentence(e2)
    return e1_hypernym, e2_hypernym


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
def build_dict(sentences, remove_stop_words=False, low_freq_thresh=0):
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

# change number to 0 if it is larger than a cap
def threshold(num, cap):
    if(num > cap):
        return 0
    return num

# vectorizes a word's hypernym, assuming that it consists of multiple words. Even if there is just one word,
# it will be treated as a list
def vectorize_hypernym_word(hyp, hypernym_dict, config):
    return [threshold(hypernym_dict[h], config.hyp_embed_num) if h!="" and h in hypernym_dict else 0 for h in hyp]

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


    for idx, (sent, pos1, pos2) in enumerate(zip(sentences, e1_pos, e2_pos)):
        # all unseen words are mapped to the index 0
        vec = [word_dict[w] if w in word_dict else 0 for w in sent]
        sents_vec[idx, :len(vec)] = vec

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
    
    # returning the ending positions of both entities
    position1 = list(map(lambda x: x[1], e1_pos))
    position2 = list(map(lambda x: x[1], e2_pos))
    if config.use_elmo is True: 
        e1_pos_as_list = list(map(lambda x: [x[0], x[1]], e1_pos))
        e2_pos_as_list = list(map(lambda x: [x[0], x[1]], e2_pos))
        return sents_vec, np.array(relations).astype(np.int64), e1_vec, e2_vec, dist1, dist2, \
    padded_elmo_embeddings, position1, position2, e1_pos_as_list, e2_pos_as_list
    
    return sents_vec, np.array(relations).astype(np.int64), e1_vec, e2_vec, dist1, dist2, position1, \
            position2
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
