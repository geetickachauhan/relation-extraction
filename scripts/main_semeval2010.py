from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import logging
import os
import sys
sys.path.append('..')
import time
import datetime
import random
import uuid # for generating a unique id for the cnn

import relation_extraction.data.utils
#import argparse
from relation_extraction.data.models.model import Model
import parser

import datetime
import copy
import json

logging.getLogger().setLevel(logging.INFO)
#parser = argparse.ArgumentParser()

relation_dict = {0:'Component-Whole(e2,e1)', 1:'Instrument-Agency(e2,e1)', 2:'Member-Collection(e1,e2)',
3:'Cause-Effect(e2,e1)', 4:'Entity-Destination(e1,e2)', 5:'Content-Container(e1,e2)',
6:'Message-Topic(e1,e2)', 7:'Product-Producer(e2,e1)', 8:'Member-Collection(e2,e1)',
9:'Entity-Origin(e1,e2)', 10:'Cause-Effect(e1,e2)', 11:'Component-Whole(e1,e2)',
12:'Message-Topic(e2,e1)', 13:'Product-Producer(e1,e2)', 14:'Entity-Origin(e2,e1)',
15:'Content-Container(e2,e1)', 16:'Instrument-Agency(e1,e2)', 17:'Entity-Destination(e2,e1)',
18:'Other'}

isreversed_dictionary = {0:'1', 1:'1', 2:'0', 3:'1', 4:'0', 5:'0', 6:'0', 7:'1',
8:'1', 9:'0', 10:'0', 11:'0', 12:'1', 13:'0', 14:'1', 15:'1', 16:'0', 17:'1', 18:'0'}
config = parser.get_config()

config.classnum = max(relation_dict.keys()) #TODO (geeticka): remove all arguments from config that are not
# passed in, for example folds and macro_f1_folds etc

#TODO: (geeticka) change above to not say +1 : we are not considering the "other" class.

# remove the is reversed FEATURE
def res(path): return os.path.join(config.data_root, path)

TRAIN, DEV, TEST = 0, 1, 2
#TODO: (geeticka) when you read the dependency paths with labels, use get_only_words
dataset = \
utils.Dataset(res('pickled-files/seed_{K}_10-dep-dir-fold-border_{N}.pkl').format(K=config.pickle_seed,
    N=config.border_size))
print("border size:", 'pickled-files/seed_{K}_10-dep-dir-fold-border_{N}.pkl'.format(K=config.pickle_seed,
    N=config.border_size))

# performs the prediction, but makes sure that if all the scores are negative, predict the class "Other"
def prediction(scores):
    data_size = scores.shape[0]
    pred = np.zeros(data_size)
    for idx in range(data_size):
        data_line = scores[idx]
        if all(data_line <= 0.):
            pred[idx] = 18
        else:
            pred[idx] = np.argmax(data_line)

    return pred

# calculates the accuracy since we have a different prediction procedure
def accuracy(preds, labels):
    correct = np.equal(preds, labels).astype(np.int8)
    # correct[labels==18] = 1
    return correct.sum()

def run_epoch(session, model, batch_iter, epoch, verbose=True, is_training=True):
    start_time = time.time()
    acc_count = 0
    step = 0 #len(all_data)
    tot_data = 0
    preds = []
    scores = []


    for batch in batch_iter:
        step += 1
        tot_data += batch.shape[0]
        batch = (x for x in zip(*batch))
        # because the batch contains the sentences, e1, e2 etc all as separate lists, zip(*) makes it
        # such that every line of the new tuple contains the first element of sentences, e1, e2 etc
        sents, relations, e1, e2, dist1, dist2 = batch
        sents = np.vstack(sents)
        in_x, in_e1, in_e2, in_dist1, in_dist2, in_y, in_epoch = model.inputs
        feed_dict = {in_x: sents, in_e1: e1, in_e2: e2, in_dist1: dist1, in_dist2: dist2, \
                in_y: relations, in_epoch: epoch}
        if is_training:
            _, scores, loss, summary = session.run(
                [model.train_op, model.scores, model.loss, model.merged_summary],
                feed_dict=feed_dict
            )
            pred = prediction(scores)
            acc = accuracy(pred, relations)
            # global_step is not step + epoch*config.batch_size
            global_step = tf.train.global_step(session, model.global_step)
            model.writer.add_summary(summary, global_step)
            # summary, merged_summary
            acc_count += acc
            if verbose and step%10 == 0:
                logging.info(
                    "  step: %d acc: %.2f%% loss: %.2f time: %.2f"
                    "" % (
                        step, acc_count / (step * config.batch_size) * 100, loss,
                        time.time() - start_time
                    )
                )
        else:
            #TODO: (geeticka) figure out why merged_summary doesn't exist for the non train model
            scores, = session.run(
                    [model.scores],
                    feed_dict=feed_dict
            )
            pred = prediction(scores)
            acc = accuracy(pred, relations)
            acc_count += acc

        preds.extend(pred)
            #global_step = tf.train.global_step(session, model.global_step)
            #model.writer.add_summary(summary, global_step)

    return acc_count / (tot_data), preds


# this is the split dataset just for the train data
#above is the dataset containing the splitted information along with the dependency paths

# vectorize will have different sizes for train and test depending on whether the data is truncated to
# not include the paths as a different column

# a function to open the file and return a list consisting of the lines in the file
# this is needed to create the dev file from the train file, otherwise a TypeError is thrown because
# when a file is originally opened, it is of type '_io.TextIOWrapper'
def openFileAsList(filename):
    with open(filename) as f:
        mylist = [line.rstrip('\n') for line in f]
    return mylist

# let us remove the depedency data
#split the data, and get the dependency information for the non cross validated data
def preprocess_data_noncrossvalidated(data):
    data = utils.split_data_cut_sentence(data, config.border_size)
    return data



def init():

    #
    # Config log
    if config.log_file is None:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    else:
        if not os.path.exists(config.save_path): os.makedirs(config.save_path)

        logging.basicConfig(
            filename=config.log_file, filemode='a', level=logging.DEBUG, format='%(asctime)s %(message)s',
            datefmt='%m-%d %H:%M'
        )
    if config.fold is None and config.cross_validate is False:
        config.train_text_dataset_file = res(config.train_text_dataset_path)
        config.test_text_dataset_file = res(config.test_text_dataset_path)
        train_data = openFileAsList(config.train_text_dataset_file)
        test_data = open(config.test_text_dataset_file, 'r')
    elif config.fold is None and config.cross_validate is True:
        print('Error: Fold is not None but cross validate is True')
        logging.info('Error: Fold is not None but cross validate is True')
        return
    else:
        train_data = dataset.get_data_for_fold(config.fold)
        dev_data = dataset.get_data_for_fold(config.fold, DEV)
        test_data = dataset.get_data_for_fold(config.fold, TEST)

    # now each of the above data contains the following in order:
    # sentences, relations, e1_pos, e2_pos, paths, paths_e1_pos, paths_e2_pos, paths_with_edge_names

    # if you are using the pickle file with unsplit sentences, you will do the following:
    # random select dev set
    if config.cross_validate is False:
        if config.use_test is False:
            devsize = int(len(train_data)*0.15)
            select_index = random.sample(range(0, len(train_data)), devsize)
            dev_data = [train_data[idx] for idx in range(len(train_data)) if idx in select_index]
            train_data = list(set(train_data) - set(dev_data))
            if config.early_stop is True:
                early_stop_size = int(len(devsize)*0.5)
                select_index = random.sample(range(0, len(dev_data)), early_stop_size)
                early_stop_data = [dev_data[idx] for idx in range(len(dev_data)) if idx in select_index]
                dev_data = list(set(dev_data) - set(early_stop_data))
        elif config.use_test is True:
            dev_data = open(config.test_text_dataset_file, 'r') # means we will report test scores
        # split data
        train_data = preprocess_data_noncrossvalidated(train_data)
        dev_data = preprocess_data_noncrossvalidated(dev_data)
        test_data = preprocess_data_noncrossvalidated(test_data)
        if config.use_test is False and config.early_stop is True:
            early_stop_data = preprocess_data_noncrossvalidated(early_stop_data)
        elif config.use_test is True and config.early_stop is True:
            raise NotImplemented

    # if you are using the pickle file with everything split up and dependency information
    # stored, do the following:
    # only need below if doing early stop
    if config.early_stop is True and config.cross_validate is True:
        early_stop_size = int(len(train_data[0])*config.early_stop_size)
        select_index = random.sample(range(0, len(train_data[0])), early_stop_size)
        new_train_data = []
        early_stop_data = []
        for items in train_data:
            early_stop_items = [items[idx] for idx in range(len(items)) if idx in select_index]
            train_items = [items[idx] for idx in range(len(items)) if idx not in select_index]
            #train_items = list(set(items) - set(dev_items))
            new_train_data.append(train_items)
            early_stop_data.append(early_stop_items)
        train_data = tuple(new_train_data)
        early_stop_data = tuple(early_stop_data)

    logging.info('size of train data: %d' % len(train_data[0]))
    logging.info('size of dev data: %d' % len(dev_data[0]))
    logging.info('size of test data: %d' % len(test_data[0]))
    if config.early_stop is True:
        logging.info('size of early stop data: %d' % len(early_stop_data[0]))
    print("early stop is", config.early_stop)
    print("lr_values and boundaries are", config.lr_values, config.lr_boundaries)
    print("seed for random initialization is ",  config.seed)

    # Build vocab
    all_data = train_data[0] + dev_data[0] + test_data[0]
    if config.early_stop is False:
        early_stop_data = dev_data
    #TODO: (geeticka) improve the above; handle the situation when the early stop data is non existant
    early_stop_data_addition = early_stop_data[0]

    if config.early_stop is True:
        all_data = all_data + early_stop_data_addition
    word_dict = utils.build_dict(all_data, config.remove_stop_words, config.low_freq_thresh)
    logging.info('total words: %d' % len(word_dict))

    embeddings = utils.load_embedding_senna(config, word_dict)

    # grabbing the length of the longest sentence in each of the train, test and dev
    max_len_train = len(max(train_data[0], key=lambda x:len(x)))
    max_len_dev = len(max(dev_data[0], key=lambda x:len(x)))
    max_len_test = len(max(test_data[0], key=lambda x:len(x)))
    config.max_len = max(max_len_train, max_len_dev, max_len_test)
    # config.max_len = max(max_len_train, max_len_test)

    # max entity lengths, computing them via subtraction because
    # we have access to the positions of the words inside of the sentence
    max_e1_len_train = max(map(lambda x: x[1]-x[0]+1, train_data[2]))
    max_e1_len_dev = max(map(lambda x: x[1]-x[0]+1, dev_data[2]))
    max_e1_len_test = max(map(lambda x: x[1]-x[0]+1, test_data[2]))
    config.max_e1_len = max(max_e1_len_train, max_e1_len_dev, max_e1_len_test)
    # config.max_e1_len = max(max_e1_len_train, max_e1_len_test)

    max_e2_len_train = max(map(lambda x: x[1]-x[0]+1, train_data[3]))
    max_e2_len_dev = max(map(lambda x: x[1]-x[0]+1, dev_data[3]))
    max_e2_len_test = max(map(lambda x: x[1]-x[0]+1, test_data[3]))
    config.max_e2_len = max(max_e2_len_train, max_e2_len_dev, max_e2_len_test)

    if config.early_stop is True:
        max_len_earlystop = len(max(early_stop_data[0], key=lambda x:len(x)))
        max_e1_len_earlystop = max(map(lambda x: x[1]-x[0]+1, early_stop_data[2]))
        max_e2_len_earlystop = max(map(lambda x: x[1]-x[0]+1, early_stop_data[3]))
        max_len_dep_earlystop = len(max(early_stop_data[index], key=lambda x:len(x) if x is not None else 0))
        config.max_len = max(config.max_len, max_len_earlystop)
        config.max_e1_len = max(config.max_e1_len, max_e1_len_earlystop)
        config.max_e2_len = max(config.max_e2_len, max_e2_len_earlystop)
        config.max_len_dep = max(config.max_len_dep, max_len_dep_earlystop)

    train_vec = utils.vectorize(config, train_data, word_dict)
    test_vec = utils.vectorize(config, test_data, word_dict)
    dev_vec = utils.vectorize(config, dev_data, word_dict)
    if config.early_stop is True:
        early_stop_vec = utils.vectorize(config, early_stop_data, word_dict)

    # finally I need the original entity info in the test file
    # previously there was [] as the input
    dev_data_orin = dev_data[1]
    if config.early_stop is True:
        print("returning the early stop data")
        early_stop_data_orin = early_stop_data[1]
        return embeddings, train_vec, dev_vec, test_vec, dev_data_orin, early_stop_vec, early_stop_data_orin

    return embeddings, train_vec, dev_vec, test_vec, dev_data_orin
    # return embeddings, train_vec, test_vec, test_data[1]
    # need to return the relations for the test data in the last part

def test_writer(scores, answer_path):
        np.save(answer_path, scores)
        answer_txt_path = answer_path.replace('npy', 'txt')
        # scores = scores.tolist()
        with open(answer_txt_path, 'w') as inFile:
                for score in scores:
                        inFile.write('\t'.join([str(item) for item in score])+'\n')

        if config.cross_validate is False:
            print('Answer file writting done!')

def test_writer_for_perl_evaluation(relations, preds, answer_dict, answer_path):
        # all_relations = relations_dev_df['Left Entity'].unique().tolist()
        #for testfile you will want to write down starting from 8001 as the sentence ID
        # if you use Di's key file
        # basically rewrite this such that you now accept data in Di's format
        # data is going to be the test file, preds is predictions, answer_dict is the relations dictionary
        # answer_path is where to store these files
        sentence = 1
        output_filepath_gold = answer_path.replace('.txt', '') +'_gold.txt'
        with open(answer_path, 'w') as outFile_prediction, open(output_filepath_gold, 'w') as outFile_gold:
                for relation, pred in zip(relations, preds):
                        # argument1 =  entity.split('(')[1].split(',')[0]
                        # if argument1 not in all_relations: continue
                        # relation = '(' + ''.join(entity.split('(')[1:])
                        # outFile_prediction.write('{0}{1}\n'.format(answer_dict[pred], relation))
                        # outFile_gold.write('{0}\n'.format(entity))
                        outFile_prediction.write('{0}\t{1}\n'.format(sentence, answer_dict[pred]))
                        outFile_gold.write('{0}\t{1}\n'.format(sentence, answer_dict[relation]))
                        sentence += 1
        #print('Answer file writing done!')
        return output_filepath_gold

def read_macro_f1_from_result_file(result_filepath):
        '''
        Retrieve the macro F1 score from the result file that perl eval/semeval2018_task7_scorer-v1.2.pl generates
        '''
        result_file = open(result_filepath, 'r')
        for cur_line in result_file:
                if cur_line.startswith('<<< The official score is'):
                        cur_line = cur_line.replace('<<< The official score is (9+1)-way evaluation with directionality taken into account: macro-averaged F1 = ','')
                        cur_line = cur_line.replace('% >>>','')
                        macro_f1 = float(cur_line)
        result_file.close()
        return macro_f1

def get_current_date():
    '''
    https://www.saltycrane.com/blog/2008/06/how-to-get-current-date-and-time-in/
    '''
    now = datetime.datetime.now()
    return str(now.year) + '-' + str(now.month) + '-' + str(now.day)
    # this is solely used to identify when the command to start the experiment was run

def get_current_time_in_seconds():
        '''
        http://stackoverflow.com/questions/415511/how-to-get-current-time-in-python
        '''
        return(time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime()))

def get_current_time_in_miliseconds():
        '''
        http://stackoverflow.com/questions/5998245/get-current-time-in-milliseconds-in-python
        '''
        return(get_current_time_in_seconds() + '-' + str(datetime.datetime.now().microsecond))

def create_folder_if_not_exists(dir):
        '''
        Create the folder if it doesn't exist already.
        '''
        if not os.path.exists(dir):
                os.makedirs(dir)

# given data that has different lists for relations, sentences etc,
# combine them such that relation and sentence etc are combined per entry
# eg: if sentence was [0,1] and relation was [5,6]
# return value would be [0,5] and [1,6]
def stack_data(data):
    return list(zip(*data))

#read the macro F1 from the necessary filepath
def evaluate(result_filepath, answer_filepath, data_orin, preds):
    output_filepath_gold = test_writer_for_perl_evaluation(
        data_orin, preds, relation_dict, answer_filepath
    )
    command = (
        "perl eval/semeval2010_task8_scorer-v1.2.pl {0} {1} > {2}"
        "".format(
            answer_filepath, output_filepath_gold, result_filepath
        )
    )

    os.system(command)
    macro_f1 = read_macro_f1_from_result_file(result_filepath)
    return macro_f1

def output_model(config):
    #parameters['data_augmentation'] = config.data_augmentation
    train_start_time_in_miliseconds = get_current_time_in_miliseconds()
    config.train_start_folds.append(train_start_time_in_miliseconds)
    #TODO: update this based on whatever hyperparameter changes you make
    #hyperparameters = ['hyp_embed_size', 'hyp_embed_reduce']
    results, parameters = get_results_dict(config, train_start_time_in_miliseconds)
    hyperparameters = ['id', 'dataset', 'pos_embed_size', 'num_filters', 'filter_sizes', 'keep_prob', 'early_stop', 'patience']
    hyperparam_dir_addition = '-'.join(['{}_{:.6f}'.format(parameter, parameters[parameter]) if
            type(parameters[parameter])==float else '{}_{}'.format(parameter,
                parameters[parameter]) for parameter in hyperparameters])
    config.add_hyperparam_details = "Di's code + senna embeddings" + \ + " Early Stop: " + \
    str(config.early_stop) + " Patience: " + str(config.patience) + " Pickle Seed: " + str(config.pickle_seed) + \
    " Removing stop words: " + str(config.remove_stop_words) + " Low frequency words threshold: " + \
    str(config.low_freq_thresh) + " Seed for initializer" + str(config.seed) + \
    " ; use the full train and test: " + str(config.use_test)
    #config.add_hyperparam_details = "No cross validation with WPE: dev results reported are actually on test data"

    config.parameters = parameters
    create_folder_if_not_exists(config.output_dir)
    if config.fold is not None and config.cross_validate is True:
        folder_string = "CrossValidation"
        config.output_folder = os.path.join(config.output_dir, "CrossValidation", model_name, 'Fold'+str(config.fold))
    else:
        folder_string = "NoCrossValidation"
        config.output_folder = os.path.join(config.output_dir,"NoCrossValidation", model_name)
    #config.tensorboard_folder = os.path.join(config.output_dir, "Tensorboard", hyperparam_dir_addition)

    config.tensorboard_folder = os.path.join(config.output_dir, folder_string)
    config.result_folder = os.path.join(config.output_dir, folder_string, model_name, 'Result')
    print("Tensorboard folder, shared across all folds, different for each run is",
            config.tensorboard_folder)
    create_folder_if_not_exists(config.output_folder)
    create_folder_if_not_exists(config.tensorboard_folder)
    create_folder_if_not_exists(config.result_folder)
    config.test_answer_filepath = os.path.join(config.output_folder, config.test_answer_file)

    return results, parameters, model_name

def main():

        if config.early_stop is True:
            embeddings, train_vec, dev_vec, test_vec, dev_data_orin, early_stop_vec, \
            early_stop_data_orin = init()
        else:
            embeddings, train_vec, dev_vec, test_vec, dev_data_orin = init()
        # embeddings, train_vec, test_vec, test_relations = init()
        bz = config.batch_size

        # Need to clean up the following; send it to a function
        # Add some logging


        results, parameters = output_model(config)
        # above method is general and so below I am adding stff specific to folds
        parameters['fold'] = config.fold
        results['fold'] = config.fold
        results['epoch'] = {}
        if config.fold is not None and config.cross_validate is True:
            parameters['fold'] = config.fold
            model_name = 'cnn_{0}'.format(train_start_time_in_miliseconds +'-'+ 'Fold-'+
                str(config.fold) + hyperparam_dir_addition)
        else:
            model_name = 'cnn_{0}'.format(train_start_time_in_miliseconds+ hyperparam_dir_addition)

        # below is code for running the model itself
        with tf.Graph().as_default():
            with tf.name_scope("Train"):
                with tf.variable_scope("Model", reuse=None):
                    m_train = Model(config, embeddings, is_training=True)

            with tf.name_scope("Valid"):
                with tf.variable_scope("Model", reuse=True):
                    m_eval = Model(config, embeddings, is_training=False)

            # Start TensorFlow session
            sv = tf.train.Supervisor(logdir=config.save_path, global_step=m_train.global_step)

            print("Output folder is the following", config.output_folder)

            configProto = tf.ConfigProto()
            configProto.gpu_options.allow_growth = True

            with sv.managed_session(config=configProto) as session:
                if config.early_stop is True:
                    best_early_stop_macro_f1 = 0
                    best_early_stop_macro_f1_epoch_number = -1
                    patience_counter = 0
                try:
                    for epoch in range(config.num_epoches):
                        results['epoch'][epoch] = {}
                        train_iter = utils.batch_iter(config.seed, stack_data(train_vec), bz, shuffle=True)
                        dev_iter   = utils.batch_iter(config.seed, stack_data(dev_vec),   bz, shuffle=False)
                        test_iter  = utils.batch_iter(config.seed, stack_data(test_vec),  bz, shuffle=False)
                        if config.early_stop is True:
                            early_stop_iter = utils.batch_iter(config.seed, stack_data(early_stop_vec), bz, shuffle=False)
                        train_verbosity = False if config.cross_validate is False else True
                        train_acc, _ = run_epoch(session, m_train, train_iter, epoch, verbose=False)

                        # TODO(geeticka): Why separate model (e.g. why m_eval vs. m_train)?
                        dev_acc, dev_preds = run_epoch(
                            session, m_eval, dev_iter, epoch, verbose=False, is_training=False
                        )

                        config.result_filepath = os.path.join(config.output_folder, config.result_file)
                        config.dev_answer_filepath = os.path.join(
                            config.output_folder, config.dev_answer_file
                        )

                        macro_f1_dev = evaluate(config.result_filepath, config.dev_answer_filepath,
                                dev_data_orin, dev_preds)

#                        output_filepath_gold = test_writer_for_perl_evaluation(
#                            dev_data_orin, dev_preds, relation_dict, config.dev_answer_filepath
#                        )
#                        command = (
#                            "perl eval/semeval2010_task8_scorer-v1.2.pl {0} {1} > {2}"
#                            "".format(
#                                config.dev_answer_filepath, output_filepath_gold, config.result_filepath
#                            )
#                        )
#
#                        os.system(command)
#                        macro_f1 = read_macro_f1_from_result_file(config.result_filepath)
                        if config.early_stop is True:
                            early_stop_acc, early_stop_preds = run_epoch(
                                    session, m_eval, early_stop_iter, epoch, verbose=False, is_training=False
                            )
                            early_stop_result_filepath = os.path.join(config.output_folder,
                                    "result-earlystop.txt")
                            early_stop_answer_filepath = os.path.join(config.output_folder,
                            "answers_for_early_stop.txt")
                            macro_f1_early_stop = evaluate(early_stop_result_filepath,
                                    early_stop_answer_filepath, early_stop_data_orin, early_stop_preds)

                        if config.cross_validate is False:
                            print('macro_f1 dev: {0}'.format(macro_f1_dev))
                            print('{0},{1:.2f},{2:.2f},{3}'.format(epoch + 1, train_acc*100, dev_acc*100,
                                macro_f1_dev))

                        if config.early_stop is True:
                            patience_counter += 1
                            if macro_f1_early_stop > best_early_stop_macro_f1:
                                best_early_stop_macro_f1 = macro_f1_early_stop
                                best_early_stop_macro_f1_epoch_number = epoch
                                patience_counter = 0

                        # Recording epoch information
                        results['epoch'][epoch]['dev'] = {'f1_macro': macro_f1_dev, 'accuracy': dev_acc}
                        results['epoch'][epoch]['train'] = {'accuracy': train_acc}

                        if config.early_stop is True:
                           # for early stop
                            if patience_counter > config.patience:
                                print('Patience exceeded: early stop')
                                results['execution_details']['early_stop'] = True
                                results['epoch'][epoch]['early_stop'] = {'f1_macro': macro_f1_early_stop,
                                        'accuracy': early_stop_acc}
                                results['epoch'][epoch]['dev'] = {'f1_macro': macro_f1_dev, 'accuracy':
                                        dev_acc}
                                results['epoch'][epoch]['train'] = {'accuracy': train_acc}
                                config.macro_f1_folds.append(macro_f1_dev)

                                if config.cross_validate is True:
                                    print('Last epoch macro_f1 dev: {0}'.format(macro_f1_dev))
                                    print('{0},{1:.2f},{2:.2f},{3}'.format(epoch+1, train_acc*100,
                                        dev_acc*100, macro_f1_dev))
                                break

                        if epoch == config.num_epoches - 1:
                            config.macro_f1_folds.append(macro_f1_dev)
                            if config.cross_validate is True:
                                print('Last epoch macro_f1 dev: {0}'.format(macro_f1_dev))
                                print(
                                    '{0},{1:.2f},{2:.2f},{3}'.format(
                                        epoch+1, train_acc*100, dev_acc*100, macro_f1_dev
                                    ),
                                )

                except KeyboardInterrupt:
                    results['execution_details']['keyboard_interrupt'] = True
                if config.save_path:
                    sv.saver.save(session, config.save_path, global_step=sv.global_step)

        train_end_time = time.time()
        results['execution_details']['num_epochs'] = epoch
        results['execution_details']['train_duration'] = train_end_time - results['execution_details']['train_start']
        results['execution_details']['train_end'] = train_end_time

        json.dump(results, open(os.path.join(config.output_folder, 'results.json'), 'w'), indent = 4, sort_keys=True)
        # return best_acc


if __name__ == '__main__':

        assert len(config.lr_boundaries) == len(config.lr_values) - 1

        # create the necessary output folders
        create_folder_if_not_exists('output/' + config.dataset + '/')
        config.output_dir = 'output/' + config.dataset + '/'
        config.id = uuid.uuid4()
        date = get_current_date() # this is to get the date when the experiment was started,
        # not necessarily when the training started

        # see https://stackoverflow.com/questions/34344836/will-hashtime-time-always-be-unique

        # config.tensorboard_folder = 'output/' + config.dataset + '/Tensorboard/'
        # config.result_folder = 'output/' + config.dataset + '/Result/'
        print("Cross validate is ", config.cross_validate)


        if config.cross_validate is True:
            num_folds = 10 # this value will need to be changed depending on the dataset
            for config.fold in range(0, num_folds):
                print('Fold {} Starting!'.format(config.fold))
                main()
            mean_macro_f1 = np.mean(config.macro_f1_folds)
            std_macro_f1 = np.std(config.macro_f1_folds)
            print("All macro F1 scores", config.macro_f1_folds)
            print("Cross validated F1 scores: %.2f +- %.2f"%(mean_macro_f1, std_macro_f1))

            # code to dump the data
            result = {}
            parameters, _ = get_results_dict(config, 0) # we don't care about second val and we also don't care about individual training time here
            parameters['train_start_folds'] = config.train_start_folds
            result['model_options'] = copy.copy(parameters)
            result['macro_f1_folds'] = config.macro_f1_folds
            result['mean_macro_f1'] = mean_macro_f1
            result['std_macro_f1'] = std_macro_f1
            json.dump(result, open(os.path.join(config.result_folder, 'result.json'), 'w'), indent = 4,
                    sort_keys=True)
            config.final_result_folder = os.path.join(config.output_dir, 'Final_Result')
            final_result_path = os.path.join(config.final_result_folder, 'final_result.csv')
            # need to change the format in which this is written

            # 1 column for fold #, dictionary with all the hyperparam details, and the last column for the result of the fold.
            if(os.path.exists(final_result_path)):
                result_dataframe = pd.read_csv(final_result_path)
                if 'Unnamed: 0' in result_dataframe.columns:
                    result_dataframe.drop('Unnamed: 0', axis=1, inplace=True)
            else:
                # need to change below to Fold, parameters, macro_f1
                result_dataframe = pd.DataFrame(
                    columns = [
                        'Fold Number', 'Parameters', 'Macro F1', 'Train Start Time', 'Hyperparam Tuning Mode',
                        'ID', 'Date of Starting Command'
                    ], # will need to handle date of starting command differently for hyperparam tuning
                )
            start_index = len(result_dataframe.index)
            curr_fold = 0
            # lr_string = str(config.lr_values) + " " + str(config.lr_boundaries)
            # if config.sgd_momentum is True:
            #     lr_string = lr_string + " " + "momentum: " + str(config.momentum)
            params_to_exclude = ['train_start_folds']
            parms_to_add_to_df = {key: parameters[key] for key in parameters if key not in params_to_exclude}
            for i in range(start_index, start_index + num_folds):
                result_dataframe.loc[i] = [
                    curr_fold, str(parms_to_add_to_df), config.macro_f1_folds[curr_fold],
                    config.train_start_folds[curr_fold], config.hyperparam_tuning_mode, config.id, date
                ]
                curr_fold += 1
            result_dataframe.to_csv(final_result_path, index=False)

        else:
            ensemble_num = 1
            for ii in range(ensemble_num):
                    main()
