from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import logging
import os
import sys
sys.path.append('..')
import time
import random
import uuid # for generating a unique id for the cnn
import pandas as pd 

import relation_extraction.data.utils as data_utils
from relation_extraction.data.converters.converter_ddi import relation_dict as ddi_relation_dict
from relation_extraction.data.converters.converter_i2b2 import relation_dict as i2b2_relation_dict
from relation_extraction.data.converters.converter_semeval2010 import relation_dict as semeval_relation_dict
from relation_extraction.models import model_utils
import main_utils
#import argparse
from relation_extraction.models.model import CRCNN as Model
import parser

import copy
import json

from sklearn.metrics import f1_score

log = logging.getLogger()
#parser = argparse.ArgumentParser()

config = parser.get_config()

if not config.preprocessing_type in ['original', 'entity_blinding', 'punct_digit', 'punct_stop_digit',
    'ner_blinding']:
    raise NotImplementedError('Preprocessing types can only be original, entity_blinding, punct_digit, punct_stop_digit or ner_blinding')
if not config.dataset in ['ddi', 'semeval2010', 'i2b2']:
    raise NotImplementedError('Datasets currently supported are ddi, semeval 2010 or i2b2')


post = '_' + config.preprocessing_type
config.data_root = "/data/medg/misc/geeticka/relation_extraction/" + config.dataset + \
        "/pre-processed/" + config.preprocessing_type + "/"

# eval_metric refers to macro_f1 or micro_f1: it is a more general name to store the metric values
if config.dataset == 'semeval2010':
    relation_dict = semeval_relation_dict
    config.classnum = max(relation_dict.keys()) # we are not considering the "other" class.
    folds = 10
    evaluation_metric_print = 'macro_f1'
    accumulated_metrics_print = '<macro_f1>'
    #TODO (geeticka): remove all arguments from config that are not passed in, for example folds and macro_f1_folds etc
elif config.dataset == 'ddi':
    relation_dict = ddi_relation_dict
    config.classnum = max(relation_dict.keys()) # 4 classes are being predicted
    config.embedding_file = '/data/medg/misc/geeticka/relation_extraction/biomed-embed/wikipedia-pubmed-and-PMC-w2v.txt'
    folds = 5
    evaluation_metric_print = 'macro_f1'
    accumulated_metrics_print = '<macro_f1: (5way with none, 5 way without none, 2 way)>'
elif config.dataset == 'i2b2':
    relation_dict = i2b2_relation_dict
    config.classnum = max(relation_dict.keys()) + 1 # we do not have an 'other' class here
    config.embedding_file = '/data/medg/misc/geeticka/relation_extraction/biomed-embed/wikipedia-pubmed-and-PMC-w2v.txt'
    #TODO: insert folds information; for now just have dummy folds
    folds = 5
    evaluation_metric_print = 'micro_f1'
    accumulated_metrics_print = '<micro_f1: (8way, 2way, Prob-Treat, Prob-Test, Prob-Prob) all without None>'


config.train_text_dataset_path = 'train{post}.txt'.format(post=post)
config.test_text_dataset_path = 'test{post}.txt'.format(post=post)
def res(path): return os.path.join(config.data_root, path)

if config.use_elmo is True: 
    prefix = 'elmo/'
    mode = 'elmo'
else: 
    prefix = ''
    mode = 'normal'

dataset = \
data_utils.Dataset(res(prefix+'pickled-files/seed_{K}_{folds}-fold-border_{N}{post}.pkl').format(K=config.pickle_seed,
    N=config.border_size, folds=folds, post=post))
print("pickled files:", res(prefix+'pickled-files/seed_{K}_{folds}-fold-border_{N}{post}.pkl').format(K=config.pickle_seed,
    N=config.border_size, folds=folds, post=post))

date_of_experiment_start = None

def init():

    #
    # Config log
    if config.log_file is None:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    else:
        main_utils.create_folder_if_not_exists(config.save_path)

        logging.basicConfig(
            filename=config.log_file, filemode='a', level=logging.DEBUG, format='%(asctime)s %(message)s',
            datefmt='%m-%d %H:%M'
        )

    if config.cross_validate is True and config.fold is None or config.fold is not None and \
            config.cross_validate is False:
        raise NotImplementedError("If cross validate is true, fold has to not be None")
    
    # TODO: insert checks for config.cross validate etc
    ###
    ### Getting the train, dev/ early stop data in the splitted format needed before vectorization
    ###
    if config.early_stop is True:
        train_data, dev_data, early_stop_data, config.train_text_dataset_file, config.test_text_dataset_file =\
                main_utils.get_data(res, dataset, config.cross_validate, config.train_text_dataset_path, config.test_text_dataset_path,
                        config.fold, config.use_test, config.early_stop, config.early_stop_size,
                        config.border_size, mode=mode)
        data_size = {'train': len(train_data[0]), 'dev': len(dev_data[0]), 'early_stop':
                len(early_stop_data[0])}
        data = {'train': train_data, 'dev': dev_data, 'early_stop': early_stop_data}

    else:
        train_data, dev_data, config.train_text_dataset_file, config.test_text_dataset_file = \
                main_utils.get_data(res, dataset, config.cross_validate, config.train_text_dataset_path, config.test_text_dataset_path,
                        config.fold, config.use_test, config.early_stop, config.early_stop_size,
                        config.border_size, mode=mode)
        data_size = {'train': len(train_data[0]), 'dev': len(dev_data[0])}
        data = {'train': train_data, 'dev': dev_data}

    ### Build word dictionary
    word_dict = main_utils.get_word_dict(data, config.low_freq_thresh, config.early_stop)
    
    ### Log information
    data_size['num_words'] = len(word_dict)
    main_utils.log_info(log, data_size, config)
   
    ### Load Embeddings
    if config.dataset == 'semeval2010':
        embeddings = data_utils.load_embedding_senna(config, word_dict)
    elif config.dataset == 'ddi' or config.dataset == 'i2b2':
        embeddings = data_utils.load_embedding(config, word_dict)

    ### Get maximum sentence and entity lengths
    config.max_len, config.max_e1_len, config.max_e2_len = \
            main_utils.get_maximum_entity_and_sentence_length(data, config.early_stop)

    ### vectorize the data
    train_vec = data_utils.vectorize(config, train_data, word_dict)
    dev_vec = data_utils.vectorize(config, dev_data, word_dict)
    if config.early_stop is True:
        early_stop_vec = data_utils.vectorize(config, early_stop_data, word_dict)

    # finally I need the original entity info in the test file
    # previously there was [] as the input
    dev_data_orin = dev_data[1]
    if config.early_stop is True:
        print("returning the early stop data")
        early_stop_data_orin = early_stop_data[1]
        return embeddings, train_vec, dev_vec, dev_data_orin, early_stop_vec, early_stop_data_orin

    return embeddings, train_vec, dev_vec, dev_data_orin



def main(date_of_experiment_start):

    if config.early_stop is True:
        embeddings, train_vec, dev_vec, dev_data_orin, early_stop_vec, \
        early_stop_data_orin = init()
    else:
        embeddings, train_vec, dev_vec, dev_data_orin = init()
    # embeddings, train_vec, test_vec, test_relations = init()
    bz = config.batch_size

    results, parameters, model_name = main_utils.output_folder_creation(config, date_of_experiment_start)
    # above method is general and so below I am adding stff specific to folds
    parameters['fold'] = config.fold
    results['fold'] = config.fold
    results['epoch'] = {}

    # below is code for running the model itself
    with tf.Graph().as_default():
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None):
                m_train = Model(config, embeddings, is_training=True)
                m_train.__run__()

        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=True):
                m_eval = Model(config, embeddings, is_training=False)
                m_eval.__run__()

        # Start TensorFlow session
        sv = tf.train.Supervisor(logdir=config.save_path, global_step=m_train.global_step)

        print("Output folder is the following", config.output_folder)

        configProto = tf.ConfigProto()
        configProto.gpu_options.allow_growth = True

        with sv.managed_session(config=configProto) as session:
            print('Format of evaluation printing is as follows')
            dev_or_test = 'dev' if config.use_test is False else 'test'
            print('{metric} {eval_data}: {metric_val}'.format(metric=evaluation_metric_print, 
                eval_data=dev_or_test, metric_val=accumulated_metrics_print))
            print(
                '<epoch>,<train accuracy>,<dev accuracy>,{}'.format(evaluation_metric_print)
            )
            if config.early_stop is True:
                best_early_stop_eval_metric = 0
                best_early_stop_eval_metric_epoch_number = -1
                patience_counter = 0
            try:
                for epoch in range(config.num_epoches):
                    results['epoch'][epoch] = {}
                    train_iter = data_utils.batch_iter(config.seed, main_utils.stack_data(train_vec), bz, shuffle=True)
                    dev_iter   = data_utils.batch_iter(config.seed, main_utils.stack_data(dev_vec),   bz, shuffle=False)
                    if config.early_stop is True:
                        early_stop_iter = data_utils.batch_iter(config.seed, main_utils.stack_data(early_stop_vec), bz, shuffle=False)
                    train_verbosity = True if config.cross_validate is False else False
                    train_acc, _ = model_utils.run_epoch(session, m_train, train_iter, epoch,
                            config.batch_size, config.dataset, config.classnum, verbose=False, mode=mode)

                    # TODO(geeticka): Why separate model (e.g. why m_eval vs. m_train)?
                    dev_acc, dev_preds = model_utils.run_epoch(
                        session, m_eval, dev_iter, epoch, config.batch_size, config.dataset,
                        config.classnum, verbose=False, is_training=False, mode=mode
                    )

                    config.result_filepath = os.path.join(config.output_folder, config.result_file)
                    config.dev_answer_filepath = os.path.join(
                        config.output_folder, config.dev_answer_file
                    )

                    eval_metric_dev = main_utils.evaluate(config.result_filepath, config.dev_answer_filepath,
                            relation_dict, dev_data_orin, dev_preds, config.dataset)

                    if config.early_stop is True:
                        early_stop_acc, early_stop_preds = model_utils.run_epoch(
                                session, m_eval, early_stop_iter, epoch, config.batch_size,
                                config.dataset, config.classnum, verbose=False, is_training=False
                        )
                        early_stop_result_filepath = os.path.join(config.output_folder,
                                "result-earlystop.txt")
                        early_stop_answer_filepath = os.path.join(config.output_folder,
                        "answers_for_early_stop.txt")
                        eval_metric_early_stop = main_utils.evaluate(early_stop_result_filepath,
                                early_stop_answer_filepath, relation_dict, early_stop_data_orin, 
                                early_stop_preds, config.dataset)
                        if config.dataset == 'ddi' or config.dataset == 'i2b2': # because eval is a tuple
                            eval_metric_early_stop = eval_metric_early_stop[0]

                    if config.cross_validate is False:
                        print('{metric} {eval_data}: {metric_val}'.format(metric=evaluation_metric_print, 
                            eval_data=dev_or_test, metric_val=eval_metric_dev))
                        print('{0},{1:.2f},{2:.2f},{3}'.format(epoch + 1, train_acc*100, dev_acc*100,
                            eval_metric_dev))

                    if config.early_stop is True:
                        patience_counter += 1
                        if eval_metric_early_stop > best_early_stop_eval_metric:
                            best_early_stop_eval_metric = eval_metric_early_stop
                            best_early_stop_eval_metric_epoch_number = epoch
                            patience_counter = 0

                    # Recording epoch information
                    results['epoch'][epoch][dev_or_test] = {evaluation_metric_print: eval_metric_dev, 'accuracy': dev_acc}
                    results['epoch'][epoch]['train'] = {'accuracy': train_acc}

                    if config.early_stop is True:
                       # for early stop
                        if patience_counter > config.patience:
                            print('Patience exceeded: early stop')
                            results['execution_details']['early_stop'] = True
                            results['epoch'][epoch]['early_stop'] = {evaluation_metric_print: eval_metric_early_stop,
                                    'accuracy': early_stop_acc}
                            results['epoch'][epoch][dev_or_test] = {evaluation_metric_print: eval_metric_dev, 'accuracy':
                                    dev_acc}
                            results['epoch'][epoch]['train'] = {'accuracy': train_acc}
                            config.eval_metric_folds.append(eval_metric_dev)

                            if config.cross_validate is True:
                                print('Last epoch {metric} {eval_data}: {metric_val}'.format(metric=evaluation_metric_print,
                                            eval_data=dev_or_test, metric_val=eval_metric_dev))
                                print('{0},{1:.2f},{2:.2f},{3}'.format(epoch+1, train_acc*100,
                                    dev_acc*100, eval_metric_dev))
                            break

                    if epoch == config.num_epoches - 1:
                        config.eval_metric_folds.append(eval_metric_dev)
                        if config.cross_validate is True:
                            print('Last epoch {metric} {eval_data}: {metric_val}'.format(metric=evaluation_metric_print, 
                                eval_data=dev_or_test, metric_val=eval_metric_dev))
                            print(
                                '{0},{1:.2f},{2:.2f},{3}'.format(
                                    epoch+1, train_acc*100, dev_acc*100, eval_metric_dev
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

    main_utils.perform_assertions(config)

    # create the necessary output folders
    config.output_dir = '/scratch/geeticka/relation-extraction/output/' + config.dataset + '/'
    main_utils.create_folder_if_not_exists(config.output_dir)
    config.id = str(uuid.uuid4())
    date = main_utils.get_current_date() # this is to get the date when the experiment was started,
    date_of_experiment_start = date
    start_time = time.time()
    # not necessarily when the training started

    # see https://stackoverflow.com/questions/34344836/will-hashtime-time-always-be-unique

    print("Cross validate is ", config.cross_validate)


    if config.cross_validate is True:
        num_folds = folds
        for config.fold in range(0, num_folds):
            start_time = time.time()
            print('Fold {} Starting!'.format(config.fold))
            main(date_of_experiment_start)
            end_time = time.time()
            execution_time = (end_time - start_time)/3600.0 #in hours
            execution_time = round(execution_time, 2)
            config.execution_time_folds.append(execution_time)

        #TODO: handling of below can be better: just make it automatic rather than dataset specific
        if config.dataset == 'ddi':
            eval_metric_type_by_fold = [x for x in zip(*config.eval_metric_folds)] # 3 types of macro F1 X folds
            mean_eval_metric = [np.mean(x) for x in eval_metric_type_by_fold]
            std_eval_metric = [np.std(x) for x in eval_metric_type_by_fold]
            print("Cross validated scores: %.2f +- %.2f %.2f +- %.2f %.2f +- %.2f"%(mean_eval_metric[0], 
                std_eval_metric[0], mean_eval_metric[1], std_eval_metric[1], mean_eval_metric[2], std_eval_metric[2]))
        elif config.dataset == 'semeval2010':
            mean_eval_metric = np.mean(config.eval_metric_folds)
            std_eval_metric = np.std(config.eval_metric_folds)
            print("Cross validated scores: %.2f +- %.2f"%(mean_eval_metric, std_eval_metric))
        elif config.dataset == 'i2b2':
            eval_metric_type_by_fold = [x for x in zip(*config.eval_metric_folds)]
            mean_eval_metric = [np.mean(x) for x in eval_metric_type_by_fold]
            std_eval_metric = [np.std(x) for x in eval_metric_type_by_fold]
            print("Cross validated scores: %.2f +- %.2f %.2f +- %.2f %.2f +- %.2f %.2f +- %.2f"%(mean_eval_metric[0], 
                std_eval_metric[0], mean_eval_metric[1], std_eval_metric[1], mean_eval_metric[2], std_eval_metric[2], 
                mean_eval_metric[3], std_eval_metric[3]))

        ### printing, creation of json file
        print("All {} scores {}".format(evaluation_metric_print, config.eval_metric_folds))
        print("ID of the model is", config.id)
        total_execution_time = sum(config.execution_time_folds)
        print("Execution Time (hr)", total_execution_time)
        # code to dump the data
        result = {}
        parameters, _ = parser.get_results_dict(config, 0) # we don't care about second val and we also don't care about individual training time here
        parameters['train_start_folds'] = config.train_start_folds
        result['model_options'] = copy.copy(parameters)
        result['macro_f1_folds'] = config.eval_metric_folds
        result['mean_macro_f1'] = mean_eval_metric
        result['std_macro_f1'] = std_eval_metric
        json.dump(result, open(os.path.join(config.result_folder, 'result.json'), 'w'), indent = 4,
                sort_keys=True)
        
        ### Dumping the CSV file
        main_utils.dump_csv(config, num_folds, '_crossval')
    else:
        ensemble_num = 1
        for ii in range(ensemble_num):
                main(date_of_experiment_start)
        end_time = time.time()
        execution_time = (end_time - start_time)/3600.0
        print("ID of the model is", config.id)
        print("Execution time (in hr): ", execution_time)
        main_utils.dump_csv(config, 1, '_nocrossval')
