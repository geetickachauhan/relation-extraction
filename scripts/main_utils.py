'''
Author: Geeticka Chauhan
This is a utility file that is used by the main script. These are not data handling utilities
'''

import os
import sys
sys.path.append('..')
import time
import datetime
import parser
import main_utils
import random 
import relation_extraction.data.utils as data_utils
from relation_extraction.hyperparam_tuning.distributions import CNN_dist
import pandas as pd

TRAIN, DEV, TEST = 0, 1, 2

def set_hyperparams(config):
    hyperparams = CNN_dist._sample()
    config.num_epoches = int(hyperparams['num_epoches'])
    config.filter_sizes = hyperparams['filter_size']
    config.batch_size = int(hyperparams['batch_size'])
    config.early_stop = hyperparams['early_stop']
    learning_rate_init = round(hyperparams['learning_rate_init'], 5)
    if config.early_stop is True:
        config.patience = int(config.num_epoches/5)

    if hyperparams['learning_rate'] == 'constant':
        config.lr_values = [float(learning_rate_init), float(learning_rate_init)]
        config.lr_boundaries = [int(config.num_epoches/2)]
    else: # this is going to be complicated
        config.lr_values = [float(0.001), float(learning_rate_init)]
        config.lr_boundaries = [int(config.num_epoches/2)]

# Dump the CSV file, postfix is to specify whether this is cross val, on test data or just a dev set
def dump_csv(config, parameters, num_folds, date, evaluation_metric_print, postfix=''):
    config.final_result_folder = os.path.join(config.output_dir, 'Final_Result')
    create_folder_if_not_exists(config.final_result_folder)
    final_result_path = os.path.join(config.final_result_folder, 'final_result'+ postfix + '.csv')
    if config.hyperparam_tuning_mode is True:
        final_result_path = os.path.join(config.final_result_folder, 'final_result_hyperparam'+ postfix + '.csv')
    # need to change the format in which this is written
    # 1 column for fold #, dictionary with all the hyperparam details, and the last column for the result of the fold.
    if(os.path.exists(final_result_path)):
        result_dataframe = pd.read_csv(final_result_path)
        if 'Unnamed: 0' in result_dataframe.columns:
            result_dataframe.drop('Unnamed: 0', axis=1, inplace=True)
    else:
        eval_column = get_eval_column(evaluation_metric_print)
        # need to change below to Fold, parameters, eval_metric
        columns = [
            'Fold Number', eval_column, 'Parameters', 'Train Start Time', 'Hyperparam Tuning Mode',
            'ID', 'Date of Starting Command', 'Execution Time (hr)']
        if num_folds == 1: # means we are not doing cross val
            columns.remove('Fold Number')
        result_dataframe = pd.DataFrame(
            columns=columns, # will need to handle date of starting command differently for hyperparam tuning
        )
    start_index = len(result_dataframe.index)
    curr_fold = 0
    params_to_exclude = ['train_start_folds']
    parms_to_add_to_df = {key: parameters[key] for key in parameters if key not in params_to_exclude}
    for i in range(start_index, start_index + num_folds):
        main_cols = [
            config.eval_metric_folds[curr_fold], str(parms_to_add_to_df),
            config.train_start_folds[curr_fold], config.hyperparam_tuning_mode, config.id, date,
            config.execution_time_folds[curr_fold]
        ]
        if num_folds == 1: tmp = main_cols
        else:
            tmp = [curr_fold]
            tmp.extend(main_cols)
        result_dataframe.loc[i] = tmp
        curr_fold += 1
    result_dataframe.to_csv(final_result_path, index=False)

def perform_assertions(config):
    if config.hyperparam_tuning_mode is True and config.random_search is True:
        if config.cross_validate is True or config.cross_validate_report is True: raise Exception("Random Search is only supported with non cross val")
        if config.use_test is True: raise Exception("You cannot use test set when performing hyperparam tuning")
    assert len(config.lr_boundaries) == len(config.lr_values) - 1
    if config.use_bert_CLS is True and config.use_bert_tokens is True:
        raise Exception('Current version of the code only supports either bert-CLS or bert-tokens mode, not both')
    if config.use_elmo is True and (config.use_bert_CLS is True or config.use_bert_tokens is True):
        raise Exception("Current version of the code does not support using elmo and bert embeddings at the same time")
    if config.dataset != 'semeval2010' and config.border_size != -1:
        raise Exception("Current implementation only supports border size -1 for non semeval datasets")
    if config.use_test is True and (config.cross_validate is True or config.cross_validate_report is True):
        raise NotImplementedError("You cannot use test data and perform cross validation at the same time")
    if config.cross_validate is True and config.cross_validate_report is True:
        raise NotImplementedError("You cannot perform regular cross val with cross val at the reporting mode at the same time")
    if config.use_elmo is True and config.preprocessing_type != 'original' and config.dataset != 'i2b2' and \
            config.dataset != 'ddi':
        raise NotImplementedError("You can only use the elmo embeddings with the original preprocessing")

def get_maximum_entity_and_sentence_length(data, early_stop):

    max_len = max_length_all_data(data['train'][0], data['dev'][0], 'sentence')
    max_e1_len = max_length_all_data(data['train'][2], data['dev'][2], 'entity')
    max_e2_len = max_length_all_data(data['train'][3], data['dev'][3], 'entity')

    if early_stop is True:
        max_len_earlystop = max_sent_len(data['early_stop'][0])
        max_e1_len_earlystop = max_ent_len(data['early_stop'][2])
        max_e2_len_earlystop = max_ent_len(data['early_stop'][3])
        max_len = max(max_len, max_len_earlystop)
        max_e1_len = max(max_e1_len, max_e1_len_earlystop)
        max_e2_len = max(max_e2_len, max_e2_len_earlystop)
    return max_len, max_e1_len, max_e2_len

def get_word_dict(data, low_freq_thresh, early_stop):
    # Build vocab, pretend that your test set does not exist because when you need to use test 
    # set, you can just make sure that what we report on (i.e. dev set here) is actually the test data
    all_data = data['train'][0] + data['dev'][0]

    if early_stop is True:
        all_data = all_data + data['early_stop'][0]
    word_dict = data_utils.build_dict(all_data, low_freq_thresh)
    return word_dict

def log_info(log, data_size, config):

    log.info('ID of the model is %s' % config.id)
    log.info('size of train data: %d' % data_size['train'])
    log.info('size of dev data: %d' % data_size['dev'])
    if config.early_stop is True:
        log.info('size of early stop data: %d' % data_size['early_stop'])
    print("early stop is", config.early_stop)
    print("lr_values and boundaries are", config.lr_values, config.lr_boundaries)
    print("seed for random initialization is ",  config.seed)
    print('use_test is', config.use_test)
    print('use_elmo is', config.use_elmo)
    print("use_bert_CLS is ", config.use_bert_CLS)
    print("use_bert_tokens is", config.use_bert_tokens)
    log.info('total words: %d' % data_size['num_words'])

# get the train and dev data , as well as early stop data in the correct form
def get_data(res, dataset, config, mode='normal'):
    def transpose(data):
        # convert the list of tuples (outer dimension num of sentences) to tuple of lists (outer dim features)
        transposed_data = []
        for x in zip(*data):
            transposed_data.append(list(x))
        return tuple(transposed_data)

    # setting the train, dev data; pretend that test data does not exist
    train_text_dataset_file = res(config.train_text_dataset_path)
    test_text_dataset_file = res(config.test_text_dataset_path)
    if config.cross_validate is False and config.cross_validate_report is False:
        train_data = main_utils.openFileAsList(train_text_dataset_file)
    elif config.cross_validate is True:
        train_data = dataset.get_data_for_fold(config.fold, mode=mode)
        dev_data = dataset.get_data_for_fold(config.fold, DEV, mode=mode)
    elif config.cross_validate_report is True:
        train_data = dataset.get_train_dev_data_for_fold(config.fold, mode=mode)
        dev_data = dataset.get_data_for_fold(config.fold, TEST, mode=mode)

    # now each of the above data contains the following in order:
    # sentences, relations, e1_pos, e2_pos

    # if you are using the pickle file with unsplit sentences, you will do the following:
    # random select dev set when there is no cross validation
    if config.cross_validate is False and config.cross_validate_report is False:
        # split data
        train_data = preprocess_data_noncrossvalidated(train_data, config.border_size)
        if mode == 'elmo':
            train_elmo = data_utils.get_elmo_embeddings(res('elmo/train_' + config.preprocessing_type +'_border_' + str(config.border_size) + '.hdf5'))
            train_data = train_data + train_elmo
        elif mode == 'bert-tokens' or mode == 'bert-CLS':
            if mode == 'bert-tokens': bert_function = data_utils.get_bert_token_embeddings
            elif mode == 'bert-CLS': bert_function = data_utils.get_bert_CLS_embeddings
            train_bert = bert_function(res(mode + '/train_' + config.preprocessing_type + '_border_' + str(config.border_size) + '.json'))
            train_data = train_data + train_bert

        if config.use_test is False:
            train_data = list(zip(*train_data)) # need to convert to dimensionality of sentences
            devsize = int(len(train_data)*0.15)
            if config.hyperparam_tuning_mode is True:
                random.seed(config.seed) # set the seed in order to ensure that same dev set sampled when 
            # hyperparam tuning mode is on
            select_index = random.sample(range(0, len(train_data)), devsize)
            dev_data = [train_data[idx] for idx in range(len(train_data)) if idx in select_index]
            tmp_train_data = [train_data[idx] for idx in range(len(train_data)) if idx not in select_index]
            train_data = tmp_train_data
            if config.early_stop is True: # this is assuming that we are not using test data
                early_stop_size = int(len(dev_data)*config.early_stop_size)
                if config.hyperparam_tuning_mode is True:
                    random.seed(config.seed)
                select_index = random.sample(range(0, len(dev_data)), early_stop_size)
                early_stop_data = [dev_data[idx] for idx in range(len(dev_data)) if idx in select_index]
                tmp_dev_data = [dev_data[idx] for idx in range(len(dev_data)) if idx not in select_index]
                dev_data = tmp_dev_data
                early_stop_data = transpose(early_stop_data)
            train_data = transpose(train_data)
            dev_data = transpose(dev_data)

        elif config.use_test is True:
            dev_data = open(test_text_dataset_file, 'r') # means we will report test scores
            dev_data = preprocess_data_noncrossvalidated(dev_data, config.border_size)
            if mode == 'elmo':
                test_elmo = data_utils.get_elmo_embeddings(res('elmo/test_' + config.preprocessing_type + '_border_' + str(config.border_size) + '.hdf5'))
                dev_data = dev_data + test_elmo
            if mode == 'bert-CLS' or mode == 'bert-tokens': # CLS is a fixed size sentence representation - no need to pad that 
                if mode == 'bert-tokens': bert_function = data_utils.get_bert_token_embeddings
                elif mode == 'bert-CLS': bert_function = data_utils.get_bert_CLS_embeddings
                test_bert = bert_function(res(mode + '/test_' + config.preprocessing_type + \
                        '_border_' + str(config.border_size) + '.json'))
                dev_data = dev_data + test_bert

            if config.early_stop is True: # be careful to choose early stop set only from the training data
                train_data = list(zip(*train_data))
                early_stop_size = int(len(train_data)* config.early_stop_size)
                select_index = random.sample(range(0, len(train_data)), early_stop_size)
                early_stop_data = [train_data[idx] for idx in range(len(train_data)) if idx in select_index]
                tmp_train_data = [train_data[idx] for idx in range(len(train_data)) if idx not in select_index]
                train_data = tmp_train_data
                early_stop_data = transpose(early_stop_data)
                train_data = transpose(train_data)

    # only need below if doing early stop
    if config.early_stop is True and (config.cross_validate is True or config.cross_validate_report is True):
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

    if config.early_stop is True:
        return train_data, dev_data, early_stop_data, train_text_dataset_file, test_text_dataset_file
    return train_data, dev_data, train_text_dataset_file, test_text_dataset_file


# This method performs the work of creating the necessary output folders for the model
def output_folder_creation(config, date_of_experiment_start):
    # config is pass by reference in this case!
    train_start_time_in_miliseconds = get_current_time_in_miliseconds()
    config.train_start_folds.append(train_start_time_in_miliseconds)
    results, parameters = parser.get_results_dict(config, train_start_time_in_miliseconds)
    hyperparameters = ['dataset', 'pos_embed_size', 'num_filters', 'filter_sizes', 'keep_prob', 'early_stop', 'patience']
    hyperparam_dir_addition = '-'.join(['{}_{:.6f}'.format(parameter, parameters[parameter]) if
            type(parameters[parameter])==float else '{}_{}'.format(parameter,
                parameters[parameter]) for parameter in hyperparameters])

    #if config.fold is not None and config.cross_validate is True:
    #    model_name = 'cnn_{0}'.format(config.id + '_' + train_start_time_in_miliseconds +'-'+ 'Fold-'+
    #        str(config.fold) + hyperparam_dir_addition)
    #else:
    model_name = 'cnn_{0}'.format(config.id + '_' + date_of_experiment_start + hyperparam_dir_addition)
    
    config.parameters = parameters
    if config.cross_validate is True or config.cross_validate_report is True:
        folder_string = "CrossValidation"
        config.output_folder = os.path.join(config.output_dir, "CrossValidation", model_name, 'Fold'+str(config.fold))
    else:
        folder_string = "NoCrossValidation"
        config.output_folder = os.path.join(config.output_dir,"NoCrossValidation", model_name)
    #config.tensorboard_folder = os.path.join(config.output_dir, "Tensorboard", hyperparam_dir_addition)

    config.tensorboard_folder = config.output_folder
    config.result_folder = os.path.join(config.output_dir, folder_string, model_name, 'Result')
    print("Tensorboard folder, for current fold is", config.tensorboard_folder)
    create_folder_if_not_exists(config.output_folder)
    create_folder_if_not_exists(config.tensorboard_folder)
    create_folder_if_not_exists(config.result_folder)
    config.test_answer_filepath = os.path.join(config.output_folder, config.test_answer_file)

    return results, parameters, model_name

# get the evaluation column as Macro F1 from macro_f1
def get_eval_column(evaluation_metric_print):
    eval_column = evaluation_metric_print.split('_')
    eval_column[0] = eval_column[0].capitalize()
    eval_column[1] = eval_column[1].capitalize()
    eval_column = " ".join(eval_column)
    return eval_column

# given entity positions, find max length
def max_ent_len(data):
    return max(map(lambda x: x[1]-x[0]+1, data))

# given the data, find the maximum length
def max_sent_len(data): return len(max(data, key=lambda x:len(x)))

# whether to retun the max length for sentence or entity in the train, dev data
def max_length_all_data(train, dev, kind='sentence'):
    function = max_sent_len if kind == 'sentence' else max_ent_len
    max_train = function(train)
    max_dev = function(dev)
    return max(max_train, max_dev)

# a function to open the file and return a list consisting of the lines in the file
# this is needed to create the dev file from the train file, otherwise a TypeError is thrown because
# when a file is originally opened, it is of type '_io.TextIOWrapper'
def openFileAsList(filename):
    with open(filename) as f:
        mylist = [line.rstrip('\n') for line in f]
    return mylist

#split the data, and get the dependency information for the non cross validated data
def preprocess_data_noncrossvalidated(data, border_size):
    data = data_utils.split_data_cut_sentence(data, border_size)
    return data


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


# Following methods are specific to the semeval 2010 dataset
def test_writer(scores, answer_path):
        np.save(answer_path, scores)
        answer_txt_path = answer_path.replace('npy', 'txt')
        # scores = scores.tolist()
        with open(answer_txt_path, 'w') as inFile:
                for score in scores:
                        inFile.write('\t'.join([str(item) for item in score])+'\n')

        if config.cross_validate is False and config.cross_validate is False:
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

def read_macro_f1_from_result_file_semeval(result_filepath):
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

def read_macro_f1_from_result_file_ddi(result_filepath):
        '''
        Retrieve the macro F1 score from the result file that perl eval/ddi_task9.2_scorer.pl generates
        '''
        result_file = open(result_filepath, 'r')
        for cur_line in result_file:
            if cur_line.startswith('<<< The 5-way evaluation with None:'):
                        cur_line = cur_line.replace('<<< The 5-way evaluation with None: macro-averaged F1 = ','')
                        cur_line = cur_line.replace('% >>>','')
                        macro_f1_5way_with_none = float(cur_line)
            if cur_line.startswith('<<< The 5-way evaluation without None:'):
                        cur_line = cur_line.replace('<<< The 5-way evaluation without None: macro-averaged F1 = ','')
                        cur_line = cur_line.replace('% >>>','')
                        macro_f1_5way_without_none = float(cur_line)
            if cur_line.startswith('<<< The 2-way evaluation (just detection of relation):'):
                        cur_line = cur_line.replace('<<< The 2-way evaluation (just detection of relation): macro-averaged F1 = ','')
                        cur_line = cur_line.replace('% >>>','')
                        macro_f1_2way = float(cur_line)
        result_file.close()
        return macro_f1_5way_with_none, macro_f1_5way_without_none, macro_f1_2way

def read_micro_f1_from_result_file_i2b2(result_filepath):
        '''
        Retrieve the macro F1 score from the result file that perl eval/i2b2_relations_scorer.pl generates
        '''
        result_file = open(result_filepath, 'r')
        for cur_line in result_file:
            if cur_line.startswith('<<< The 8-way evaluation:'):
                        cur_line = cur_line.replace('<<< The 8-way evaluation: micro-averaged F1 = ','')
                        cur_line = cur_line.replace('% >>>','')
                        micro_f1_8way = float(cur_line)
            if cur_line.startswith('<<< The Problem-Treatment evaluation:'):
                        cur_line = cur_line.replace('<<< The Problem-Treatment evaluation: micro-averaged F1 = ','')
                        cur_line = cur_line.replace('% >>>','')
                        micro_f1_probtreat = float(cur_line)
            if cur_line.startswith('<<< The Problem-Test evaluation:'):
                        cur_line = cur_line.replace('<<< The Problem-Test evaluation: micro-averaged F1 = ','')
                        cur_line = cur_line.replace('% >>>','')
                        micro_f1_probtest = float(cur_line)
            if cur_line.startswith('<<< The Problem-Problem evaluation:'):
                        cur_line = cur_line.replace('<<< The Problem-Problem evaluation: micro-averaged F1 = ','')
                        cur_line = cur_line.replace('% >>>','')
                        micro_f1_probprob = float(cur_line)
            if cur_line.startswith('<<< The 2-way evaluation:'):
                        cur_line = cur_line.replace('<<< The 2-way evaluation: micro-averaged F1 = ','')
                        cur_line = cur_line.replace('% >>>','')
                        micro_f1_2way = float(cur_line)
        result_file.close()
        return micro_f1_8way, micro_f1_2way, micro_f1_probtreat, micro_f1_probtest, micro_f1_probprob

#read the macro F1 from the necessary filepath
def evaluate(result_filepath, answer_filepath, relation_dict, data_orin, preds, dataset):
    output_filepath_gold = test_writer_for_perl_evaluation(
        data_orin, preds, relation_dict, answer_filepath
    )
    if dataset == 'semeval2010':
        eval_script = 'semeval2010_task8_scorer-v1.2.pl'
    elif dataset == 'ddi':
        eval_script = 'ddi_task9.2_scorer.pl'
    elif dataset == 'i2b2':
        eval_script = 'i2b2_relations_scorer.pl'
    command = (
        "perl ../eval/{0} {1} {2} > {3}"
        "".format(
            eval_script, answer_filepath, output_filepath_gold, result_filepath
        )
    )

    os.system(command)
    if dataset == 'semeval2010':
        macro_f1 = read_macro_f1_from_result_file_semeval(result_filepath)
        return macro_f1
    elif dataset == 'ddi':
        macro_f1_5way_with_none, macro_f1_5way_without_none, macro_f1_2way = \
                read_macro_f1_from_result_file_ddi(result_filepath)
        return macro_f1_5way_with_none, macro_f1_5way_without_none, macro_f1_2way
    elif dataset == 'i2b2':
        micro_f1_8way, micro_f1_2way, micro_f1_probtreat, micro_f1_probtest, micro_f1_probprob = \
                read_micro_f1_from_result_file_i2b2(result_filepath)
        return micro_f1_8way, micro_f1_2way, micro_f1_probtreat, micro_f1_probtest, micro_f1_probprob
