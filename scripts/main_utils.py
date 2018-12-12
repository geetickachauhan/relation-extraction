'''
Author: Geeticka Chauhan
This is a utility file that is used by the main script. These are not data handling utilities
'''

import os
import sys
sys.path.append('..')
import time
import datetime
import relation_extraction.data.utils as utils


# given entity positions, find max length
def max_ent_len(data):
    return max(map(lambda x: x[1]-x[0]+1, data))

# given the data, find the maximum length
def max_sent_len(data): return len(max(data, key=lambda x:len(x)))

# whether to retun the max length for sentence or entity in the train, dev and test data
def max_length_all_data(train, dev, test, kind='sentence'):
    function = max_sent_len if kind == 'sentence' else max_ent_len
    max_train = function(train)
    max_dev = function(dev)
    max_test = function(test)
    return max(max_train, max_dev, max_test)

# a function to open the file and return a list consisting of the lines in the file
# this is needed to create the dev file from the train file, otherwise a TypeError is thrown because
# when a file is originally opened, it is of type '_io.TextIOWrapper'
def openFileAsList(filename):
    with open(filename) as f:
        mylist = [line.rstrip('\n') for line in f]
    return mylist

#split the data, and get the dependency information for the non cross validated data
def preprocess_data_noncrossvalidated(data, border_size):
    data = utils.split_data_cut_sentence(data, border_size)
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

#read the macro F1 from the necessary filepath
def evaluate(result_filepath, answer_filepath, relation_dict, data_orin, preds):
    output_filepath_gold = test_writer_for_perl_evaluation(
        data_orin, preds, relation_dict, answer_filepath
    )
    command = (
        "perl ../eval/semeval2010_task8_scorer-v1.2.pl {0} {1} > {2}"
        "".format(
            answer_filepath, output_filepath_gold, result_filepath
        )
    )

    os.system(command)
    macro_f1 = read_macro_f1_from_result_file(result_filepath)
    return macro_f1
