'''
Author: Geeticka Chauhan
Utilities to summarize the outputs produced by the model i.e. the results.txt files spitted out by the 
evaluation scripts. 
'''

import os
from sys import path
import re
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

# res will need to be passed to the last function, as an example:
#output_path = '/scratch/geeticka/relation-extraction/output/semeval2010/CrossValidation'
#def res(path): return os.path.join(output_path, path)

## Below are the methods to gather necessary information from the results file
def read_confusion_matrix_per_line(cur_line):
    if re.search(r'.*\|.*', cur_line): # only get those lines which have a pipe operator
        splitted_line = cur_line.strip().split()
        pipe_seen = 0 # the correct numbers are between two pipes
        confusion_matrix_line = []
        for val in splitted_line:
            if val.startswith('|'):
                pipe_seen += 1
            if pipe_seen == 1 and val.strip() != '|': # keep collecting the values as you are
                val = [x for x in val if x != '|'] # to handle some special cases when the pipe operator 
                # is stuck to the number (happens when the number is too long)
                val = ''.join(val)
                confusion_matrix_line.append(float(val))
        return confusion_matrix_line
    return None

def read_accuracy_per_line(cur_line):
    if cur_line.startswith('Accuracy (calculated'):
        accuracy = re.match(r'.*= (.*)%', cur_line).groups()[0]
        accuracy = float(accuracy)
        return accuracy
    return None

def read_precision_recall_f1(cur_line): # assume that the mode is once we have read 'Results for the individual' 
    match = re.match(r'.*= (.*)%.*= (.*)%.*= (.*)%$', cur_line)
    if match:
        precision, recall, f1 = match.groups()
        return float(precision), float(recall), float(f1)
    else:
        return None
    #if not cur_line.startswith('Micro-averaged result'): # you want to read only up to the point when the relations
    # will need to double check above

# confusion matrix portion refers to which part of the file to read
# for eg, this will be <<< (9+1)-WAY EVALUATION TAKING DIRECTIONALITY INTO ACCOUNT -- OFFICIAL >>> for semeval
def get_file_metrics(num_relations, result_file, confusion_matrix_portion):
    official_portion_file = False
    individual_relations_f1_portion = False
    micro_f1_portion = False
    macro_f1_portion = False
    confusion_matrix_official = [] # stores the official confusion matrix read from the file
    accuracy = None
    metrics_indiv_relations = [] # precision, recall and f1 for each relation
    metrics_micro = [] # excluding the other relation
    metrics_macro = [] # excluding the other relation
    with open(result_file, 'r') as result_file:
        for cur_line in result_file:
            cur_line = cur_line.strip()
            #if cur_line.startswith('<<< (9+1)-WAY EVALUATION TAKING DIRECTIONALITY INTO ACCOUNT -- OFFICIAL >>>'):
            if official_portion_file is True and cur_line.startswith('<<<'):
                break
            if cur_line.startswith(confusion_matrix_portion):
                official_portion_file = True
            if official_portion_file is False:
                continue
            confusion_matrix_line = read_confusion_matrix_per_line(cur_line)
            if confusion_matrix_line is not None: 
                confusion_matrix_official.append(confusion_matrix_line)
            
            acc = read_accuracy_per_line(cur_line)
            if acc is not None: accuracy = acc
            
            # figure out which sub portion of the official portion we are in 
            if cur_line.startswith('Results for the individual relations:'):
                individual_relations_f1_portion = True
            elif cur_line.startswith('Micro-averaged result'):
                micro_f1_portion = True
            elif cur_line.startswith('MACRO-averaged result'):
                macro_f1_portion = True
            
            # populate the precision, recall and f1 for the correct respective lists
            if individual_relations_f1_portion is True and micro_f1_portion is False:
                vals = read_precision_recall_f1(cur_line)
                if vals is not None: metrics_indiv_relations.append([vals[0], vals[1], vals[2]])
            elif micro_f1_portion is True and macro_f1_portion is False:
                vals = read_precision_recall_f1(cur_line)
                if vals is not None: metrics_micro.append([vals[0], vals[1], vals[2]])
            elif macro_f1_portion is True:
                vals = read_precision_recall_f1(cur_line)
                if vals is not None: metrics_macro.append([vals[0], vals[1], vals[2]])
    return confusion_matrix_official, accuracy, metrics_indiv_relations, metrics_micro, metrics_macro

# Generate confusion matrix as a pandas dataframe
def get_confusion_matrix_as_df(confusion_matrix_official, relations_as_short_list):
    index = pd.Index(relations_as_short_list, name='gold labels')
    columns = pd.Index(relations_as_short_list, name='predicted')
    confusion_matrix_df = pd.DataFrame(data=confusion_matrix_official, columns=columns,index=index)
    return confusion_matrix_df

# Give the confusions acorss each relation, with a special interest on other
def generate_confused_with_string(idx, row, relation_full_form_dictionary, full_form=False):
    # index is the current relation that we are considering and row is all the predicted examples
    confused_with_string = ""
    num_of_columns = len(row.index)
    for i in range(0, num_of_columns):
        column_name = row.index[i]
        column_value = int(row.loc[column_name])
        if column_value > 0 and column_name != idx:
            if full_form is True: column_name = relation_full_form_dictionary[column_name]
            confused_with_string += " " + column_name + "(" + str(column_value) + ")"
    return confused_with_string.strip()
    #print(row.data[0])
    #for val in row:
    #    print(val)

def generate_pretty_summary_confusion_matrix(confusion_matrix_df, relation_full_form_dictionary,
        full_form=False):
    data = [] # index will be 0,1,2 and so on, but columns will be
    # Actual label, confused with as a string, correct predictions as a number
    for index, row in confusion_matrix_df.iterrows():
        actual_label = relation_full_form_dictionary[index]
        confused_with = generate_confused_with_string(index, row, relation_full_form_dictionary, full_form) 
        # short form is the default
        correct_predictions = row[index] # eg: gives the column value for C-E for an index C-E
        #if index != '_O': confused_with_other = row['_O'] # this is specific to semeval and will need to be changed
        #else: confused_with_other = None
        data.append([actual_label, confused_with, correct_predictions])
    columns = pd.Index(['Gold Relation', 'Confused With(num_examples)', 'Correct Predictions'], name='summary')
    pretty_summary_confusion_matrix_df = pd.DataFrame(data=data, columns=columns)
    return pretty_summary_confusion_matrix_df

# Give the individual relation metrics as a dataframe
def create_metrics_indiv_relations_df(metrics_indiv_relations, relation_full_form_dictionary, relation_as_short_list):
    index_list = relation_as_short_list
    index_list_verbose = [relation_full_form_dictionary[x] for x in index_list]
    index = pd.Index(index_list_verbose, name='labels')
    columns = pd.Index(['Precision', 'Recall', 'F1'], name='metrics')
    metrics_indiv_relations_df = pd.DataFrame(data=metrics_indiv_relations, columns=columns,index=index)
    return metrics_indiv_relations_df

def create_metrics_macro_micro_df(metrics_macro, metrics_micro):
    data = metrics_macro + metrics_micro
    index = pd.Index(['macro', 'micro'], name='calculation type')
    columns = pd.Index(['Precision', 'Recall', 'F1'], name='metrics')
    metrics_macro_micro = pd.DataFrame(data=data, columns=columns,index=index)
    return metrics_macro_micro

# Finally, create a large summary function
def create_summary(result_file, relation_full_form_dictionary, relation_as_short_list,
        confusion_matrix_portion, full_form=False):
    if not os.path.exists(result_file):
        print("Check your path first!")
        return None
    num_relations = len(relation_as_short_list)
    # get the file metrics
    confusion_matrix_official, accuracy, \
    metrics_indiv_relations, metrics_micro, metrics_macro = get_file_metrics(num_relations, result_file, confusion_matrix_portion)
    # get the confusion matrix dataframe
    confusion_matrix_df = get_confusion_matrix_as_df(confusion_matrix_official, relation_as_short_list)
    # these are the summary information that will need to be returned
    pretty_summary_confusion_matrix_df = \
            generate_pretty_summary_confusion_matrix(confusion_matrix_df, relation_full_form_dictionary, full_form)
    total_correct_predictions = pretty_summary_confusion_matrix_df['Correct Predictions'].sum()
    metrics_indiv_relations_df = create_metrics_indiv_relations_df(metrics_indiv_relations, 
                                                                   relation_full_form_dictionary, 
                                                                   relation_as_short_list)
    metrics_macro_micro = create_metrics_macro_micro_df(metrics_macro, metrics_micro)
    # report accuracy as well
    return confusion_matrix_df, pretty_summary_confusion_matrix_df, total_correct_predictions, metrics_indiv_relations_df, \
    metrics_macro_micro, accuracy

# given the confusion matrix, return the sums of all the examples
def get_sum_confusion_matrix(confusion_matrix):
    sum = 0
    for column in confusion_matrix:
        sum += confusion_matrix[column].sum()
    return sum


# for each of the relations, do a t test between the two model metrics
def indiv_metric_comparison(dataset, metrics_i_model1, metrics_i_model2, model1_name, model2_name, exclude_other=True):
    print("\nTTest from %s to %s"%(model1_name, model2_name))
    if exclude_other is True:
        print("Below is the metric comparsion across the two models" + \
              " considering individual relations, excluding 'Other'")
    else:
        print("Below is the metric comparsion across the two models" + \
              " considering individual relations, including 'Other'")
    for column in metrics_i_model1:
        metric_model1 = metrics_i_model1[column].tolist()
        metric_model2 = metrics_i_model2[column].tolist()
        if exclude_other is True and dataset == 'semeval2010':
            metric_model1 = metric_model1[:-1] # excluding 'Other'
            metric_model2 = metric_model2[:-1]
        if exclude_other is True and dataset == 'i2b2':
            metric_model1 = metric_model1[1:]
            metric_model2 = metric_model2[1:]
        tt = ttest_rel(metric_model1, metric_model2)
        print("Metric: %s \t statistic %.2f \t p_value %s"%
              (column, tt.statistic, tt.pvalue))

def get_macro_micro_metric_comparison(metrics_ma_mi_model1, metrics_ma_mi_model2, model1_name, model2_name):
    print("\nAll the macro and micro metrics are calculated, excluding the 'Other' class")
    print("Macro - Micro for the %s model"%(model1_name))
    for column in metrics_ma_mi_model1:
        macro = metrics_ma_mi_model1[column].loc['macro']
        micro = metrics_ma_mi_model1[column].loc['micro']
        print("Metric: %s \t Macro-Micro %.2f"%(column, macro-micro))
        
    print("\nMacro - Micro for the %s model"%(model2_name))
    for column in metrics_ma_mi_model2:
        macro = metrics_ma_mi_model2[column].loc['macro']
        micro = metrics_ma_mi_model2[column].loc['micro']
        print("Metric: %s \t Macro-Micro %.2f"%(column, macro-micro))
        
    print("\nMacro_%s - Macro_%s"%(model1_name, model2_name))
    for column in metrics_ma_mi_model1:
        macro_model1 = metrics_ma_mi_model1[column].loc['macro']
        macro_model2 = metrics_ma_mi_model2[column].loc['macro']
        print("Metric: %s \t Difference %.2f"%(column, macro_model1-macro_model2))
    
    print("\nMicro_%s - Micro_%s"%(model1_name, model2_name))
    for column in metrics_ma_mi_model1:
        micro_model1 = metrics_ma_mi_model1[column].loc['micro']
        micro_model2 = metrics_ma_mi_model2[column].loc['micro']
        print("Metric: %s \t Difference %.2f"%(column, micro_model1-micro_model2))

def get_accuracy_difference(accuracy_model1, accuracy_model2, model1_name, model2_name):
    print("Accuracy_%s - Accuracy_%s %.2f"%(model1_name, model2_name, accuracy_model1 - accuracy_model2))

# This is the final function to call
# when exclude_other is 0, we print indiv relation metrics without the other class
# when it is 1, we print it with the other class only
# when it is 2, we print both with and without other class
def print_full_summary(dataset, model1_loc, model2_loc, model1_name, model2_name, res,
        relation_full_form_dictionary, relation_as_short_list, confusion_matrix_portion, exclude_other=0, full_form=False):
    model1 = res(model1_loc)
    model2 = res(model2_loc)
    if dataset != 'semeval2010' and dataset != 'i2b2' and dataset != 'ddi':
        raise Exception("At the moment, this summary script only supports semeval 2010, i2b2, and ddi")
    
    cm_model1, summary_cm_model1, correct_pred_model1, metrics_i_model1, \
    metrics_ma_mi_model1, accuracy_model1 \
    = create_summary(model1, relation_full_form_dictionary, relation_as_short_list, confusion_matrix_portion, full_form)
    
    cm_model2, summary_cm_model2, correct_pred_model2, metrics_i_model2, \
    metrics_ma_mi_model2, accuracy_model2 \
    = create_summary(model2, relation_full_form_dictionary, relation_as_short_list, confusion_matrix_portion, full_form)
    
    # T Test of each metrics, across the relations not Other
    if exclude_other == 1:
        indiv_metric_comparison(dataset, metrics_i_model1, metrics_i_model2, model1_name, model2_name, 
                exclude_other=False)
    else:
        indiv_metric_comparison(dataset, metrics_i_model1, metrics_i_model2, model1_name, model2_name)
        if exclude_other == 2:
            indiv_metric_comparison(dataset, metrics_i_model1, metrics_i_model2, model1_name, model2_name,
                    exclude_other=False)
    
    # Get the difference in the macro and micro scores
    get_macro_micro_metric_comparison(metrics_ma_mi_model1, metrics_ma_mi_model2, model1_name, model2_name)
    
    # Print the accuracy difference as well
    get_accuracy_difference(accuracy_model1, accuracy_model2, model1_name, model2_name)
    return summary_cm_model1, summary_cm_model2


