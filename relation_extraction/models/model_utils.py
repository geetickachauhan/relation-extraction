import time
import numpy as np
import tensorflow as tf

# performs the prediction, but makes sure that if all the scores are negative, predict the class "Other"
def prediction(scores, dataset, classnum):
    data_size = scores.shape[0]
    pred = np.zeros(data_size)
    for idx in range(data_size):
        data_line = scores[idx]
        if all(data_line <= 0.): # assigning last class which is none or other
            if dataset == 'semeval2010' or dataset == 'ddi':
                pred[idx] = classnum 
        else: # for the i2b2 data, it will do argmax
            pred[idx] = np.argmax(data_line)

    return pred

# calculates the accuracy since we have a different prediction procedure
def accuracy(preds, labels):
    correct = np.equal(preds, labels).astype(np.int8)
    # correct[labels==18] = 1
    return correct.sum()

def run_epoch(session, model, batch_iter, epoch, batch_size, dataset, classnum, verbose=True, 
        is_training=True, mode='normal'):
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
        if mode == 'elmo': 
            sents, relations, e1, e2, dist1, dist2, elmo_embeddings, position1, position2 = batch
        elif mode == 'bert-CLS' or mode == 'bert-tokens':
            sents, relations, e1, e2, dist1, dist2, bert_embeddings, position1, position2 = batch
        else: 
            sents, relations, e1, e2, dist1, dist2, position1, position2 = batch
        
        sents = np.vstack(sents)
        
        if mode == 'elmo': 
            in_x, in_e1, in_e2, in_dist1, in_dist2, in_y, in_epoch, in_elmo, in_pos1, in_pos2 = model.inputs
            feed_dict = {in_x: sents, in_e1: e1, in_e2: e2, in_dist1: dist1, in_dist2: dist2, \
                    in_y: relations, in_epoch: epoch, in_elmo: elmo_embeddings, in_pos1: position1, \
                    in_pos2: position2}
        elif mode == 'bert-CLS' or mode == 'bert-tokens': 
            in_x, in_e1, in_e2, in_dist1, in_dist2, in_y, in_epoch, in_bert, in_pos1, in_pos2 = model.inputs
            feed_dict = {in_x: sents, in_e1: e1, in_e2: e2, in_dist1: dist1, in_dist2: dist2, \
                    in_y: relations, in_epoch: epoch, in_bert: bert_embeddings, in_pos1: position1, \
                    in_pos2: position2}
        else:
            in_x, in_e1, in_e2, in_dist1, in_dist2, in_y, in_epoch, in_pos1, in_pos2 = model.inputs
            feed_dict = {in_x: sents, in_e1: e1, in_e2: e2, in_dist1: dist1, in_dist2: dist2, \
                    in_y: relations, in_epoch: epoch, in_pos1: position1, in_pos2: position2}
        
        if is_training:
            _, scores, loss, summary = session.run(
                [model.train_op, model.scores, model.loss, model.merged_summary],
                feed_dict=feed_dict
            )
            pred = prediction(scores, dataset, classnum)
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
                        step, acc_count / (step * batch_size) * 100, loss,
                        time.time() - start_time
                    )
                )
        else:
            #TODO: (geeticka) figure out why merged_summary doesn't exist for the non train model
            scores, = session.run(
                    [model.scores],
                    feed_dict=feed_dict
            )
            pred = prediction(scores, dataset, classnum)
            acc = accuracy(pred, relations)
            acc_count += acc

        preds.extend(pred)
            #global_step = tf.train.global_step(session, model.global_step)
            #model.writer.add_summary(summary, global_step)

    return acc_count / (tot_data), preds
