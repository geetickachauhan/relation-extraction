'''
Author: Geeticka Chauhan
A wrapper class for the BERT contextualized embeddings
'''

import tensorflow as tf

class bert_wrapper(object):
    def __init__(self, es):
        self.bert_es = es # bert embedding size; can be 1024 or 768
        self.bert_layers = 4 # elmo language model layers
        self.in_bert = None

    def assign_in_bert(self, in_bert):
        self.in_bert = in_bert
    
    def get_bert_embed_size(self):
        return self.bert_es

    def get_bert_layers(self):
        return self.bert_layers
    
    # make sure that assign_in_elmo has been called
    def get_bert_weighted_sum(self):
        if self.in_bert is None:
            raise Exception("You need to assign the placeholder in_bert before generating weighted sum!")

        # based upon https://stackoverflow.com/questions/50175913/tensorflow-replacing-feeding-a-placeholder-of-a-graph-with-tf-variable
        bert_weights = tf.get_variable('bert_weights', [1], trainable=False)
        bert_weights = self.in_bert + bert_weights
        print("Shape of in_bert", self.in_bert.shape)
        print("Shape of bert weights", bert_weights.shape)
        tf.summary.histogram("bert_weights", bert_weights)
        # referring to https://github.com/allenai/bilm-tf/blob/master/bilm/elmo.py in order to
        # generate the weighted sum of the elmo embeddings
        # there is also a how to https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md
        # which is helpful because they provide suggestions on the different hyperparameters
        bert_weighted_sum = self.bert_weight_layers(bert_weights)
        print("bert weighted sum", bert_weighted_sum.shape)
        tf.summary.histogram("bert_weighted_sum", bert_weighted_sum)
        return bert_weighted_sum

    # Inspired from https://github.com/allenai/bilm-tf/blob/master/bilm/elmo.py
    # the authors suggest not to do layer normalization, so I am not going to worry about that
    def bert_weight_layers(self, bert_weights):
        # Recommended hyperparameter
        #settings on https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md
        # not doing layer norm, add some dropout in the next layer of the network
        # in the reference code, mask is used for layer norm so i don't need it
        # adding l2 regularization
        n_lm_layers = self.bert_layers
        def _l2_regularizer(weights):
            return 0.001 * tf.reduce_sum(tf.square(weights))
        # set l2 regularization of 0.001

        W = tf.get_variable('Bert_weight',
                shape=(n_lm_layers,),
                initializer = tf.zeros_initializer,
                regularizer = _l2_regularizer,
                trainable = True,
        )
        normed_weights = tf.split(
                tf.nn.softmax(W + 1.0/n_lm_layers), n_lm_layers
        )
        tf.summary.histogram("bert_weights for weighted sum", normed_weights)
        # split the LM layers from (num_samples, 3, words, 1024) to
        # 3 tensors of shapes (num_samples, words, 1024)
        layers = tf.split(bert_weights, n_lm_layers, axis = 1)

        # compute the weighted, normalized LM activations
        pieces = []
        for w, t in zip(normed_weights, layers):
            pieces.append(w * tf.squeeze(t, squeeze_dims=1))
        sum_pieces = tf.add_n(pieces)

        # the code has a regularization op, which I am not sure how to use at the moment so I will skip
        gamma = tf.get_variable('Bert_gamma',
                shape=(1,),
                initializer=tf.ones_initializer,
                regularizer=None,
                trainable=True,
        )
        tf.summary.histogram("bert_gamma", gamma)
        weighted_lm_layers = sum_pieces * gamma

        return weighted_lm_layers
        # need to return the weighted sum of the hidden layer, with the gamma operator
        # the weights need to be trainable
