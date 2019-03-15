import tensorflow as tf

class elmo_wrapper(object):
    def __init__(self,):
        self.elmo_es = 1024 # elmo embedding size
        self.elmo_layers = 3 # elmo language model layers
        self.in_elmo = None

    def assign_in_elmo(self, in_elmo):
        self.in_elmo = in_elmo

    def get_elmo_embed_size(self):
        return self.elmo_es

    def get_elmo_layers(self):
        return self.elmo_layers
    
    # make sure that assign_in_elmo has been called
    def get_elmo_weighted_sum(self):
        if self.in_elmo is None:
            raise Exception("You need to assign the placeholder in_elmo before generating weighted sum!")

        # based upon https://stackoverflow.com/questions/50175913/tensorflow-replacing-feeding-a-placeholder-of-a-graph-with-tf-variable
        elmo_weights = tf.get_variable('elmo_weights', [1], trainable=False)
        elmo_weights = self.in_elmo + elmo_weights
        print("Shape of in_elmo", self.in_elmo.shape)
        print("Shape of elmo weights", elmo_weights.shape)
        tf.summary.histogram("elmo_weights", elmo_weights)
        # referring to https://github.com/allenai/bilm-tf/blob/master/bilm/elmo.py in order to
        # generate the weighted sum of the elmo embeddings
        # there is also a how to https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md
        # which is helpful because they provide suggestions on the different hyperparameters
        elmo_weighted_sum = self.elmo_weight_layers(elmo_weights)
        print("elmo weighted sum", elmo_weighted_sum.shape)
        tf.summary.histogram("elmo_weighted_sum", elmo_weighted_sum)
        return elmo_weighted_sum

    # Inspired from https://github.com/allenai/bilm-tf/blob/master/bilm/elmo.py
    # the authors suggest not to do layer normalization, so I am not going to worry about that
    def elmo_weight_layers(self, elmo_weights):
        # Recommended hyperparameter
        #settings on https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md
        # not doing layer norm, add some dropout in the next layer of the network
        # in the reference code, mask is used for layer norm so i don't need it
        # adding l2 regularization
        n_lm_layers = self.elmo_layers
        def _l2_regularizer(weights):
            return 0.001 * tf.reduce_sum(tf.square(weights))
        # set l2 regularization of 0.001

        W = tf.get_variable('Elmo_weight',
                shape=(n_lm_layers,),
                initializer = tf.zeros_initializer,
                regularizer = _l2_regularizer,
                trainable = True,
        )
        normed_weights = tf.split(
                tf.nn.softmax(W + 1.0/n_lm_layers), n_lm_layers
        )
        tf.summary.histogram("elmo_weights for weighted sum", normed_weights)
        # split the LM layers from (num_samples, 3, words, 1024) to
        # 3 tensors of shapes (num_samples, words, 1024)
        layers = tf.split(elmo_weights, n_lm_layers, axis = 1)

        # compute the weighted, normalized LM activations
        pieces = []
        for w, t in zip(normed_weights, layers):
            pieces.append(w * tf.squeeze(t, squeeze_dims=1))
        sum_pieces = tf.add_n(pieces)

        # the code has a regularization op, which I am not sure how to use at the moment so I will skip
        gamma = tf.get_variable('Elmo_gamma',
                shape=(1,),
                initializer=tf.ones_initializer,
                regularizer=None,
                trainable=True,
        )
        tf.summary.histogram("elmo_gamma", gamma)
        weighted_lm_layers = sum_pieces * gamma

        return weighted_lm_layers
        # need to return the weighted sum of the hidden layer, with the gamma operator
        # the weights need to be trainable
