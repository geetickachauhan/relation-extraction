import tensorflow as tf

#updating the Model class to be CRCNN
class Model(object):
    def __init__(self, config, embeddings, is_training=True):
        tf.set_random_seed(config.seed)
        dw   = embeddings.shape[1]
        dp   = config.pos_embed_size
        np   = config.pos_embed_num
        n    = config.max_len
        dc   = config.num_filters
        nr   = config.classnum # number of relations; don't pass via config. Some other way.
        elmo_es = 1024 #elmo embedding size
        elmo_layers = 3 # elmo language model layers
        keep_prob = config.keep_prob

        # Inputs
        # Sentences
        in_x     = tf.placeholder(dtype=tf.int32, shape=[None, n],                          name='in_x')
        # Positions
        in_dist1 = tf.placeholder(dtype=tf.int32, shape=[None, n],                          name='in_dist1')
        in_dist2 = tf.placeholder(dtype=tf.int32, shape=[None, n],                          name='in_dist2')
        # Entities
        in_e1    = tf.placeholder(dtype=tf.int32, shape=[None, config.max_e1_len],          name='in_e1')
        in_e2    = tf.placeholder(dtype=tf.int32, shape=[None, config.max_e2_len],          name='in_e2')

        # Labels
        in_y     = tf.placeholder(dtype=tf.int32, shape=[None],                              name='in_y')
        # epoch
        in_epoch = tf.placeholder(dtype=tf.int32, shape=[],                                  name='epoch')

        # embeddings
        embed = tf.get_variable(initializer=embeddings, dtype=tf.float32, name='word_embed')
        # 3 is num of layers in LM and 1024 is hidden layer dimension in the elmo model, to be converted to variable
        in_elmo = tf.placeholder(dtype=tf.float32, shape=[None, elmo_layers, n, elmo_es],     name='in_elmo')

        self.inputs = (in_x, in_e1, in_e2, in_dist1, in_dist2, in_y, in_epoch, in_elmo)
        # TODO(geeticka): Don't comment out, control with a switch. config.verbosity_level.
        #print("Embeddings shape", embed.shape)
        #print("in_dep shape", in_dep.shape)

        #TODO: (geeticka) Think about whether it makes sense to have the same initializer for the
        #regular vs dependency part of the model
        initializer = tf.truncated_normal_initializer(stddev=0.1, seed=config.seed)
        #initializer_pos_dep = tf.truncated_normal_initializer(stddev=0.1)
        #TODO: (geeticka) in the future may want to try to combine the embedding matrices for the
        # entity 1 vs entity 2; same for the dependency position embeddings
        pos1_embed = tf.get_variable(initializer=initializer,shape=[np, dp], name='position1_embed')
        pos2_embed = tf.get_variable(initializer=initializer,shape=[np, dp], name='position2_embed')
        # rel_embed = tf.get_variable(initializer=initializer,shape=[nr, dc], name='relation_embed')

        # based upon https://stackoverflow.com/questions/50175913/tensorflow-replacing-feeding-a-placeholder-of-a-graph-with-tf-variable
        elmo_weights = tf.get_variable('elmo_weights', [1], trainable=False)
        elmo_weights = in_elmo + elmo_weights
        print("Shape of in_elmo", in_elmo.shape)
        print("Shape of elmo weights", elmo_weights.shape)
        tf.summary.histogram("elmo_weights", elmo_weights)
        # referring to https://github.com/allenai/bilm-tf/blob/master/bilm/elmo.py in order to
        # generate the weighted sum of the elmo embeddings
        # there is also a how to https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md
        # which is helpful because they provide suggestions on the different hyperparameters
        elmo_weighted_sum = self.elmo_weight_layers(elmo_weights)
        print("elmo weighted sum", elmo_weighted_sum.shape)


        tf.summary.histogram("word_embedding_matrix", embed)
        tf.summary.histogram("position1_embedding_matrix", pos1_embed)
        tf.summary.histogram("position2_embedding_matrix", pos2_embed)
        tf.summary.histogram("elmo_weighted_sum", elmo_weighted_sum)


        # embdding lookup
        # e1 = tf.nn.embedding_lookup(embed, in_e1, name='e1')# bz,dw
        # e2 = tf.nn.embedding_lookup(embed, in_e2, name='e2')# bz,dw
        x     = tf.nn.embedding_lookup(embed,      in_x,     name='x')   # bz,n,dw
        dist1 = tf.nn.embedding_lookup(pos1_embed, in_dist1, name='dist1')#bz, n, k,dp
        dist2 = tf.nn.embedding_lookup(pos2_embed, in_dist2, name='dist2')# bz, n, k,dp
        # y = tf.nn.embedding_lookup(rel_embed, in_y, name='y')# bz, dc

        # build regularizer
        regularizer = tf.contrib.layers.l2_regularizer(scale=config.l2_reg_lambda)
        # main convolution (sentence and word position embeddings)
        # x: (batch_size, max_len, embdding_size, 1)
        # w: (filter_size, embdding_size, 1, num_filters)
        d = dw + elmo_es + 2*dp
        list_to_concatenate = [x, elmo_weighted_sum, dist1, dist2]
        h_pool_flat, filter_sizes = self.simple_convolution(n, d, list_to_concatenate,
                config.filter_sizes, dc, keep_prob, is_training, '', initializer, regularizer)

        #TODO: create another convolution and concatenate that pooled output with h_pool_flat after flattening
        # that too
        h_pool_flat_final = h_pool_flat
        output_d = dc*len(filter_sizes)
        # concatenate with the reverse feature
        # h_pool_flat = tf.concat([h_pool_flat, in_reversed], 1)

        # output
        W_o = tf.get_variable(initializer=initializer,shape=[output_d, nr]\
                ,name='w_o', regularizer=regularizer)
        scores = tf.matmul(h_pool_flat_final, W_o, name='scores')
        in_y_onehot = tf.one_hot(in_y, nr, on_value=1., off_value=0., axis=-1)
        # others_rel_mask = tf.cast(in_y != 1, dtype=tf.int32)
        pos_scores = tf.reduce_sum(tf.multiply(scores, in_y_onehot), axis=1)
        mask = tf.one_hot(in_y, nr, on_value=-10000., off_value=0., axis=-1)
        neg_scores = tf.reduce_max(tf.add(scores, mask), axis=1)# bz,

        m_plus = config.m_plus
        m_minus = config.m_minus
        gamma = config.gamma
        loss = tf.reduce_mean(tf.log(1 + tf.exp(gamma * (m_plus - pos_scores))) + \
               tf.log(1 + tf.exp(gamma * (m_minus + neg_scores))))

        #predict = tf.argmax(scores, 1, name="predictions")
        #predict = tf.cast(predict, dtype=tf.int32)
        #acc = tf.reduce_sum(tf.cast(tf.equal(predict, in_y), dtype=tf.int32))
        #self.predict = predict
        #self.acc = acc
        self.scores = scores

        #tf.summary.scalar("accuracy", self.acc)
        # loss = tf.reduce_mean(
        #   tf.nn.softmax_cross_entropy_with_logits(logits=scores,
        #                                           labels=tf.one_hot(in_y, nr))
        # )

        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        l2_loss = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        self.loss = loss + l2_loss
        tf.summary.scalar("loss", self.loss)

        if not is_training:
            return


        # tf.logging.set_verbosity(tf.logging.ERROR)
        global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_step = global_step
        # train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
        # reg_op = optimizer2.minimize(l2_loss)
        # optimizer
        # optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
        #boundaries = [60 * 64, 90 * 64]
        #do self.global_step
        #boundaries = [60, 90]
        #values = [0.0001, 0.00001, 0.000001]
        learning_rate = tf.train.piecewise_constant(in_epoch, config.lr_boundaries, config.lr_values)
        tf.summary.scalar("learning_rate", learning_rate)
        #print("Learning Rate", learning_rate)
        if config.sgd_momentum is False:
            optimizer = tf.train.AdamOptimizer(learning_rate)
        else:
            print("Using SGD with momentum")
            optimizer = tf.train.MomentumOptimizer(learning_rate, config.momentum)
        # optimizer2 = tf.train.AdamOptimizer(config.learning_rate2)

        #    grads_raw = optimizer.compute_gradients(loss)
        #    grads, _ = tf.clip_by_global_norm([g for g, v in grads_raw], config.grad_clipping)
        #    capped_gvs = zip(grads, [v for g, v in grads_raw])


        #grads = optimizer.compute_gradients(self.loss)
        #for g, v in grads: tf.summary.histogram('grad_%s' % v.name[:-2], g)
        # sometimes below throws a None exceptions; how can gradients be none?
        #self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(config.tensorboard_folder)
        self.writer.add_graph(tf.get_default_graph())

    # Inspired from https://github.com/allenai/bilm-tf/blob/master/bilm/elmo.py
    # the authors suggest not to do layer normalization, so I am not going to worry about that
    def elmo_weight_layers(self, elmo_weights):
        # Recommended hyperparameter
        #settings on https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md
        # not doing layer norm, add some dropout in the next layer of the network
        # in the reference code, mask is used for layer norm so i don't need it
        # adding l2 regularization
        n_lm_layers = 3
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

    def simple_convolution(self, max_sen_len, dim, list_to_concatenate, filter_sizes,
            filter_dimension, keep_prob, is_training, prefix, initializer, regularizer):
        # x: (batch_size, max_sen_len, embdding_size, 1)
        # w: (filter_size, embedding_size, 1, num_filters)
        d = dim
        dc = filter_dimension
        n = max_sen_len
        filter_sizes = [int(size) for size in filter_sizes.split(',')]
        pooled_outputs = []
        #print("x, dis1, dist2", x.shape, dist1.shape, dist2.shape)
        #print("After concatenation of x, dist1 and dist2", tf.concat([x, dist1, dist2], -1).shape)
        if len(list_to_concatenate) > 1:
            concatenated = tf.concat(list_to_concatenate, -1)
        else:
            concatenated = list_to_concatenate[0]
        x_conv = tf.reshape(concatenated, # bz, n, d
                                [-1,n,d,1])
        #reshape is needed above to add that extra 4th dimension
        #print("x_conv shape", x_conv.shape)
        if is_training and keep_prob < 1:
            x_conv = tf.nn.dropout(x_conv, keep_prob)

        for i, k in enumerate(filter_sizes):
          with tf.variable_scope("conv-%d" % k):# , reuse=False
            w = tf.get_variable(initializer=initializer,shape=[k, d, 1, dc],name='weight'+prefix,
                    regularizer=regularizer)
            b = tf.get_variable(initializer=initializer,shape=[dc],name='bias'+prefix,
                    regularizer=regularizer)
            conv = tf.nn.conv2d(x_conv, w, strides=[1,1,d,1],padding="SAME")
            #print("conv shape", conv.shape)
            h = tf.nn.tanh(tf.nn.bias_add(conv,b),name="h"+prefix) # bz, n, 1, dc
            #print("h shape", h.shape)
            # max pooling
            pooled = tf.nn.max_pool(h,
                                ksize=[1,n,1,1],
                                strides=[1,n,1,1],
                                padding="SAME"
                  )
            #print("Pooled shape", pooled.shape)
            pooled_outputs.append(pooled)
        h_pool = tf.concat(pooled_outputs, 3)
        #print("h_pool before flattening", h_pool.shape)
        h_pool_flat = tf.reshape(h_pool,[-1,dc*len(filter_sizes)])
        #print("h_pool_flat before flattening", h_pool_flat.shape)

        if is_training and keep_prob < 1:
            h_pool_flat = tf.nn.dropout(h_pool_flat, keep_prob)

        return h_pool_flat, filter_sizes
