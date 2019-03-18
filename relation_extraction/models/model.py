import tensorflow as tf
import sys
sys.path.append('../../')
from relation_extraction.models import losses
from relation_extraction.models.elmo_wrapper import elmo_wrapper
from relation_extraction.models.pooling_wrapper import pooling_wrapper

#updating the Model class to be CRCNN
class CRCNN(object):
    def __init__(self, config, embeddings, is_training=True):
        self.dw   = embeddings.shape[1]
        self.dp   = config.pos_embed_size
        self.np   = config.pos_embed_num
        self.n    = config.max_len
        self.dc   = config.num_filters
        self.nr   = config.classnum # number of relations; don't pass via config. Some other way.
        self.inputs = None
        self.keep_prob = config.keep_prob
        self.embeddings = embeddings
        self.l2_reg_lambda = config.l2_reg_lambda
        self.max_e1_len = config.max_e1_len
        self.max_e2_len = config.max_e2_len
        self.seed = config.seed
        self.filter_sizes = config.filter_sizes
        self.is_training = is_training
        self.m_plus = config.m_plus
        self.m_minus = config.m_minus
        self.gamma = config.gamma
        self.lr_boundaries = config.lr_boundaries
        self.lr_values = config.lr_values
        self.momentum = config.momentum
        self.sgd_momentum = config.sgd_momentum
        self.tensorboard_folder = config.tensorboard_folder
        self.elmo = elmo_wrapper()
        self.pooling = pooling_wrapper()
        self.use_elmo = config.use_elmo
        self.use_piecewise_pool = config.use_piecewise_pool

    def __run__(self):
        tf.set_random_seed(self.seed)
        in_x, in_dist1, in_dist2, in_e1, in_e2, in_y, in_epoch, embed, initializer, regularizer, pos1_embed, \
                pos2_embed = self.assign_placeholders()

        x, dist1, dist2 = self.embedding_lookup(embed, in_x, pos1_embed, in_dist1, pos2_embed, in_dist2)
        
        if self.use_elmo is True:
            elmo_weighted_sum = self.elmo.get_elmo_weighted_sum()
            d = self.dw + self.elmo.get_elmo_embed_size() + 2*self.dp
            list_to_concatenate = [x, elmo_weighted_sum, dist1, dist2]
        else:
            d = self.dw+2*self.dp
            list_to_concatenate = [x, dist1, dist2]
        
        mode_pool = 'simple_max_pool' if self.use_piecewise_pool is False else 'piecewise_max_pool'
        h_pool_flat, filter_sizes = self.simple_convolution(d, list_to_concatenate, '', initializer, 
                regularizer, mode_pool)
        num_pieces = self.pooling.get_num_pieces()
        if num_pieces is None:
            raise Exception("You should have called a pooling function! Number of pieces cannot be None.")
        output_d = self.dc * num_pieces * len(filter_sizes)

        # output
        W_o = tf.get_variable(initializer=initializer,shape=[output_d, self.nr]\
                ,name='w_o', regularizer=regularizer)
        self.scores = tf.matmul(h_pool_flat, W_o, name='scores')
        loss = losses.ranking_loss(in_y, self.scores, self.nr, self.m_plus, self.m_minus, self.gamma)

        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        l2_loss = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        self.loss = loss + l2_loss
        tf.summary.scalar("loss", self.loss)

        if not self.is_training:
            return

        # tf.logging.set_verbosity(tf.logging.ERROR)
        global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_step = global_step
        learning_rate = tf.train.piecewise_constant(in_epoch, self.lr_boundaries, self.lr_values)
        tf.summary.scalar("learning_rate", learning_rate)
        #print("Learning Rate", learning_rate)
        if self.sgd_momentum is False:
            optimizer = tf.train.AdamOptimizer(learning_rate)
        else:
            print("Using SGD with momentum")
            optimizer = tf.train.MomentumOptimizer(learning_rate, self.momentum)

        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.tensorboard_folder)
        self.writer.add_graph(tf.get_default_graph())

    
    def assign_placeholders(self):
        # Inputs
        # Sentences
        in_x     = tf.placeholder(dtype=tf.int32, shape=[None, self.n],                 name='in_x')
        # Positions
        in_dist1 = tf.placeholder(dtype=tf.int32, shape=[None, self.n],                 name='in_dist1')
        in_dist2 = tf.placeholder(dtype=tf.int32, shape=[None, self.n],                 name='in_dist2')
        # Entities
        in_e1    = tf.placeholder(dtype=tf.int32, shape=[None, self.max_e1_len], name='in_e1')
        in_e2    = tf.placeholder(dtype=tf.int32, shape=[None, self.max_e2_len], name='in_e2')
        # Positions of the entities (for piecewise splitting of the sentence)
        in_pos1  = tf.placeholder(dtype=tf.int32, shape=[None],                             name='in_pos1')
        in_pos2  = tf.placeholder(dtype=tf.int32, shape=[None],                             name='in_pos2')

        # Labels
        in_y     = tf.placeholder(dtype=tf.int32, shape=[None],                              name='in_y')
        # epoch
        in_epoch = tf.placeholder(dtype=tf.int32, shape=[],                                  name='epoch')

        if self.use_elmo is True:
            elmo_layers = self.elmo.get_elmo_layers()
            elmo_es = self.elmo.get_elmo_embed_size()
            # 3 is num of layers in LM and 1024 is hidden layer dimension in the elmo model, to be converted to variable
            in_elmo = tf.placeholder(dtype=tf.float32, shape=[None, elmo_layers, self.n, elmo_es],     name='in_elmo')
            self.elmo.assign_in_elmo(in_elmo)
            self.inputs = (in_x, in_e1, in_e2, in_dist1, in_dist2, in_y, in_epoch, in_elmo, in_pos1, in_pos2)
        else:
            self.inputs = (in_x, in_e1, in_e2, in_dist1, in_dist2, in_y, in_epoch, in_pos1, in_pos2)

        if self.use_piecewise_pool is True:
            self.pooling.assign_splitting_pieces(in_pos1, in_pos2)
        # embeddings
        embed = tf.get_variable(initializer=self.embeddings, dtype=tf.float32, name='word_embed')
        # TODO(geeticka): Don't comment out, control with a switch. config.verbosity_level.
        #print("Embeddings shape", embed.shape)
        #print("in_dep shape", in_dep.shape)

        #TODO: (geeticka) Think about whether it makes sense to have the same initializer for the
        #regular vs dependency part of the model
        initializer = tf.truncated_normal_initializer(stddev=0.1, seed=self.seed)
        # build regularizer
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2_reg_lambda)
        #TODO: (geeticka) in the future may want to try to combine the embedding matrices for the
        # entity 1 vs entity 2; same for the dependency position embeddings
        pos1_embed = tf.get_variable(initializer=initializer,shape=[self.np, self.dp], name='position1_embed')
        pos2_embed = tf.get_variable(initializer=initializer,shape=[self.np, self.dp], name='position2_embed')
        # rel_embed = tf.get_variable(initializer=initializer,shape=[nr, dc], name='relation_embed')


        tf.summary.histogram("word_embedding_matrix", embed)
        tf.summary.histogram("position1_embedding_matrix", pos1_embed)
        tf.summary.histogram("position2_embedding_matrix", pos2_embed)
        
        return in_x, in_dist1, in_dist2, in_e1, in_e2, in_y, in_epoch, embed, initializer, regularizer, \
                pos1_embed, pos2_embed

    def embedding_lookup(self, embed, in_x, pos1_embed, in_dist1, pos2_embed, in_dist2):
        # embdding lookup
        # e1 = tf.nn.embedding_lookup(embed, in_e1, name='e1')# bz,dw
        # e2 = tf.nn.embedding_lookup(embed, in_e2, name='e2')# bz,dw
        x     = tf.nn.embedding_lookup(embed,      in_x,     name='x')   # bz,n,dw
        dist1 = tf.nn.embedding_lookup(pos1_embed, in_dist1, name='dist1')#bz, n, k,dp
        dist2 = tf.nn.embedding_lookup(pos2_embed, in_dist2, name='dist2')# bz, n, k,dp
        # y = tf.nn.embedding_lookup(rel_embed, in_y, name='y')# bz, dc
        return x, dist1, dist2

    
    def simple_convolution(self, dim, list_to_concatenate, prefix, initializer, regularizer,\
            mode='simple_max_pool'):
        # x: (batch_size, max_sen_len, embdding_size, 1)
        # w: (filter_size, embedding_size, 1, num_filters)
        d = dim
        dc = self.dc
        n = self.n
        filter_sizes = [int(size) for size in self.filter_sizes.split(',')]
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
        if self.is_training and self.keep_prob < 1:
            x_conv = tf.nn.dropout(x_conv, self.keep_prob)

        for i, k in enumerate(filter_sizes):
          with tf.variable_scope("conv-%d" % k):# , reuse=False
            w = tf.get_variable(initializer=initializer,shape=[k, d, 1, dc],name='weight'+prefix,
                    regularizer=regularizer)
            b = tf.get_variable(initializer=initializer,shape=[dc],name='bias'+prefix,
                    regularizer=regularizer)
            conv = tf.nn.conv2d(x_conv, w, strides=[1,1,d,1],padding="SAME")
            #print("conv shape", conv.shape)
            h = tf.nn.tanh(tf.nn.bias_add(conv,b),name="h"+prefix) # bz, n, 1, dc
            if mode == 'simple_max_pool':
                bc_pmpool = self.pooling.simple_max_pooling(h)
            else:
                bc_pmpool = self.pooling.piecewise_max_pooling(h)
            pooled_outputs.append(bc_pmpool)
        h_pool_flat = tf.concat(pooled_outputs, -1) # concatenate over the last dimension which is channels
        #print("h_pool_flat", h_pool_flat.shape)

        if self.is_training and self.keep_prob < 1:
            h_pool_flat = tf.nn.dropout(h_pool_flat, self.keep_prob)

        return h_pool_flat, filter_sizes
