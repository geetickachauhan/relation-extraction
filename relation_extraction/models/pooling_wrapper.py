import tensorflow as tf

class pooling_wrapper(object):
    def __init__(self,):
        self.num_pieces = None
        self.p1_ind = None
        self.p2_ind = None

    def get_num_pieces(self):
        return self.num_pieces
    
    def assign_splitting_pieces(self, p1_ind, p2_ind):
        self.p1_ind = p1_ind
        self.p2_ind = p2_ind

    # based on
    # https://github.com/nicolay-r/sentiment-pcnn/blob/7dc58bf34eab609aebe258dc1157010653994920/networks/pcnn/pcnn_core.py
    def piecewise_splitting(self, i, p1_ind, p2_ind, bwc_conv, channels_count, outputs):

        '''
        Split the tensor into 3 pieces according to the position of the entities in the sentence
        '''
        l_ind = tf.minimum(tf.gather(p1_ind, [i]), tf.gather(p2_ind, [i])) #left
        r_ind = tf.maximum(tf.gather(p1_ind, [i]), tf.gather(p2_ind, [i])) #right

        width = tf.Variable(bwc_conv.shape[1], dtype=tf.int32) # total width (i.e. max sentence length)

        b_slice_from = [i, 0, 0]
        b_slice_size = tf.concat([[1], l_ind + 1, [channels_count]], 0)
        m_slice_from = tf.concat([[i], l_ind + 1, [0]], 0)
        m_slice_size = tf.concat([[1], r_ind - l_ind, [channels_count]], 0)
        a_slice_from = tf.concat([[i], r_ind + 1, [0]], 0)
        a_slice_size = tf.concat([[1], width - r_ind - 1, [channels_count]], 0)

        bwc_split_b = tf.slice(bwc_conv, b_slice_from, b_slice_size)
        bwc_split_m = tf.slice(bwc_conv, m_slice_from, m_slice_size)
        bwc_split_a = tf.slice(bwc_conv, a_slice_from, a_slice_size)
       

        pad_b = tf.concat([[[0,0]],
                            tf.reshape(tf.concat([width - l_ind - 1, [0]], 0), shape=[1,2]),
                            [[0,0]]],
                            axis=0)
        
        pad_m = tf.concat([[[0,0]],
                            tf.reshape(tf.concat([width - r_ind + l_ind, [0]], 0), shape=[1,2]),
                            [[0,0]]],
                            axis=0)

        pad_a = tf.concat([[[0,0]],
                            tf.reshape(tf.concat([r_ind + 1, [0]], 0), shape=[1,2]),
                            [[0,0]]],
                            axis=0)

        bwc_split_b = tf.pad(bwc_split_b, pad_b, constant_values=tf.float32.min)
        bwc_split_m = tf.pad(bwc_split_m, pad_m, constant_values=tf.float32.min)
        bwc_split_a = tf.pad(bwc_split_a, pad_a, constant_values=tf.float32.min)

        outputs = outputs.write(i, [[bwc_split_b, bwc_split_m, bwc_split_a]])

        i += 1
        return i, p1_ind, p2_ind, bwc_conv, channels_count, outputs

    def piecewise_max_pooling(self, h):
        '''
        Given the output of the tanh function, in the shape batch_size, max_sen_len, 1, channels_count
        and the location of the ending index of the 2 entities, perform the splitting and
        return a piecewise max pooled operation
        '''
        self.num_pieces = 3
        if self.p1_ind is None or self.p2_ind is None:
            raise Exception("Piecewise pooling should have gotten information about the pieces to split")
        p1_ind = self.p1_ind
        p2_ind = self.p2_ind
        dc = int(h.shape[-1])
        max_sen_len = int(h.shape[1])
        bwc_conv = tf.squeeze(h)
        bwc_conv = tf.reshape(bwc_conv, [-1, max_sen_len, dc])

        variable_batch_size = tf.shape(h)[0]
        # handling variable batch size according to
        #https://stackoverflow.com/questions/40685087/tensorflow-converting-unknown-dimension-size-of-a-tensor-to-int
        sliced = tf.TensorArray(dtype=tf.float32, size=variable_batch_size, infer_shape=False, dynamic_size=True)
        _, _, _, _, _, sliced = tf.while_loop(
                    lambda i, *_: tf.less(i, variable_batch_size),
                    self.piecewise_splitting,
                    [0, p1_ind, p2_ind, bwc_conv, dc, sliced])

        # concat is needed below to convert all individual tensors into one tensor
        sliced = tf.squeeze(sliced.concat()) # batch_size, slices, max_sen_len, channels_count
        sliced = tf.reshape(sliced, [variable_batch_size, self.num_pieces, max_sen_len, dc])
        bwgc_mpool = tf.nn.max_pool(sliced,
                ksize=[1, 1, max_sen_len, 1],
                strides=[1, 1, max_sen_len, 1],
                padding='SAME')
        #TODO (geeticka) need to reshape in piecewise_splitting
        bwc_mpool = tf.squeeze(bwgc_mpool, [2]) # because the 3rd dimension becomes 1

        bcw_mpool = tf.transpose(bwc_mpool, perm=[0,2,1]) #TODO (geeticka) why is this transpose needed?
        bc_pmpool = tf.reshape(bcw_mpool, [-1, self.num_pieces*dc]) 
        # 3 depends on the number of pieces generated in piecewise_splitting
        return bc_pmpool

    def simple_max_pooling(self, h):
        '''
        Given the output of the tanh function, in the shape batch_size, max_sen_len, 1, channels_count
        perform a simple max pooling over the whole sentence
        '''
        self.num_pieces = 1 # this is as if only one piece exists
        max_sen_len = int(h.shape[1])
        dc = int(h.shape[-1])
        pool = tf.nn.max_pool(h,
                ksize=[1, max_sen_len, 1, 1],
                strides=[1, max_sen_len, 1, 1],
                padding='SAME')
        pool = tf.reshape(pool, [-1, dc])
        return pool
