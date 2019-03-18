import tensorflow as tf

def ranking_loss(in_y, scores, nr, m_plus, m_minus, gamma, name=None):

    with tf.name_scope(name, 'ranking_loss'):
        in_y_onehot = tf.one_hot(in_y, nr, on_value=1., off_value=0., axis=-1)
        # others_rel_mask = tf.cast(in_y != 1, dtype=tf.int32)
        pos_scores = tf.reduce_sum(tf.multiply(scores, in_y_onehot), axis=1)
        mask = tf.one_hot(in_y, nr, on_value=-10000., off_value=0., axis=-1)
        neg_scores = tf.reduce_max(tf.add(scores, mask), axis=1)# bz,

        loss = tf.reduce_mean(tf.log(1 + tf.exp(gamma * (m_plus - pos_scores))) + \
               tf.log(1 + tf.exp(gamma * (m_minus + neg_scores))))
        return loss
