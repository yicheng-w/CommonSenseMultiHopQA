import tensorflow as tf

epsilon = 1e-10

def get_keep_prob(dropout_rate, is_training):
    keep_prob = tf.cond(
  		is_training,
  		lambda: tf.constant(1.0 - dropout_rate),
  		lambda: tf.constant(1.0))
    return keep_prob


def sparse_cross_entropy_with_probs(
        _sentinel=None,  # pylint: disable=invalid-name
        labels=None,
        logits=None,
        name=None):
    '''
    in case where ``logits'' is actually a probability distribution, computes
    cross_entropy directly

    logits: shape [items, num_classes]
    labels: shape [items]
    '''

    items_idx = tf.range(0, limit = tf.shape(logits)[0])
    indices = tf.stack((items_idx, labels), 1)
    probs = tf.gather_nd(logits, indices) + epsilon # [items]
    ce = -tf.log(probs)

    return ce

def sparse_gather_probs(
        _sentinel=None,
        labels=None,
        logits=None,
        name=None):
    '''
    for MC / ranking, in case we do not want to calculate cross-entropy but we
    still work w/ a prob. dist.

    logits: shape [items, num_classes]
    labels: shape [items]
    '''

    items_idx = tf.range(0, limit = tf.shape(logits)[0])
    indices = tf.stack((items_idx, labels), 1)
    probs = tf.gather_nd(logits, indices)

    return probs
