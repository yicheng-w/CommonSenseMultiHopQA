import tensorflow as tf
import numpy as np

def best_span_from_bounds(start_logits, end_logits, bound=None):
    """
    Brute force approach to finding the best span from start/end logits in tensorflow, still usually
    faster then the python dynamic-programming version
    """
    b = tf.shape(start_logits)[0]

    # Using `top_k` to get the index and value at once is faster
    # then using argmax and then gather to get in the value
    top_k = tf.nn.top_k(start_logits + end_logits, k=1)
    values, indices = [tf.squeeze(x, axis=[1]) for x in top_k]

    # Convert to (start_position, length) format
    indices = tf.stack([indices, tf.fill((b,), 0)], axis=1)

    # TODO Might be better to build the batch x n_word x n_word
    # matrix and use tf.matrix_band to zero out the unwanted ones...

    if bound is None:
        n_lengths = tf.shape(start_logits)[1]
    else:
        # take the min in case the bound > the context
        n_lengths = tf.minimum(bound, tf.shape(start_logits)[1])

    def compute(i, values, indices):
        top_k = tf.nn.top_k(start_logits[:, :-i] + end_logits[:, i:])
        b_values, b_indices = [tf.squeeze(x, axis=[1]) for x in top_k]

        b_indices = tf.stack([b_indices, tf.fill((b, ), i)], axis=1)
        indices = tf.where(b_values > values, b_indices, indices)
        values = tf.maximum(values, b_values)
        return i+1, values, indices

    _, values, indices = tf.while_loop(
        lambda ix, values, indices: ix < n_lengths,
        compute,
        [1, values, indices],
        back_prop=False)

    spans = tf.stack([indices[:, 0], indices[:, 0] + indices[:, 1]], axis=1)
    return spans, values

