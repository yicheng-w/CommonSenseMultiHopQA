import tensorflow as tf
from tensorflow.contrib.seq2seq import AttentionMechanism
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import\
    _prepare_memory

class NestedMultiLevelAttn(AttentionMechanism):
    '''
    memory in the format of [b x k x n x h], where we have (k x n) tokens
    divided into k segments, given a query q, we compute attention first at the
    local level to obtain grouped representations of shape [b x k x h], then we
    compute another level of attention to get the final set of context vectors
    with shape b x h

    the alignment returned is the initial alignment distribution with shape
    [b x k x n]
    '''

    def __init__(
            memory,
            memory_sequence_length,
            similarity_function):
        '''
        memory: tensor, shape [b x k x n x h]
        memory_sequence_length: tensor, shape [b x k]'''

        k = tf.shape(memory)[1]
        b = tf.shape(memory)[0]
        h = tf.shape(memory)[-1]

        mem_reshaped = tf.reshape(memory, [b * k, -1, h])
        mem_mask_reshaped = tf.reshape(memory_sequence_length, [-1])

        values = _prepare_memory(mem_reshaped, mem_mask_reshaped)

        self.values = tf.reshape(values, [b, k, -1, h])
        self.sim_func = similarity_function

        with tf.variable_scope("first_lv_attn"):
            self.first_lv_sim_func = similarity_function
        
        with tf.variable_scope("second_lv_attn"):
            self.second_lv_sim_func = similarity_function

    def __call__(self, query, previous_alignments):
        '''
        query should have shape [b x h]
        '''

        b = tf.shape(self.values)[0]
        k = tf.shape(self.values)[1]
        h = tf.shape(self.values)[-1]

        mem_reshaped = tf.reshape(memory, [b * k, -1, h])

        query_expand_dims = tf.expand_dims(query, 1) # [b x 1 x h]

        attn_logits = self.first_lv_sim_func(query_expand_dims, mem_reshaped)
        # [(b*k), n]

        attn_logits_reshaped = tf.reshape(attn_logits, [b, k, -1])
        alignments = tf.nn.softmax(attn_logits_reshaped, -1) # [b x k x n]

        expanded_alignments = tf.expand_dims(attn_logits, 1) # [(b*k) x 1 x n]
        w_lv_context = tf.matmul(expanded_alignments, mem_reshaped) # [(b*k) x 1 x h]
        w_lv_context = tf.squeeze(w_lv_context, 1) # [(b*k) x h]

        w_lv_context_expanded = tf.reshape(w_lv_context, [b, k, h])

        second_attn_logits = self.second_lv_sim_func(query_expand_dims,
                w_lv_context_expanded) # [b x k]
        
