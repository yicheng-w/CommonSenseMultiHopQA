import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops

from recurrent_layers import CudnnLstm, CudnnGru

VERY_NEGATIVE_NUMBER = -1e29

def exp_mask(val, mask):
    mask = tf.cast(tf.sequence_mask(mask, tf.shape(val)[1]), tf.float32)
    return val * mask + (1 - mask) * VERY_NEGATIVE_NUMBER

def get_cudnn_rnn(cell_type):
    if cell_type == 'rnn':
        return tf.contrib.cudnnrnn.CudnnRNNRelu
    elif cell_type == 'gru':
        return tf.contrib.cudnnrnn.CudnnGRU
    elif cell_type == 'lstm':
        return tf.contrib.cudnnrnn.CudnnLSTM
    else:
        raise ValueError("Invalid cell type! Got %s" % (cell_type))

def create_multi_rnn(cell_type, hidden_size, layers, keep_prob):
    is_cudnn = False

    if cell_type == 'rnn':
        create_cell = tf.contrib.rnn.BasicRNNCell
    elif cell_type == 'gru':
        create_cell = tf.contrib.rnn.GRUCell
    elif cell_type == 'basic_lstm':
        create_cell = tf.contrib.rnn.BasicLSTMCell
    elif cell_type == 'lstm':
        create_cell = tf.contrib.rnn.LSTMCell
    elif cell_type == 'cudnn_lstm':
        create_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell
    elif cell_type == 'cudnn_gru':
        create_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell
    else:
        raise ValueError("Invalid cell type! Got %s" % (cell_type))

    cell = lambda : create_cell(num_units = hidden_size)
    add_dropout = lambda cell: tf.contrib.rnn.DropoutWrapper(
            cell,
            input_keep_prob=keep_prob,
            seed=11235)

    if layers == 1 and not is_cudnn:
        return add_dropout(cell())

    cells = [cell() for _ in range(layers)]
    return add_dropout(tf.contrib.rnn.MultiRNNCell(cells))

def dot_product_attention(tensor1, tensor2, with_bias=True):
    '''a = t1 * t2 + b'''
    dots = tf.matmul(tensor1, tensor2, transpose_b = True)

    if with_bias:
        bias = tf.get_variable("bias", shape=(), dtype=tf.float32)
        dots += bias

    return dots

def tri_linear_attention(x, keys, with_bias=True):
    '''a = w1t1 + w2t2 + t1w3t2 + b'''
    init = tf.contrib.layers.xavier_initializer()

    key_w = tf.get_variable("key_w", shape=keys.shape.as_list()[-1], initializer=init, dtype=tf.float32)
    key_logits = tf.tensordot(keys, key_w, axes=[[2], [0]])  # (batch, key_len)

    x_w = tf.get_variable("input_w", shape=x.shape.as_list()[-1], initializer=init, dtype=tf.float32)
    x_logits = tf.tensordot(x, x_w, axes=[[2], [0]])  # (batch, x_len)

    dot_w = tf.get_variable("dot_w", shape=x.shape.as_list()[-1], initializer=init, dtype=tf.float32)

    # Compute x * dot_weights first, the batch mult with x
    x_dots = x * tf.expand_dims(tf.expand_dims(dot_w, 0), 0)
    dot_logits = tf.matmul(x_dots, keys, transpose_b=True)

    out = dot_logits + tf.expand_dims(key_logits, 1) + tf.expand_dims(x_logits, 2)

    if with_bias:
        bias = tf.get_variable("bias", shape=(), dtype=tf.float32)
        out += bias

    return out

def bahdanau_attention(query, keys, num_units, with_bias=True):
    '''
    a = v * tanh(Wq q + Wk k + b)
    '''
    hq = query.get_shape().as_list()[-1]
    Wq = tf.get_variable("Wq", [hq, num_units], query.dtype)

    print(query.get_shape().as_list())
    print(Wq.get_shape().as_list())

    aq = tf.einsum("bij,jk->bik", query, Wq) # [b, 1, h_attn]

    hk = keys.get_shape().as_list()[-1]
    Wk = tf.get_variable("Wk", [hk, num_units], keys.dtype)

    ak = tf.einsum("bij,jk->bik", keys, Wk) # [b, n, h_attn]

    pre_tanh = aq + ak # [b x n x h_attn]

    if with_bias:
        b = tf.get_variable("bias", shape=(), dtype=tf.float32)
        pre_tanh += b

    out = math_ops.tanh(pre_tanh) # [b x n x h_attn]

    v = tf.get_variable("V", [num_units, 1], dtype=tf.float32)

    return tf.squeeze(tf.einsum("bij,jk->bik", out, v), 2) # [b x n]


def compute_attention_mask(x_mask, mem_mask, x_word_dim, key_word_dim):
    """ computes a (batch, x_word_dim, key_word_dim) bool mask for clients that want masking """
    if x_mask is None and mem_mask is None:
        return None
    elif x_mask is None or mem_mask is None:
        raise NotImplementedError()

    x_mask = tf.sequence_mask(x_mask, x_word_dim)
    mem_mask = tf.sequence_mask(mem_mask, key_word_dim)
    join_mask = tf.logical_and(tf.expand_dims(x_mask, 2), tf.expand_dims(mem_mask, 1))
    return join_mask

def attention_layer(x, a, x_mask, a_mask, sim_func, scope="",
        output_alignment=False):
    '''
    computes enhanced representation of x attending on a
    x: tensor, shape [b x n x h]
    a: tensor, shape [b x m x h]
    x_mask: tensor, true length of x, shape [b]
    a_mask: tensor, true length of a, shape [b]
    sim_func: similarity function used to compute attention scores, has
        signature sim_func(tensor1, tensor2) -> attn score
    '''
    n = tf.shape(x)[1]
    m = tf.shape(a)[1]

    dist_matrix = sim_func(x, a)
    #print("Sim matrix:")
    #print(dist_matrix.get_shape().as_list()) # b x n x m

    joint_mask = compute_attention_mask(x_mask, a_mask, n, m)
    if joint_mask is not None:
        dist_matrix += VERY_NEGATIVE_NUMBER * \
                (1 - tf.cast(joint_mask, dist_matrix.dtype))

    probs = tf.nn.softmax(dist_matrix) # b x n x m
    #print("Probs:")
    #print(probs.get_shape().as_list())

    attention_vector = tf.matmul(probs, a) # b x n x h
    #print("Attn vect:")
    #print(attention_vector.get_shape().as_list())
    
    if output_alignment:
        return attention_vector, probs
    else:
        return attention_vector

def bidaf_attention(context_h, query_h, context_mask, query_mask, sim_func,
        output_alignment = False):
    '''
    slightly modified version of c. clark's bidaf code

    output_alignment: boolean, whether to print out the alignment matrix
    '''
    context_word_dim = tf.shape(context_h)[1]
    query_word_dim = tf.shape(query_h)[1]

    dist_matrix = sim_func(context_h, query_h)
    joint_mask = compute_attention_mask(context_mask, query_mask,
            context_word_dim, query_word_dim)
    if joint_mask is not None:
        dist_matrix += VERY_NEGATIVE_NUMBER * \
                (1 - tf.cast(joint_mask, dist_matrix.dtype))
    query_probs = tf.nn.softmax(dist_matrix)
    # probability of each query_word per context_word

    # Batch matrix multiplication to get the attended vectors
    select_query = tf.matmul(query_probs, query_h)  # (batch, context_words, q_dim)

    # select query-to-context
    context_dist = tf.reduce_max(dist_matrix, axis=2)  # (batch, context_word``s)
    context_probs = tf.nn.softmax(context_dist)  # (batch, context_words)
    select_context = tf.einsum("ai,aik->ak", context_probs, context_h)  # (batch, context_dim)
    select_context = tf.expand_dims(select_context, 1)

    output = tf.concat([context_h, select_query, context_h * select_query,
        context_h * select_context], axis=2)

    if output_alignment:
        return output, query_probs

    return output

def mem_nn_hop(input, memories):
    '''
    mem_nn_hop: one hop of a memory network (https://www.arxiv.org/pdf/1503.08895.pdf)
    input: tensor, the input (question), shape [b x h]
    memories: tensor, memories to look at (context), shape [b x n x h]
    '''
    u_k = input

    # hack to get around no reduce_dot
    u_temp = tf.transpose(tf.expand_dims(u_k, -1), [0, 2, 1]) # [b x 1 x h]
    dotted = tf.reduce_sum(memories * u_temp, 2) # dot product --> b x n

    # Calculate probabilities
    probs = tf.nn.softmax(dotted) # b x n
    probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1]) # b x 1 x n
    c_temp = tf.transpose(memories, [0, 2, 1]) # b x h x n
    o_k = tf.reduce_sum(c_temp * probs_temp, 2) # b x h
    u_k = u_k + o_k

    return u_k

def mem_nn(input, memories, input_pos_enc, n):
    '''
    mem_nn: memory network
    input: tensor, the input (question), shape [b x m x h]
    memories: tensor, memories to look at (context), shape [b x n x h]
    input_pos_enc: tensor, pos. encoding of the input, shape [b x m x h]
    n: int, # of hops to perform
    '''
    u = tf.reduce_sum(input_pos_enc * input, 1) # [b x h]
    print(u.get_shape().as_list())
    for _ in range(n):
        u = mem_nn_hop(u, memories)
    return u

def fuse_gate(lhs, rhs):
    dim = lhs.shape.as_list()[-1]
    assert rhs.shape.as_list()[-1] == dim
    lhs1 = tf.layers.dense(lhs, dim, activation=None, name='lhs1')
    rhs1 = tf.layers.dense(rhs, dim, activation=None, name='rhs1')

    print(lhs1.get_shape().as_list())

    z = tf.sigmoid(lhs1 + rhs1)

    return z * lhs + (1-z) * rhs


def diin_fuse_gate(lhs, rhs):
    dim = lhs.shape.as_list()[-1]
    assert rhs.shape.as_list()[-1] == dim

    lhs1 = tf.layers.dense(lhs, dim, activation=None, name='lhs1')
    rhs1 = tf.layers.dense(rhs, dim, activation=None, name='rhs1')

    print(lhs_1.get_shape().as_list())

    z = tf.tanh(lhs1 + rhs1)

    lhs2 = tf.layers.dense(lhs, dim, activation=None, name='lhs2')
    rhs2 = tf.layers.dense(rhs, dim, activation=None, name='rhs2')

    f = tf.sigmoid(lhs2 + rhs2)

    print(f.get_shape().as_list())

    lhs3 = tf.layers.dense(lhs, dim, activation=None, name='lhs3')
    rhs3 = tf.layers.dense(rhs, dim, activation=None, name='rhs3')

    r = tf.sigmoid(lhs3 + rhs3)

    print(r.get_shape().as_list())

    out = f * lhs + r * z

    print(out.get_shape().as_list())

    return out

def concat_with_product(t1, t2):
    return tf.concat([t1, t2, t1 * t2], axis = len(t1.shape) - 1)

def self_attention_encoder(x, sim_func, mask=None, merge_function = None,
        output_alignment=False):
    '''
    self attention encoder
    x: tensor, thing to encode, shape [b x n x h]
    sim_func: similarity function of two tensors
    mask: length of x, tensor, shape [b]
    merge_function: function of two inputs to merge x with x_self_attn, often a
        fuse gate
    output_alignment: boolean, whether to return the alignment matrix
    '''

    x_dim = tf.shape(x)[1]

    dist = sim_func(x, x) # [b x n x n]
    joint_mask = compute_attention_mask(mask, mask, x_dim, x_dim)
    if joint_mask is not None:
        dist += VERY_NEGATIVE_NUMBER * (1 - tf.cast(joint_mask,
            dist.dtype))

    dist = tf.nn.softmax(dist) # [b x n x n]
    print("[b x n x n]")
    print(dist.get_shape().as_list())

    out = tf.matmul(dist, x) # [b x n x h]

    print("[b x n x h]")
    print(out.get_shape().as_list())

    if merge_function is not None:
        out = merge_function(x, out)

    if output_alignment:
        return out, dist

    return out

def bi_rnn_encoder(cell_type, hidden_size, num_layers, keep_prob, inputs,
        input_lengths, output_layer = None):
    fw_cell = create_multi_rnn(cell_type, hidden_size, num_layers, keep_prob)
    bw_cell = create_multi_rnn(cell_type, hidden_size, num_layers, keep_prob)
    if input_lengths != None:
	outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
		fw_cell,
		bw_cell,
		inputs,
		input_lengths,
		dtype=tf.float32)
    else:
	outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
		fw_cell,
		bw_cell,
		inputs,
		dtype=tf.float32)
 
    outputs = tf.concat(outputs, axis=2)

    if output_layer is not None:
        outputs = output_layer(outputs)

    return outputs, final_state

def rnn_encoder(cell_type, hidden_size, num_layers, keep_prob, inputs,
        input_lengths, output_layer=None):
    cell = create_multi_rnn(cell_type, hidden_size, num_layers, keep_prob)
    outputs, final_state = tf.nn.dynamic_rnn(
            cell,
            inputs,
            input_lengths,
            dtype=tf.float32)

    if output_layer is not None:
        outputs = output_layer(outputs)
    
    return outputs, final_state

def bi_cudnn_rnn_encoder(cell_type, hidden_size, num_layers, dropout_rate, inputs,
        input_lengths, is_train, output_layer = None):
    if cell_type == 'lstm':
        RnnLayer = CudnnLstm
    elif cell_type == 'gru':
        RnnLayer = CudnnGru
    else:
        raise ValueError()

    layer = RnnLayer(
            n_units = hidden_size,
            n_layers = num_layers)

    inputs = tf.layers.dropout(inputs, dropout_rate, training=is_train)

    outputs = layer.apply(is_train, inputs, input_lengths)
    print(outputs.get_shape().as_list())

    if output_layer is not None:
        outputs = output_layer(outputs)

    return outputs, None

def bi_cudnn_maxout_rnn_encoder(cell_type, hidden_size, num_layers,
        dropout_rate, inputs, input_lengths, is_train, output_layer=None,
        num_rnns=2):
    outputs = []

    for i in range(num_rnns):
        with tf.variable_scope("worker_%d" % i):
            cur_output, _ = bi_cudnn_rnn_encoder(
                    cell_type,
                    hidden_size,
                    num_layers,
                    dropout_rate,
                    inputs,
                    input_lengths,
                    is_train,
                    output_layer)

            outputs.append(cur_output)

    out = tf.reduce_max(tf.stack(outputs, -1), -1)

    print(out.get_shape().as_list())

    return out, None

#def bi_cudnn_rnn_encoder(cell_type, hidden_size, num_layers, keep_prob, inputs,
#        input_lengths, output_layer = None):
#
#    if cell_type == 'lstm':
#        RnnLayer = tf.contrib.cudnn_rnn.CudnnLSTM
#    elif cell_type == 'gru':
#        RnnLayer = tf.contrib.cudnn_rnn.CudnnGRU
#    elif cell_type == 'rnn':
#        RnnLayer = tf.contrib.cudnn_rnn.CudnnRNNRelu
#    else:
#        raise ValueError("Invalid RNN type! Got %s" % (cell_type))
#
#    input_h = inputs.get_shape().as_list()[-1]
#    batch_size = inputs.get_shape().as_list()[0]
#
#    layer = RnnLayer(
#            num_layers=num_layers,
#            num_units=hidden_size,
#            input_size=input_h,
#            direction=cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION)
#
#    param_shape = layer.params_size().eval()
#
#    params = tf.get_variable(
#        "rnn_parameters",
#        param_shape,
#        tf.float32)
#
#    input_transposed = tf.transpose(inputs, [1, 0, 2])
#    print(input_transposed.get_shape().as_list())
#    print("Should be [None, batch_sz, h]")
#
#    input_processed = tf.layers.dropout(input_transposed, keep_prob)
#
#    if cell_type == 'lstm':
#        if num_layers == 1:
#            initial_state_h = tf.zeros(
#                    (batch_size, hidden_size), tf.float32)
#            initial_state_c = tf.zeros(
#                    (batch_size, hidden_size), tf.float32)
#
#        else:
#            raise ValueError()
#
#        out = layer(input_processed, initial_state_h, initial_state_c,
#                params, True)
#    else:
#        if num_layers == 1:
#            initial_state = tf.zeros(
#                    (batch_size, hidden_size), tf.float32)
#        else:
#            raise ValueError()
#
#        out = layer(input_processed, initial_state, params, True)
#
#    output, out_states_h, out_states_c = out
#
#    output = tf.transpose(output, [1, 0, 2])
#    print(output.get_shape().as_list())
#    print("Should be [batch_sz, None, h]")
#
#    out_states = tf.contrib.rnn.LSTMStateTuple(out_states_c, out_states_h)
#
#    return output, out_states


#def cross_attention(mem_states, query_states, max_mem_steps, max_query_steps, scope):
#    # first calculate the similarity matrix
#    # mem state size --> batch_size, max_mem_enc_steps, 2*dim_hidden
#    # query state size --> batch_size, max_query_enc_steps, 2*dim_hidden
#    # size of simialrity matrix = batch_size, max_context_enc_steps,max_query_enc_steps
#    max_query_steps = tf.shape(query_states)[1] 
#    max_mem_steps = tf.shape(mem_states)[1] 
#    batch_size = self.opt.batch_size
#    similarity_matrix = [] 
#    with tf.variable_scope(scope+'/similarity_matrix', reuse=False):
#        weight = tf.get_variable('weights',[6*enc_hidden_sz,1])
#
#    for i in range(max_mem_steps):
#            repeat_vc = tf.tile(tf.expand_dims(mem_states[:,i],0),[max_query_steps,1,1])
#            repeat_vc = tf.transpose(repeat_vc,[1,0,2])
#            h = tf.concat([repeat_vc,query_states,repeat_vc*query_states],axis=2)
#            score = tf.matmul(h,tf.tile(tf.expand_dims(weight,0),[batch_size,1,1]))
#            similarity_matrix.append(score)
#
#    similarity_matrix = tf.stack(similarity_matrix) # size = max_context_enc_steps,batch_size,max_query_enc_steps, 1
#    similarity_matrix = tf.reshape(similarity_matrix,[max_mem_steps,batch_size,max_query_steps])
#    similarity_matrix = tf.transpose(similarity_matrix,[1,0,2])
#
#
#    '''renormalize attention'''
#
#    query_on_mem_context = tf.matmul(tf.nn.softmax(similarity_matrix),query_states)
#    print ("query on mem", query_on_mem_context)
#    mem_on_query_context = tf.matmul(tf.nn.softmax(tf.transpose(similarity_matrix,[0,2,1])),mem_states)
#    print ("mem on query", mem_on_query_context)
#    #mem_on_query_context = tf.reduce_max(mem_on_query_context,1)
#    #mem_on_query_context = tf.tile(tf.expand_dims(mem_on_query_context,1), [1, max_mem_steps]) 
#    print ("mem on query after reduction and tiling", mem_on_query_context)
#    return query_on_mem_context,mem_on_query_context
