import collections
import functools
import math

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

from tensorflow.contrib.rnn import LSTMStateTuple

from tensorflow.contrib.seq2seq import AttentionMechanism

from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import\
        _prepare_memory

_zero_state_tensors = rnn_cell_impl._zero_state_tensors

class PointerWrapperState(
    collections.namedtuple("PointerWrapperState",
         ("cell_state", "attention", "time", "alignment_history",
             "p_gen_history", "vocab_dist_history", "copy_dist_history",
             "final_dist_history"))):
    def clone(self, **kwargs):
        return super(PointerWrapperState, self)._replace(**kwargs)


def _compute_pgen_and_attention(values, cell_output, cell_state,
        input, attention_layer,
        first_lv_sim_func, second_lv_sim_func):

    print(values.get_shape().as_list())
    b = values.get_shape().as_list()[0]
    k = values.get_shape().as_list()[1]
    h = values.get_shape().as_list()[-1]
    print(h)

    mem_reshaped = tf.reshape(values, [b * k, -1, h])

    query_expand_dims = tf.expand_dims(cell_output, 1) # [b x 1 x h]

    print(query_expand_dims.get_shape().as_list())

    query_expand_dims_tiled = tf.tile(tf.expand_dims(query_expand_dims, 1),
            [1, k, 1, 1])
    print(query_expand_dims_tiled.get_shape().as_list())
    q_dim = query_expand_dims_tiled.get_shape().as_list()[-1]
    query_expand_dims_tiled = tf.reshape(query_expand_dims_tiled, [b*k, -1,
        q_dim])

    print(query_expand_dims_tiled.get_shape().as_list())

    with tf.variable_scope("first_lv_sim"):
        attn_logits = first_lv_sim_func(query_expand_dims_tiled, mem_reshaped)
    # [(b*k), n]

    attn_logits_reshaped = tf.reshape(attn_logits, [b, k, -1])
    alignments = tf.nn.softmax(attn_logits_reshaped, -1) # [b x k x n]

    expanded_alignments = tf.expand_dims(attn_logits, 1) # [(b*k) x 1 x n]
    w_lv_context = tf.matmul(expanded_alignments, mem_reshaped) # [(b*k) x 1 x h]
    w_lv_context = tf.squeeze(w_lv_context, 1) # [(b*k) x h]

    w_lv_context_expanded = tf.reshape(w_lv_context, [b, k, h])

    with tf.variable_scope("second_lv_sim"):
        second_attn_logits = second_lv_sim_func(query_expand_dims,
            w_lv_context_expanded) # [b x k]
    second_probs = tf.nn.softmax(second_attn_logits, -1) # [b x k]
    expanded_probs = tf.expand_dims(second_probs, 1) # [b x 1 x k]

    context = tf.matmul(expanded_probs, w_lv_context_expanded) # [b x 1 x h]
    context = tf.squeeze(context, 1) # [b x h]

    ptr_inputs = tf.concat([context, cell_state.c, cell_state.h, input], -1,
            name='ptr_inputs')
    p_gen = tf.layers.dense(
            ptr_inputs, 1, activation=tf.sigmoid, use_bias=True,
            name='pointer_generator')

    if attention_layer is not None:
        attention = attention_layer(array_ops.concat([cell_output, context], 1))
    else:
        attention = context

    return attention, alignments, p_gen
        
def _calc_copy_dist(attn_dist, batch_size, vocab_size, num_source_OOVs,
        enc_batch_extended_vocab):
    
    # assign probabilities from copy distribution
    # into correspodning positions in extended_vocab_dist
    
    # to do this, we need to use scatter_nd
    # scatter_nd (in this case) requires two numbers
    # one is the index in batch-dimension
    # the other is the index in vocab-dimension
    # So first, we create a batch-matrix like:
    # [[1, 1, 1, 1, 1, ...],
    #  [2, 2, 2, 2, 2, ...],
    #  [...]
    #  [N, N, N, N, N, ...]]
    
    # [1, 2, ..., N]
    # to [[1], [2], ..., [N]]
    # and finally to the final shape
    enc_seq_len = tf.shape(enc_batch_extended_vocab)[1]
    batch_nums = tf.range(0, limit=batch_size)
    batch_nums = tf.expand_dims(batch_nums, 1)
    batch_nums = tf.tile(batch_nums, [1, enc_seq_len])
    
    # stick together batch-dim and index-dim
    indices = tf.stack((batch_nums, enc_batch_extended_vocab), axis=2)
    extended_vsize = vocab_size + num_source_OOVs
    scatter_shape = [batch_size, extended_vsize]
    # scatter the attention distributions
    # into the word-indices
    P_copy_projected = tf.scatter_nd(
        indices, attn_dist, scatter_shape)

    return P_copy_projected

def _calc_final_dist(vocab_dist, copy_dist, p_gen,
            batch_size, vocab_size, num_source_OOVs):
    '''
    calculate the final distribution w/ ptr net (one step)
    vocab_dist: predicted vocab distribution, tensor, shape b x v_size
    attn_dist: predicted attn distribution, tensor, shape b x v_size_ext
    p_gen: generation probability, tensor, shape b
    batch_size: int, batch size
    vocab_size: int, v_size
    num_source_OOVs: int, # of oovs
    enc_batch_extended_vocab: encoded context w/ extra vocabulary, e.g. replace
        all UNK with actual oov indices
    '''
    # P(gen) x P(vocab)
    weighted_P_vocab = p_gen * vocab_dist
    # (1 - P(gen)) x P(attention)
    weighted_P_copy = (1 - p_gen) * copy_dist
    
    # get the word-idx for all words
    extended_vsize = vocab_size + num_source_OOVs
    # placeholders to OOV words
    extra_zeros = tf.zeros((batch_size, num_source_OOVs))
    # this distribution span the entire words
    weighted_P_vocab_extended = tf.concat(
        axis=1, values=[weighted_P_vocab, extra_zeros])
   
    # Add the vocab distributions and the copy distributions together
    # to get the final distributions, final_dists is a list length
    # max_dec_steps; each entry is (batch_size, extended_vsize)
    # giving the final distribution for that decoder timestep
    # Note that for decoder timesteps and examples corresponding to
    # a [PAD] token, this is junk - ignore.
    final_dists = weighted_P_vocab_extended + weighted_P_copy

    return final_dists

def _convert_to_output_dist(full_vocab_dist, vocab_size, unk_id):
    '''
    convert a final distribution over the full vocab into an output distribution
        that maps oov probs to unk token

    full_vocab_dist: complete vocab distribution, shape b x (v_size + oov_size)
    vocab_size: int, vocab size
    unk_id: int, unk token id (e.g. things to map oov to)
    '''

    extra_unk_probs = full_vocab_dist[:, vocab_size:] # [b x oov_size]
    extra_unk_probs = tf.reduce_sum(extra_unk_probs, axis=1) # [b]

    batch_size = tf.shape(full_vocab_dist)[0]
    batch_idx = tf.range(0, limit=batch_size)
    unk_idx = tf.fill([batch_size], unk_id)
    scatter_idx = tf.stack((batch_idx, unk_idx), axis=1) # [b x 2]
    scatter_shape = [batch_size, vocab_size]

    unk_dist = tf.scatter_nd(
            scatter_idx, extra_unk_probs, scatter_shape)

    known_vocab_dist = full_vocab_dist[:, :vocab_size] # [b x vocab_size]

    return known_vocab_dist + unk_dist

class HierarchicalAttnPointerWrapper(rnn_cell_impl.RNNCell):
    """Wraps an cell with attention and pointer net
    """

    def __init__(self,
                 cell,
                 memory,
                 memory_sequence_length,
                 output_layer,
                 max_oovs,
                 batch_size,
                 memory_full_vocab,
                 first_lv_sim_func,
                 second_lv_sim_func,
                 attention_layer_size=None,
                 alignment_history=False,
                 cell_input_fn=None,
                 output_attention=False,
                 output_generation_distribution=False,
                 output_copy_distribution=False,
                 output_combined_distribution=True,
                 initial_cell_state=None,
                 unk_id=None,
                 name=None):

        super(HierarchicalAttnPointerWrapper, self).__init__(name=name)
        if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
            raise TypeError(
                    "cell must be an RNNCell, saw type: %s" % type(cell).__name__)

        self._is_multi = False

        if cell_input_fn is None:
            cell_input_fn = (
                    lambda inputs, attention: array_ops.concat([inputs, attention], -1))
        else:
            if not callable(cell_input_fn):
                raise TypeError(
                        "cell_input_fn must be callable, saw type: %s"
                        % type(cell_input_fn).__name__)

        if attention_layer_size is not None:
            attention_layer_sizes = tuple(
                    attention_layer_size
                    if isinstance(attention_layer_size, (list, tuple))
                    else (attention_layer_size,))
            if len(attention_layer_sizes) != 1:
                raise ValueError(
                        "If provided, attention_layer_size must contain exactly one "
                        "integer per attention_mechanism, saw: %d vs 1"
                        % (len(attention_layer_sizes)))

            self._attention_layers = tuple(
                    layers_core.Dense(
                            attention_layer_size, name="attention_layer", use_bias=False)
                    for attention_layer_size in attention_layer_sizes)
            self._attention_layer_size = sum(attention_layer_sizes)
        else:
            self._attention_layers = None
            self._attention_layer_size = memory.get_shape()[-1].value

        self._cell = cell
        self._cell_input_fn = cell_input_fn
        self._output_attention = output_attention
        self._output_generation_distribution = output_generation_distribution
        self._output_copy_distribution = output_copy_distribution
        self._output_combined_distribution = output_combined_distribution
        self._unk_id = unk_id
        self._alignment_history = alignment_history
        self._output_layer = output_layer
        self._max_oovs = max_oovs
        self._batch_size = batch_size

        [self._b, self._k, _, h] = memory.get_shape().as_list()
        #self._k = tf.shape(memory)[1].value
        #self._b = tf.shape(memory)[0].value
        #h = tf.shape(memory)[-1].value

        b = self._b
        k = self._k

        mem_reshaped = tf.reshape(memory, [b * k, -1, h])
        print(mem_reshaped.get_shape().as_list())
        mem_mask_reshaped = tf.reshape(memory_sequence_length, [-1])

        self._memory = tf.reshape(
                _prepare_memory(mem_reshaped, mem_mask_reshaped, False),
                [b, k, -1, h])
        self._memory_full_vocab = memory_full_vocab

        self._attention_mechanisms = [None] # placeholder

        with tf.variable_scope("first_lv_attn"):
            self._first_lv_sim_func = first_lv_sim_func

        with tf.variable_scope("second_lv_attn"):
            self._second_lv_sim_func = second_lv_sim_func

        if self._output_combined_distribution or \
                self._output_generation_distribution or \
                self._output_copy_distribution or \
                self._output_attention:
            assert self._output_combined_distribution ^\
                self._output_generation_distribution ^\
                self._output_copy_distribution ^\
                self._output_attention, "Can only output one type!"

        if self._output_combined_distribution or self._output_copy_distribution:
            assert self._unk_id is not None

        with ops.name_scope(name, "AttnPointerWrapperInit"):
            if initial_cell_state is None:
                self._initial_cell_state = None
            else:
                final_state_tensor = nest.flatten(initial_cell_state)[-1]
                state_batch_size = (
                        final_state_tensor.shape[0].value
                        or array_ops.shape(final_state_tensor)[0])
                error_message = (
                        "When constructing AttnPointerWrapper %s: " % self._base_name +
                        "Non-matching batch sizes between the memory "
                        "(encoder output) and initial_cell_state.  Are you using "
                        "the BeamSearchDecoder?  You may need to tile your initial state "
                        "via the tf.contrib.seq2seq.tile_batch function with argument "
                        "multiple=beam_width.")
                with ops.control_dependencies(
                        self._batch_size_checks(state_batch_size, error_message)):
                    self._initial_cell_state = nest.map_structure(
                            lambda s: array_ops.identity(s, name="check_initial_cell_state"),
                            initial_cell_state)

    def _batch_size_checks(self, batch_size, error_message):
        #return True
        return [check_ops.assert_equal(batch_size, batch_size,
            message=error_message)
                for attention_mechanism in self._attention_mechanisms]

    def _item_or_tuple(self, seq):
        """Returns `seq` as tuple or the singular element.

        Which is returned is determined by how the AttentionMechanism(s) were passed
        to the constructor.

        Args:
            seq: A non-empty sequence of items or generator.

        Returns:
             Either the values in the sequence as a tuple if AttentionMechanism(s)
             were passed to the constructor as a sequence or the singular element.
        """
        t = tuple(seq)
        if self._is_multi:
            return t
        else:
            return t[0]

    @property
    def output_size(self):
        if self._output_combined_distribution or \
                self._output_copy_distribution or \
                self._output_generation_distribution:
            return self._output_layer.units
        if self._output_attention:
            return self._attention_layer_size
        else:
            return self._cell.output_size

    @property
    def state_size(self):
        return PointerWrapperState(
                cell_state=self._cell.state_size,
                time=tensor_shape.TensorShape([]),
                attention=self._attention_layer_size,
                alignment_history=self._item_or_tuple(
                    () for _ in self._attention_mechanisms),
                p_gen_history=self._item_or_tuple(
                    () for _ in self._attention_mechanisms),
                vocab_dist_history=self._output_layer.units,
                copy_dist_history=self._output_layer.units + self._max_oovs,
                final_dist_history=self._output_layer.units + self._max_oovs)    # sometimes a TensorArray

    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            error_message = (
                    "When calling zero_state of AttentionWrapper %s: " % self._base_name +
                    "Non-matching batch sizes between the memory "
                    "(encoder output) and the requested batch size.  Are you using "
                    "the BeamSearchDecoder?  If so, make sure your encoder output has "
                    "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
                    "the batch_size= argument passed to zero_state is "
                    "batch_size * beam_width.")
            with ops.control_dependencies(
                    self._batch_size_checks(batch_size, error_message)):
                cell_state = nest.map_structure(
                        lambda s: array_ops.identity(s, name="checked_cell_state"),
                        cell_state)

            state = PointerWrapperState(
                cell_state=cell_state,
                time=array_ops.zeros([], dtype=dtypes.int32),
                attention=_zero_state_tensors(self._attention_layer_size,
                    batch_size, dtype),
                alignment_history=self._item_or_tuple(
                    tensor_array_ops.TensorArray(dtype=dtype, size=0,
                    dynamic_size=True)
                    if self._alignment_history else ()
                        for _ in self._attention_mechanisms),
                p_gen_history=self._item_or_tuple(
                    tensor_array_ops.TensorArray(dtype=dtype, size=0,
                        dynamic_size=True) for _ in self._attention_mechanisms),
                vocab_dist_history=tensor_array_ops.TensorArray(
                    dtype=tf.float32, size=0, dynamic_size=True),
                copy_dist_history=tensor_array_ops.TensorArray(
                    dtype=tf.float32, size=0, dynamic_size=True),
                final_dist_history=tensor_array_ops.TensorArray(
                    dtype=tf.float32, size=0, dynamic_size=True))

            return state

    def call(self, inputs, state):
        if not isinstance(state, PointerWrapperState):
            raise TypeError("Expected state to be instance of PointerWrapperState. "
                                            "Received type %s instead."  % type(state))

        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        if isinstance(cell_state, LSTMStateTuple):
            last_out_state = cell_state
        else:
            last_out_state = cell_state[-1]

        cell_batch_size = (
                cell_output.shape[0].value or array_ops.shape(cell_output)[0])
        error_message = (
                "When applying AttentionWrapper %s: " % self.name +
                "Non-matching batch sizes between the memory "
                "(encoder output) and the query (decoder output).  Are you using "
                "the BeamSearchDecoder?  You may need to tile your memory input via "
                "the tf.contrib.seq2seq.tile_batch function with argument "
                "multiple=beam_width.")
        with ops.control_dependencies(
                self._batch_size_checks(cell_batch_size, error_message)):
            cell_output = array_ops.identity(
                    cell_output, name="checked_cell_output")

        if self._is_multi:
            previous_pgens = state.p_gen_history
            previous_alignment_history = state.alignment_history
        else:
            previous_alignment_history = [state.alignment_history]
            previous_pgens = [state.p_gen_history]

        previous_vocab_dist_history = state.vocab_dist_history
        print(previous_vocab_dist_history)
        print(cell_output)
        print(self._output_layer)
        vocab_dist = tf.nn.softmax(self._output_layer(cell_output))
        print(vocab_dist)
        vocab_dist_history = previous_vocab_dist_history.write(
                state.time, vocab_dist)
        print("Vocab dist history")
        print(vocab_dist_history)

        all_alignments = []
        all_attentions = []
        all_histories = []
        all_p_gens = []

        this_alignment = None
        this_p_gen = None

        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            attention, alignments, p_gen = _compute_pgen_and_attention(
                    self._memory, cell_output, last_out_state, inputs,
                    self._attention_layers[i]
                        if self._attention_layers
                        else None,
                    self._first_lv_sim_func, self._second_lv_sim_func)

            this_alignment = alignments
            this_p_gen = p_gen

            alignment_history = previous_alignment_history[i].write(
                    state.time, alignments) if self._alignment_history else ()
            p_gen_hist = previous_pgens[i].write(
                    state.time, p_gen)

            all_alignments.append(alignments)
            all_histories.append(alignment_history)
            all_attentions.append(attention)
            all_p_gens.append(p_gen_hist)


        previous_final_dist_history = state.final_dist_history
        previous_copy_dist_history = state.copy_dist_history

        vocab_size = self._output_layer.units

        b = self._batch_size
        full_vocab_reshaped = tf.reshape(self._memory_full_vocab, [b, -1])
        alignments_reshaped = tf.reshape(this_alignment, [b, -1]) # [b x (k*n)]

        copy_dist = _calc_copy_dist(alignments_reshaped, self._batch_size,
                vocab_size, self._max_oovs, full_vocab_reshaped)

        copy_dist_history = previous_copy_dist_history.write(
                state.time, copy_dist)


        final_dist = _calc_final_dist(vocab_dist, copy_dist,
                this_p_gen, self._batch_size, vocab_size, self._max_oovs)

        final_dist_history = previous_final_dist_history.write(
                state.time, final_dist)

        print("Final_dist_history")
        print(final_dist_history)

        attention = array_ops.concat(all_attentions, 1)

        next_state = PointerWrapperState(
                time=state.time + 1,
                cell_state=next_cell_state,
                attention=attention,
                alignment_history=self._item_or_tuple(all_histories),
                p_gen_history=self._item_or_tuple(all_p_gens),
                vocab_dist_history=vocab_dist_history,
                copy_dist_history=copy_dist_history,
                final_dist_history=final_dist_history)

        if self._output_generation_distribution:
            return vocab_dist, next_state
        elif self._output_copy_distribution:
            return (_convert_to_output_dist(copy_dist, vocab_size, self._unk_id),
                    next_state)
        elif self._output_combined_distribution:
            return (_convert_to_output_dist(final_dist, vocab_size,
                    self._unk_id), next_state)
        elif self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state
