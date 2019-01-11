import random
import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core

from itertools import chain

from tf_utils.utils import get_keep_prob, sparse_cross_entropy_with_probs,\
        sparse_gather_probs
from tf_utils.ops import create_multi_rnn, bidaf_attention,\
    tri_linear_attention, bi_rnn_encoder, concat_with_product,\
    self_attention_encoder, bi_cudnn_rnn_encoder, attention_layer, fuse_gate,\
    rnn_encoder

from tf_utils.pointer_wrapper import AttnPointerWrapper

from data import encode_data, generate_oov_vocab

from commonsense_utils.commonsense_data import COMMONSENSE_REL_LOOKUP

from prototypes import BaseModel

from multiprocessing import Queue

from elmo import BidirectionalLanguageModel, weight_layers, TokenBatcher

class GatedMultiBidaf_ResSA_ELMo_PtrGen_CS_path_cp_rels_MC(BaseModel):
    '''
    multi hop bidaf + residual self attn + ELMo
    '''
    def __init__(self, opts, vocab): # opts = tf.app.flags.FLAGS object
        self.opt = opts
        self.vocab = vocab
        batch_size = self.opt.batch_size
        dropout_rate = self.opt.dropout_rate

        self.emb = tf.get_variable("embeddings", shape=[vocab.size(),
            vocab.emb_size()], initializer=tf.contrib.layers.xavier_initializer())

        # create placeholders
        self.encoder_inputs = tf.placeholder(shape=[batch_size, None],
                dtype=tf.int32, name='encoder_inputs')
        self.encoder_input_lengths = tf.placeholder(shape=[batch_size],
                dtype=tf.int32, name='encoder_input_lengths')
        self.decoder_targets = tf.placeholder(shape=[batch_size, None],
                dtype=tf.int32, name='decoder_targets')
        self.decoder_target_lengths = tf.placeholder(shape=[batch_size],
                dtype=tf.int32, name='decoder_target_lengths')
        self.decoder_inputs = tf.placeholder(shape=[batch_size, None],
                dtype=tf.int32, name='decoder_inputs')
        self.decoder_input_lengths = tf.placeholder(shape=[batch_size],
                dtype=tf.int32, name='decoder_input_lengths')

        self.max_decoder_length = tf.placeholder(shape=[],
                dtype=tf.int32, name='max_decoder_length')
        self.max_encoder_length = tf.placeholder(shape=[],
                dtype=tf.int32, name='max_encoder_length')

        self.is_training = tf.placeholder(shape=[], dtype=tf.bool,
                name='is_training')

        self.memory_vectors = tf.placeholder(shape=[batch_size, None],
                dtype=tf.int32, name='memory_vectors')
        self.memory_vector_lengths = tf.placeholder(shape=[batch_size],
                dtype=tf.int32, name='memory_vector_lengths')

        self.oov_size = tf.placeholder(shape=[], dtype=tf.int32, name='oov_size')
        self.memory_vectors_full_vocab = tf.placeholder(shape=[batch_size,None],
                dtype=tf.int32, name='memory_vectors_full_vocab')

        self.memory_elmo_token_ids = tf.placeholder(shape=[batch_size, None],
                dtype=tf.int32)
        self.query_elmo_token_ids = tf.placeholder(shape=[batch_size, None],
                dtype=tf.int32)

        self.commonsense_vectors = tf.placeholder(shape=[batch_size, self.opt.max_num_paths, 16],
                dtype=tf.int32, name='commonsense_vectors')
        self.commonsense_lengths = tf.placeholder(shape=[batch_size],
                dtype=tf.int32, name='commonsense_lengths')
        self.commonsense_path_lengths = tf.placeholder(shape=[batch_size, self.opt.max_num_paths],
                dtype=tf.int32, name='commonsense_triple_lengths')


        self.elmo_batcher = TokenBatcher(self.opt.elmo_vocab_file)

        self.keep_prob = get_keep_prob(dropout_rate, self.is_training)

        self.oov_dicts = Queue()

        self.elmo_bilm = BidirectionalLanguageModel(
                self.opt.elmo_options_file,
                self.opt.elmo_weight_file,
                use_character_inputs=False,
                embedding_weight_file=self.opt.elmo_token_embedding_file)


        self.graph_built = False
        self.train_op_added = False

    def build_graph(self):
        batch_size = self.opt.batch_size


        elmo_bilm = self.elmo_bilm
        context_elmo_embeddings_op = elmo_bilm(self.memory_elmo_token_ids)
        query_elmo_embeddings_op = elmo_bilm(self.query_elmo_token_ids)

        with tf.variable_scope("elmo_encodings_input"):
            elmo_context_input = weight_layers('input',
                    context_elmo_embeddings_op,
                    l2_coef=0.0)['weighted_op']

            context_len = tf.shape(self.memory_vectors)[1]
            elmo_context_input = elmo_context_input[:, :context_len]


        with tf.variable_scope("elmo_encodings_input", reuse=True):
            elmo_query_input = weight_layers('input',
                    query_elmo_embeddings_op,
                    l2_coef=0.0)['weighted_op']

            query_len = tf.shape(self.encoder_inputs)[1]
            elmo_query_input = elmo_query_input[:, :query_len]

        print("ELMo shapes:")
        print(elmo_context_input.get_shape().as_list())
        print(elmo_query_input.get_shape().as_list())

        with tf.device("/cpu:0"):
            with tf.variable_scope("embedding"):
                embedded_input_seq = tf.nn.embedding_lookup(self.emb,
                        self.encoder_inputs)
                embedded_dec_input_seq = tf.nn.embedding_lookup(self.emb,
                        self.decoder_inputs)
                embedded_dec_target_seq = tf.nn.embedding_lookup(self.emb,
                        self.decoder_targets)
                embedded_memory_vectors = tf.nn.embedding_lookup(self.emb,
                        self.memory_vectors)
                embedded_commonsense_vectors = tf.nn.embedding_lookup(self.emb,
                        self.commonsense_vectors)

        enc_hidden_sz = self.opt.hidden_size_encoder
        enc_num_layers = self.opt.num_layers_encoder

        # add elmo
        embedded_memory_vectors = tf.concat([embedded_memory_vectors,
            elmo_context_input], -1)
        embedded_input_seq = tf.concat([embedded_input_seq,
            elmo_query_input], -1)

        mem_rep = embedded_memory_vectors

        print(mem_rep.get_shape().as_list())

        for i in range(self.opt.num_attn_hops):
            with tf.variable_scope("attn_layer_%d" % i):
                with tf.variable_scope("mem_encoder"):
                    mem_rep, _ = bi_cudnn_rnn_encoder(
                            'lstm',
                            enc_hidden_sz,
                            enc_num_layers,
                            self.opt.dropout_rate,
                            mem_rep,
                            self.memory_vector_lengths,
                            self.is_training)

                with tf.variable_scope("ques_encoder"):
                    ques_inp, _ = bi_cudnn_rnn_encoder(
                            'lstm',
                            enc_hidden_sz,
                            enc_num_layers,
                            self.opt.dropout_rate,
                            embedded_input_seq,
                            self.encoder_input_lengths,
                            self.is_training)

                # attend
                mem_rep = bidaf_attention(mem_rep, ques_inp,
                        self.memory_vector_lengths,
                        self.encoder_input_lengths,
                        tri_linear_attention)

                print(mem_rep.get_shape().as_list())

                num_units = mem_rep.get_shape().as_list()[-1]

                with tf.variable_scope("commonsense_encoder"):
                    #commonsense size : [batch, max_num_paths, lens, embed]
                    embedded_commonsense_vectors = tf.reshape(
                            embedded_commonsense_vectors,
                            [self.opt.batch_size, self.opt.max_num_paths, -1])
 
                    #project

                    cs_projected = tf.layers.dense(
                            inputs=embedded_commonsense_vectors,
                            units=num_units,
                            activation=tf.nn.relu,
                            name='commonsense_proj')

                    print "projected", cs_projected

                    attn_mem_cs_rep = attention_layer(mem_rep, cs_projected,
                            self.memory_vector_lengths,
                            self.commonsense_lengths,
                            tri_linear_attention, 'memory2commonsense')

                    attn_output_proj = tf.layers.dense(
                            inputs = attn_mem_cs_rep,
                            units=num_units,
                            activation=tf.nn.relu,
                            name='cs_attn_output_proj')

                    mem_rep = fuse_gate(mem_rep, attn_output_proj)


        with tf.variable_scope("res_self_attn"):
            units = mem_rep.get_shape().as_list()[-1]
            print(units)

            mem_proj = tf.layers.dense(
                    inputs=mem_rep,
                    units=units,
                    activation=tf.nn.relu,
                    name="self_attn_input_proj")

            print(mem_proj.get_shape().as_list())

            with tf.variable_scope("input_proj"):
                self_attn_mem_input, _ = bi_cudnn_rnn_encoder(
                        'lstm',
                        enc_hidden_sz,
                        enc_num_layers,
                        self.opt.dropout_rate,
                        mem_proj,
                        self.memory_vector_lengths,
                        self.is_training)

            self_attn_mem = self_attention_encoder(
                    x = self_attn_mem_input,
                    sim_func = tri_linear_attention,
                    mask = self.memory_vector_lengths,
                    merge_function = concat_with_product)

            print(self_attn_mem.get_shape().as_list())

            with tf.variable_scope("output_proj"):
                self_attn_output_proj, _ = bi_cudnn_rnn_encoder(
                        'lstm',
                        units / 2,
                        enc_num_layers,
                        self.opt.dropout_rate,
                        self_attn_mem,
                        self.memory_vector_lengths,
                        self.is_training)

            mem_rep = mem_rep + self_attn_output_proj
            
            print(mem_rep.get_shape().as_list())

        sos_id = self.vocab.start_token_id
        eos_id = self.vocab.end_token_id

        dec_hidden_sz = self.opt.hidden_size_encoder 
        dec_num_layers = self.opt.num_layers_decoder

        train_helper = tf.contrib.seq2seq.TrainingHelper(
                embedded_dec_input_seq,
                self.decoder_input_lengths)

        pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.emb,
                start_tokens = tf.fill([batch_size], sos_id),
                end_token = -1) # XXX hack here to allow correct loss #eos_id)

        def decode(helper, scope, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                attention_over_context = tf.contrib.seq2seq.BahdanauAttention(
                        num_units = self.opt.decoder_attn_size,
                        memory = mem_rep,
                        memory_sequence_length = self.memory_vector_lengths)

                decoder_cell = create_multi_rnn('basic_lstm', dec_hidden_sz,
                        dec_num_layers, self.keep_prob)

                projection_layer = layers_core.Dense(
                        self.vocab.size(),
                        use_bias=True,
                        name='output_projection')

                decoder_cell = AttnPointerWrapper(
                        cell=decoder_cell,
                        attention_mechanism=attention_over_context,
                        output_layer=projection_layer,
                        max_oovs=self.opt.max_oovs,
                        batch_size=self.opt.batch_size,
                        memory_full_vocab=self.memory_vectors_full_vocab,
                        attention_layer_size = self.opt.decoder_attn_size / 2,
                        alignment_history = True,
                        output_combined_distribution=True,
                        unk_id=self.vocab.unk_token_id)

                decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell = decoder_cell,
                        helper = helper,
                        initial_state = decoder_cell.zero_state(
                            batch_size = self.opt.batch_size,
                            dtype = tf.float32))

                outputs = tf.contrib.seq2seq.dynamic_decode(
                        decoder = decoder,
                        output_time_major = False,
                        maximum_iterations = self.max_decoder_length)

                return outputs

        train_outputs, train_state, self.train_final_lengths = decode(train_helper, 'decode')

        train_logits = tf.transpose(train_state.final_dist_history.stack(), 
                [1, 0, 2])


        output_mask = tf.sequence_mask(self.decoder_target_lengths,
                dtype=tf.float32, maxlen=self.max_decoder_length)

        logical_mask = tf.sequence_mask(self.decoder_target_lengths,
                maxlen=self.max_decoder_length)

        pred_probs = tf.contrib.seq2seq.sequence_loss(
                logits=train_logits,
                targets=self.decoder_targets,
                weights=output_mask,
                softmax_loss_function=sparse_gather_probs,
                average_across_timesteps=False,
                average_across_batch=False)

        # because we want to reduce product, masked part should all be 1
        # since they are set to 0 in sequence_loss, we add 1 to those things
        pred_probs = pred_probs + tf.cast(
                tf.logical_not(logical_mask), tf.float32)
        self.preds = tf.reduce_sum(tf.log(pred_probs), 1) # multiply across timesteps
        # use logsum to prevent instability
        print(self.preds.get_shape().as_list())

        self.loss = tf.contrib.seq2seq.sequence_loss(
                logits=train_logits,
                targets=self.decoder_targets,
                weights=output_mask,
                softmax_loss_function=sparse_cross_entropy_with_probs)

        self.graph_built = True

    def add_train_op(self):
        opt = tf.train.AdamOptimizer(self.opt.learning_rate)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.loss, tvars)
        clipped_grads, norm = tf.clip_by_global_norm(
                grads, self.opt.clipping_threshold)

        self._train_op = opt.apply_gradients(
                zip(clipped_grads, tvars))

        self.saver = tf.train.Saver(
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),
                max_to_keep=None)

        self.train_op_added = True

    def encode(self, batch, is_train):
        opt = self.opt
        batch_size = opt.batch_size

        questions = map(lambda item: item['ques'], batch)
        answers = map(lambda item: item['answer1'] if random.getrandbits(1) else
                item['answer2'], batch)
        summaries = map(lambda item: item['summary'], batch)
        commonsense = map(lambda item: item['commonsense'], batch)
        
        max_ques_len = min(opt.max_iterations, max(map(len, questions)))

        max_answer_len = 1 + min(opt.max_target_iterations, max(map(len, answers)))

        max_summary_len = min(opt.max_context_iterations, max(map(len, summaries)))

        max_commonsense_path_len = 16
        max_commonsense_len = opt.max_num_paths
        
        encoded_questions, encoded_question_len = encode_data(
                questions, max_ques_len, self.vocab)
        encoded_answers_inp, encoded_answers_inp_len = encode_data(
                answers, max_answer_len, self.vocab, add_sos=True)
        assert encoded_answers_inp.shape[1] == max_answer_len, "%d %d" % (
            encoded_answers_inp.shape[1], max_answer_len)
        assert max(encoded_answers_inp_len) <= max_answer_len, "%d %d" % (
            max(encoded_answers_inp_len, max_answer_len))
        encoded_summary, encoded_summary_len = encode_data(
                summaries, max_summary_len, self.vocab)

        summary_oovs, oov_count = generate_oov_vocab(summaries, self.vocab,
                max_oovs=opt.max_oovs)

        encoded_summary_full_vocab, _ = encode_data(
                summaries, max_summary_len, self.vocab, oov_vocab=summary_oovs)

        encoded_answers_tgt, encoded_answers_tgt_len = encode_data(
                answers, max_answer_len, self.vocab, oov_vocab=summary_oovs,
                add_eos=True)
        assert encoded_answers_tgt.shape[1] == max_answer_len, "%d %d" % (
            encoded_answers_tgt.shape[1], max_answer_len)
        assert max(encoded_answers_tgt_len) <= max_answer_len, "%d %d" % (
            max(encoded_answers_tgt_len, max_answer_len))

        encoded_commonsense, encoded_commonsense_len, encoded_path_len =\
            encode_data(commonsense, max_commonsense_len, self.vocab,
                    path_len=max_commonsense_path_len)

        self.oov_dicts.put(summary_oovs)
        # ELMo inputs
        elmo_context_ids = self.elmo_batcher.batch_sentences(summaries)
        elmo_question_ids = self.elmo_batcher.batch_sentences(questions)

        feed_dict = {
            self.encoder_inputs: encoded_questions, 
            self.encoder_input_lengths: encoded_question_len, 
            self.decoder_targets: encoded_answers_tgt, 
            self.decoder_target_lengths: encoded_answers_tgt_len, 
            self.decoder_inputs: encoded_answers_inp,
            self.decoder_input_lengths: encoded_answers_tgt_len, 
            self.max_decoder_length: max_answer_len, 
            self.max_encoder_length: max_ques_len, 
            self.is_training: is_train, 
            self.memory_vectors: encoded_summary, 
            self.memory_vector_lengths: encoded_summary_len,
            self.oov_size: oov_count,
            self.memory_vectors_full_vocab: encoded_summary_full_vocab,
            self.memory_elmo_token_ids: elmo_context_ids,
            self.query_elmo_token_ids: elmo_question_ids,
            self.commonsense_vectors: encoded_commonsense, 
            self.commonsense_path_lengths: encoded_path_len, 
            self.commonsense_lengths: encoded_commonsense_len}

        return feed_dict

    def get_batch_oov(self):
        return self.oov_dicts.get(timeout=1)

    def train_step(self, sess, fd):
        if not self.graph_built:
            raise ValueError("Graph is not built yet!")
        if not self.train_op_added:
            raise ValueError("Train ops have not been added yet!")

        _, loss = sess.run([self._train_op, self.loss], feed_dict=fd)
        return loss

    def eval(self, sess, fd):
        if not self.graph_built:
            raise ValueError("Graph is not built yet!")
        if not self.train_op_added:
            raise ValueError("Train ops have not been added yet!")

        loss = 0
        preds = sess.run(self.preds, feed_dict=fd)

        return loss, preds

    def save_to(self, sess, path):
        if not self.graph_built:
            raise ValueError("Graph is not built yet!")
        if not self.train_op_added:
            raise ValueError("Train ops have not been added yet!")

        self.saver.save(sess, path)

    def restore_from(self, sess, path):
        if not self.graph_built:
            raise ValueError("Graph is not built yet!")
        if not self.train_op_added:
            raise ValueError("Train ops have not been added yet!")

        self.saver.restore(sess, path)
