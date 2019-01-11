from prototypes import Vocab
import numpy as np
import pickle

import h5py

class GenModelVocab(Vocab):
    '''
    vocab object for generative models (include pad/eos/sos)
    '''

    PAD = 'PAD'
    BOS = 'BOS'
    EOS = 'EOS'
    UNK = 'UNK'

    def __init__(self, w_list, emb_size, forced_word_list=[], cs = [],threshold=None):
        if threshold is not None:
            filtered_w = {}
            for w in w_list.keys():
                if w_list[w] >= threshold:
                    filtered_w[w] = w_list[w]

            w_list = filtered_w

        cs_list = {}
        for c in cs:
            if (not c in w_list) and (not c in forced_word_list):
                print "c", c
                cs_list[c] = c

        w_list = w_list.keys() + cs_list.keys() + forced_word_list 

        self.word_list = [self.PAD, self.BOS, self.EOS, self.UNK] + w_list
        self._size = len(self.word_list)
        self._emb_size = emb_size
        self._word2id = {}
        for i, w in enumerate(self.word_list):
            self._word2id[w] = i

        self.start_token_id = self._word2id[self.BOS]
        self.end_token_id = self._word2id[self.EOS]
        self.unk_token_id = self._word2id[self.UNK]

    def size(self):
        return self._size

    def emb_size(self):
        return self._emb_size

    def word2id(self, w):
        if w in self._word2id:
            return self._word2id[w]
        else:
            return self._word2id[self.UNK]

    def id2word(self, id):
        return self.word_list[id]

    def is_oov(self, w):
        return w not in self._word2id

class GloVEVocab(Vocab):
    '''
    very similar to vocab class, but uses glove vectors for initialization
    '''

    PAD = 'PAD'
    BOS = 'BOS'
    EOS = 'EOS'
    UNK = 'UNK'

    def __init__(self, w_list, glove_file, forced_word_list=[], threshold=None):
        if threshold is not None:
            filtered_w = {}
            for w in w_list.keys():
                if w_list[w] >= threshold:
                    filtered_w[w] = w_list[w]

            w_list = filtered_w

        w_list = w_list.keys() + forced_word_list 

        self.word_list = [self.PAD, self.BOS, self.EOS, self.UNK] + w_list
        self._size = len(self.word_list)
        self._word2id = {}
        for i, w in enumerate(self.word_list):
            self._word2id[w] = i

        self.start_token_id = self._word2id[self.BOS]
        self.end_token_id = self._word2id[self.EOS]
        self.unk_token_id = self._word2id[self.UNK]

        emb = [None for _ in self.word_list]

        r = np.random.RandomState(0)

        contains = 0
        dim = None
        with open(glove_file, 'r') as vocab_f:
            for line in vocab_f:

                row = line.strip().split(' ')

                word = row[0]
                embeddings = row[1:]
                if dim is None:
                    dim = len(embeddings)
                    print("Using dimension %d" % dim)
                elif len(embeddings) != dim:
                    continue

                if word in self._word2id:
                    emb[self._word2id[word]] = np.asarray(embeddings,
                            dtype=np.float32)
                    contains += 1

        self._emb_size = dim

        for i in range(len(emb)):
            if emb[i] is None:
                #emb[i] = np.zeros(dim, dtype=np.float32)
                emb[i] = -0.001 + 0.002 * r.rand(dim)

        print("Could find %d / %d words, rest initialized to random" %
                (contains, len(self.word_list)))

        self._embeddings = np.stack(emb, 0)

        print(self._embeddings.shape)
        print(self._embeddings.dtype)

        #h5f = h5py.File("out_pp2/new_emb.h5", 'w')
        #h5f.create_dataset('embeddings', data=self._embeddings)
        #h5f.close()
        #exit(0)

    def word2id(self, word):
        if word not in self._word2id:
            return self._word2id[self.UNK]
        return self._word2id[word]

    def id2word(self, id):
        return self._words[id]

    def word2embeddings(self, word):
        wid = self.word2id(word)
        return self.embeddings[wid]

    def embeddings(self):
        return self._embeddings

    def size(self):
        return self._size

    def embed_dim(self):
        return self._embedding_d

    def is_oov(self, w):
        return w not in self._word2id


def save_vocab(vocab, loc):
    with open(loc, 'w') as out_f:
        pickle.dump(vocab, out_f)

def restore_vocab(loc):
    with open(loc, 'r') as in_f:
        return pickle.load(in_f)


def encode_commonsense(data, max_len, max_triple_len, vocab, oov_vocab):
    '''
    encode commonsense into np arrays
    data: list of list of "triples", batch_size x max_time_step x max_triple_step
    max_len: int, commonsense cutoff length
    max_triple: int, triple cutoff length
    vocab: vocabulary object
    '''
    batch_size = len(data)

    if oov_vocab is None:
        oov_vocab = {}

    cutoff = max_len

    encoded_data = np.zeros(shape=[batch_size, max_len, max_triple_len], dtype=np.int32)
    encoded_data_len = np.zeros(shape=[batch_size], dtype=np.int32)
    encoded_triple_len = np.zeros(shape=[batch_size, max_len], dtype=np.int32)

    for i, ex in enumerate(data):
        for j, cs in enumerate(ex):
            if j >= cutoff:
                break
            for k, w in enumerate(cs): 
                if w in oov_vocab:
                    wid = oov_vocab[w]
                else:
                    wid = vocab.word2id(w)

                encoded_data[i, j, k] = wid
            encoded_triple_len[i, j] = len(cs)
        encoded_data_len[i] = min(cutoff, len(ex))

        if encoded_data_len[i] == 0: # this will throw an error in ptr
            encoded_data_len[i] = 1
            encoded_triple_len[i, 0] = 3
            encoded_data[i, 0, 0] = vocab.word2id(vocab.PAD)
            encoded_data[i, 0, 1] = vocab.word2id(vocab.PAD)
            encoded_data[i, 0, 2] = vocab.word2id(vocab.PAD)

    return encoded_data, encoded_data_len, encoded_triple_len

def encode_commonsense_paths(data, max_len, max_path_len, vocab, oov_vocab):
    '''
    encode commonsense into np arrays
    data: list of list of "triples", batch_size x max_time_step x max_triple_step
    max_len: int, commonsense cutoff length
    max_triple: int, triple cutoff length
    vocab: vocabulary object
    '''
    batch_size = len(data)

    if oov_vocab is None:
        oov_vocab = {}

    cutoff = max_len

    encoded_data = np.zeros(shape=[batch_size, max_len, max_path_len], dtype=np.int32)
    encoded_data_len = np.zeros(shape=[batch_size], dtype=np.int32)
    encoded_path_len = np.zeros(shape=[batch_size, max_len], dtype=np.int32)

    for i, ex in enumerate(data):
        for j, cs in enumerate(ex):
            if j >= cutoff:
                break
            for k, w in enumerate(cs): 
                if k >= max_path_len:
                    break

                if w in oov_vocab:
                    wid = oov_vocab[w]
                else:
                    wid = vocab.word2id(w)

                encoded_data[i, j, k] = wid
            encoded_path_len[i, j] = len(cs)
        encoded_data_len[i] = min(cutoff, len(ex))

    return encoded_data, encoded_data_len, encoded_path_len



def encode_data(data, max_len, vocab, oov_vocab=None,
        preprocess=lambda w: w.lower(), add_sos=False, add_eos=False, triple_len=-1,
        path_len=-1):
    '''
    encode data into np arrays
    data: list of list of words, batch_size x max_time_step
    max_len: int, cutoff length
    vocab: vocabulary object
    oov_vocab: extra vocab object that include oov
    preprocess: function on each word, default is lower-case it
    add_sos: add start of sentence?
    add_eos: add end of sentence?
    '''
    if triple_len > 0:
       return encode_commonsense(data, max_len, triple_len, vocab, oov_vocab)
    elif path_len > 0:
       return encode_commonsense_paths(data, max_len, path_len, vocab, oov_vocab)



    if preprocess is None:
        preprocess = lambda w: w

    if oov_vocab is None:
        oov_vocab = {}

    batch_size = len(data)

    if add_sos: # we add sos by concat at the end
        max_len -= 1

    cutoff = max_len

    if add_eos:
        cutoff -= 1

    encoded_data = np.zeros(shape=[batch_size, max_len], dtype=np.int32)
    encoded_data_len = np.zeros(shape=[batch_size], dtype=np.int32)

    for i, ex in enumerate(data):
        for j, w in enumerate(ex):
            if j >= cutoff:
                break

            try:
                w = preprocess(w)
            except:
                print(w)
                exit(1)

            if vocab.is_oov(w):
                if w in oov_vocab:
                    wid = oov_vocab[w]
                else:
                    wid = vocab.word2id(w)
            else:
                wid = vocab.word2id(w)

            encoded_data[i, j] = wid

        encoded_data_len[i] = min(cutoff, len(ex))
        if add_eos:
            encoded_data[i, min(cutoff, len(ex))] = vocab.word2id(vocab.EOS)
            encoded_data_len[i] += 1

    if add_sos:
        sos = np.zeros(shape=[batch_size, 1], dtype=np.int32)
        sos.fill(vocab.word2id(vocab.BOS))
        encoded_data = np.concatenate((sos, encoded_data), axis=1)
        encoded_data_len += 1

    return encoded_data, encoded_data_len


def generate_oov_vocab(data, vocab, preprocess=lambda w: w.lower(),
        max_oovs=0):
    '''
    generate a dictionary mapping oov words in data to new word_ids
    '''

    wid_prefix = vocab.size()
    oov_count = 0

    oovs = {}

    for ex in data:
        for w in ex:
            w = preprocess(w)
            if vocab.is_oov(w) and w not in oovs:
                oovs[w] = wid_prefix + oov_count
                oov_count += 1

                if max_oovs > 0 and oov_count == max_oovs:
                    return oovs, oov_count

    return oovs, oov_count

def translate_spans(prediction, item):
    summary = item['summary']
    start_idx, end_idx = prediction

    return summary[start_idx : end_idx + 1]

def translate(sentence, vocab, oovs, clean = True, markup = False):
    oov_id2word = dict([(v, k) for k, v in oovs.items()])
    punc_list = [vocab.word2id('.'), vocab.word2id('!'), vocab.word2id('?')]
    sentence = np.asarray(sentence)
    response = []
    for ind, w_id in enumerate(sentence):
        if clean and w_id == vocab.end_token_id or w_id == 0:
            return response

        if clean and ind + 1 != len(sentence) and \
              sentence[ind + 1] == vocab.end_token_id and \
              sentence[ind] in punc_list:
            return response

        if w_id in oov_id2word:
            if markup:
                response.append("__" + oov_id2word[w_id] + "__")
            else:
                response.append(oov_id2word[w_id])
        else:
            response.append(vocab.id2word(w_id))
    return response 

def chunk_data(data, chunks, max_len_total, method = 'greedy'):
    '''
    divide data into chunks
    data: list of list of words
    chunks: # of chunks to split data into
    max_len_total: total cutoff length (chunk length = max_len_total / chunks)
    method: method of chunking, either greedy or even
        greedy: fill up first chunk, then move to second, etc.
        even: fill each chunk as full as possible
    '''

    chunk_size = int(max_len_total / float(chunks) + 0.5)

    chunked_data = []

    for ex in data:
        assert type(ex[0]) == str

        if method == 'greedy':
            chunked_ex = []

            cursor = 0

            for _ in range(chunks):
                if cursor > len(ex):
                    chunked_ex.append([])

                chunked_ex.append(ex[cursor : cursor + chunk_size])
                cursor += chunk_size

        elif method == 'even':
            raise NotImplementedError()

        else:
            raise ValueError()

        chunked_data.append(chunked_ex)

    return chunk_size, chunked_data
