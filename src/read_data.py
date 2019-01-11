import os
import cPickle as pickle
#import pickle
import time
import csv
import random
import itertools
import nltk
import math
import numpy as np
import json

import copy

import multiprocessing as mp

from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from commonsense_utils.graph import Commonsense_Graph
from commonsense_utils.commonsense_data import COMMONSENSE_REL_LOOKUP

from collections import defaultdict

import random

from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gc

import commonsense_utils.general as CG

from commonsense_utils.general import sample_relations


class DataSet(object):
    def __init__(self, data, data_type):
        self.data = data 
        self.data_type = data_type
        self.num_examples = self.get_data_size()

    def get_data_size(self):
        return len(self.data) 

    def get_num_batches(self, batch_size):
        return int(math.ceil(self.num_examples/batch_size))

    def get_word_lists(self):
        words = {}

        for i in range(len(self.data)):
            for k,v in self.data[i].items():
                if type(v) != list:
                    continue
                if k == 'commonsense':
                    v = [x_i for x in v for x_i in x]
                if k != 'doc_num':
                    for w in v:
                        if type(w) == list:
                            continue
                        if w.find(" ") == -1:
                            w = w.lower()
                            if w in words:
                                words[w] += 1
                            else:
                                words[w] = 1
                        else:
                            for w in w.split(" "):
                                w = w.lower()
                                if w in words:
                                    words[w] += 1
                                    words[w] = 1
        return words

    def get_by_idxs(self, idxs, batch_size, pad_to_full_batch):
        '''
        return data objects in a list based on the given idxs
        '''
        out = [copy.deepcopy(self.data[i]) for i in idxs]
        if pad_to_full_batch:
            while len(out) < batch_size:
                out.append(out[0])

        assert len(out) == batch_size

        return out

    def get_batches(self, batch_size, shuffle=True, front_heavy=False,
            pad_to_full_batch=False):
        # front_heavy: put long context first, so oov happens early
        # compute # of batches needed
        # pad_to_full_batch: pad until it's a full batch
        if pad_to_full_batch:
            num_batches = int(math.ceil(float(self.num_examples)/batch_size))
        else:
            num_batches = self.num_examples / batch_size

        idx = range(self.num_examples)
        if front_heavy:
            idx = sorted(idx, key=lambda i:
                    len(self.data[i]['summary']) * len(self.data[i]['answer1']),
                    reverse=True)
            #print(len(self.data[idx[0]]['summary']))
        elif shuffle:
            random.shuffle(idx)

        for i in range(num_batches):
            idx_slice = idx[i * batch_size : (i+1) * batch_size]
            yield self.get_by_idxs(idx_slice, batch_size, pad_to_full_batch)

def get_stop_words(total_words, num_stop_words):
    fdist = FreqDist(total_words)
    stop_words = fdist.most_common(num_stop_words)
    stop_words = [t[0] for t in stop_words]
    pronoun_list = ["he", "she", "him", "her", "his", "them", "their", "they"] 
    filtered_stop_words = []
    for p in stop_words:
       if p not in pronoun_list:
           filtered_stop_words.append(p)
    return filtered_stop_words 

def get_stop_words_1(data, num_stop_words):
    total_words = []
    for d in data:
       total_words.extend(d["ques"])
       total_words.extend(d["answer1"])
       for d_i in d["summary"]:
           total_words.extend(d_i)
    fdist = FreqDist(total_words)
    stop_words = fdist.most_common(num_stop_words)
    stop_words = [t[0] for t in stop_words]
    pronoun_list = ["he", "she", "him", "her", "his", "them", "their", "they"] 
    filtered_stop_words = []
    for p in stop_words:
       if p not in pronoun_list:
           filtered_stop_words.append(p)
    return filtered_stop_words 

def extract_commonsense(bundle):
    i, d, stop_words, relations, max_context_iterations, \
        one_hop= bundle
    data_dict = d
    question = list(d['ques'])
    context = lower_all([dij for di in d['summary'] for dij in di])
    if one_hop:
        tree = CG.build_trees_one_hop(relations, question, stop_words, context)
    else:
        tree = CG.build_trees(relations, question, stop_words, context)  
    selected_relations = CG.sample_relations(tree, context) 

    data_dict['commonsense'] = selected_relations

    summ = []
    total_len = 0

    for doc in data_dict['summary']:
        summ.extend(doc)
        total_len += len(doc)
        if total_len > max_context_iterations:
            break

    data_dict['summary'] = summ

    return i, data_dict

def read_wikihop_data_wcs(path, commonsense_file, 
        max_context_iterations, stop_words, tfidf_layer=1, one_hop = False,
        only_follows = False):
    data = parse_raw_wikihop_data(path, tfidf_layer, only_follows)
    print("building relations...")
    relations = CG.get_relations(commonsense_file)
    print("done")

    cs_data = [None for _ in data]
    augmented_data = [(i, d, stop_words, relations,
        max_context_iterations, one_hop) for i, d in enumerate(data)]
    print("building commonsense...")

    pool = mp.Pool(16)

    for i, d in tqdm(pool.imap_unordered(extract_commonsense, augmented_data,
        100), total=len(cs_data)):
        cs_data[i] = d

    pool.close()
    pool.join()

    print("done!")

    return cs_data

def lower_all(a_list):
    '''
    lower everything in a list of strings
    '''

    lowered = map(lambda w: w.lower(), a_list)

    return list(lowered)

def parse_raw_wikihop_data(path, tfidf_layer, only_follows = False):
    data = []

    print("Loading wikihop data from %s..." % path)
    with open(path) as input_json:
        reader = json.load(input_json)
        i = 0
        for id, row in tqdm(enumerate(reader)):
            i += 1
            #if i > 10: 
            #  break
            # we lowercase everything because some mturkers used all caps and others used all lowercase
            #print row
            question = lower_all(row['query'].replace("_", " ").encode('utf-8').split(" "))
            orig_answer = row['answer'].encode("utf-8")
            answer1 = lower_all(nltk.word_tokenize(row['answer'].encode('utf-8')))
            orig_cand = row['candidates']
            cand = [lower_all(nltk.word_tokenize(c)) for c in row['candidates']]
            
            documents = row['supports']

            top_docs = filter_tfidf_wikihop(question, documents, tfidf_layer)
            #top_docs = [p[1] for p in top_docs]
            docs = []
            for d in top_docs:
                try:
                    words = nltk.word_tokenize(d)
                    words = [d_.encode('utf-8') for d_ in words]
                    docs.append(words)
                except UnicodeDecodeError:
                    continue 
            context = []
            for i, c in enumerate(docs):             
                context.append(c)
            data_dict = {
                    'summary': context,
                    'ques': question,
                    'candidates': cand,
                    'orig_cand': orig_cand,
                    'answer1': answer1,
                    'orig_answer': orig_answer,
                    'data_id': row['id'] }

            if only_follows:
                follows = True

                for annotation in row['annotations']:
                    if 'follows' not in annotation:
                        follows = False

                if follows:
                    data.append(data_dict)
            else:
                data.append(data_dict)

    return data

def filter_tfidf_wikihop(ques, paragraphs, tfidf_layer):
    tfidf = TfidfVectorizer(strip_accents="unicode")    
    text = paragraphs
    n_to_select = len(paragraphs)
    
    try:
       para_feat = tfidf.fit_transform(text)
       q_feat = tfidf.transform([" ".join(ques)])
    except ValueError:
       return []

    dist = pairwise_distances(q_feat, para_feat, "cosine").ravel()
    paragraphs = [(i, p) for i, p in enumerate(paragraphs)]
    sorted_ix = np.lexsort(([x[0] for x in paragraphs], dist))
    pars = [paragraphs[i] for i in sorted_ix] 
    sorted_supports = [paragraphs[i][1] for i in sorted_ix[:n_to_select]] 

    if tfidf_layer == 1:
        return sorted_supports
    elif tfidf_layer == 2:
        para_features = tfidf.fit_transform(sorted_supports[1:])
        q_features = tfidf.transform([" ".join(sorted_supports[0])])
        dists = pairwise_distances(q_features, para_features, "cosine").ravel()
        sorted_ix = np.lexsort((sorted_supports[1:], dists))
        supports = [sorted_supports[idx + 1] for idx in sorted_ix] # backward-shift by one space
        
        assert len(sorted_supports) == len(supports) + 1
        supports.insert(0, sorted_supports[0])

        return supports
    else:
        raise NotImplementedError

def filter_tfidf(ques, paragraphs, stop_words):
    tfidf = TfidfVectorizer(strip_accents="unicode", stop_words=stop_words)    
    n_to_select = 1 
    text = []
    for para in paragraphs:
       text.append(" ".join(para))
    try:
       para_feat = tfidf.fit_transform(text)
       q_feat = tfidf.transform([" ".join(ques)])
    except ValueError:
       return []
    dist = pairwise_distances(q_feat, para_feat, "cosine").ravel()
    paragraphs = [(i, p) for i, p in enumerate(paragraphs)]
    sorted_ix = np.lexsort(([x[0] for x in paragraphs], dist))
    return [paragraphs[i] for i in sorted_ix[:n_to_select]] 

def create_processed_wikihop_dataset_cs(config):
    train_path = os.path.join(config.data_dir, 'wikihop',
            'train.json')
    dev_path = os.path.join(config.data_dir, 'wikihop',
            'dev.json')

    print("train file: " + train_path)
    print("dev file: " + dev_path)

    stop_words = ['the', ',', '.', 'of', 'and', 'in', 'is', 'a', 'to', ')', '(', "''", '``', 'as', 'by', 'it', 'was', 'or', 'with', 'on', 'an', 'for', "'s", 'from', 'are', 'its', 'city', 'that', 'which', 'also']

    train_data = read_wikihop_data_wcs(train_path,
            config.commonsense_file,
            config.max_context_iterations,
            stop_words,
            config.tfidf_layer)
    dev_data = read_wikihop_data_wcs(dev_path,
            config.commonsense_file,
            config.max_context_iterations,
            stop_words,
            config.tfidf_layer)
    return train_data, dev_data

def get_total_words(summaries, qaps_file):
    total_words = []
    with open(qaps_file, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        next(spamreader)
        for row in spamreader:
            ques = nltk.word_tokenize(row[2].decode('ascii', 'ignore').encode('utf-8'))
            answ = nltk.word_tokenize(row[3].decode('ascii', 'ignore').encode('utf-8'))
            answ_2 = nltk.word_tokenize(row[4].decode('ascii', 'ignore').encode('utf-8'))
            qap = ques + answ + answ_2
            total_words.extend([str(w).lower() for w in qap])
    for k,v in summaries.items():
        total_words.extend(str(w).lower() for w in nltk.word_tokenize(" ".join(v)))     
    total_words = list(total_words)
    return total_words

def create_processed_dataset(config, data_type):
    debug = config.debug
    data_summary_path = os.path.join(
            config.data_dir, 'third_party', 'wikipedia', 'summaries.csv')
    data_input_path = os.path.join(config.data_dir, 'qaps.csv')
    relations = CG.get_relations(config.commonsense_file)

    summaries = {}

    with open(data_summary_path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        next(spamreader)

        print("Loading summary data!")
        for row in tqdm(spamreader):
            set_type = row[1].encode('utf-8')

            if set_type != data_type:
                continue

            doc_num = row[0]
            tokenized_summary = nltk.word_tokenize(row[3].decode('ascii', 'ignore').encode('utf-8').lower())
            summaries[doc_num] = tokenized_summary

    total_words = get_total_words(summaries, data_input_path)
    stop_words = get_stop_words(total_words, config.num_stop_words)
        
    data_list = []
    data_idx = 0

    with open(data_input_path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        next(spamreader)
        print("Loading QA data!")
        for row in tqdm(spamreader):
            set_type = row[1].encode('utf-8')
            if set_type != data_type:
                continue


            doc_num = row[0]
            summary = summaries[doc_num]
            ques = nltk.word_tokenize(row[2].decode('ascii',
                'ignore').encode('utf-8').lower())
            answ = nltk.word_tokenize(row[3].decode('ascii',
                'ignore').encode('utf-8').lower())
            answ_2 = nltk.word_tokenize(row[4].decode('ascii',
                'ignore').encode('utf-8').lower())

            data_pt = {}

            data_pt['doc_num'] = doc_num
            data_pt['ques'] = ques
            data_pt['answer1'] = answ
            data_pt['answer2'] = answ_2
            data_pt['summary'] = summary

            data_list.append(data_pt)

            data_idx += 1

    if config.load_commonsense:
        print("Building commonsense...")
        for i in tqdm(range(len(data_list))):
            data_pt = data_list[i]
            ques = data_pt['ques']
            summary = data_pt['summary']
            subgraph = CG.build_trees(relations, ques, stop_words, summary)
            selected_relations = sample_relations(subgraph, summary)
            data_pt['commonsense'] = selected_relations
        print("Done!")

    return data_list

def save_wikihop_processed_dataset(config, data, data_type):
    if data_type == 'train':
        path = config.processed_dataset_train
    elif data_type == 'valid':
        path = config.processed_dataset_valid
    elif data_type == 'test':
        path = config.processed_dataset_test


    with open(path, 'wb') as f:
        for data_pt in data:
            try:
                f.write(json.dumps(data_pt) + '\n')
            except UnicodeDecodeError:
                continue

def save_processed_dataset(config, data, data_type):
    if data_type == 'train':
        path = config.processed_dataset_train
    elif data_type == 'valid':
        path = config.processed_dataset_valid
    elif data_type == 'test':
        path = config.processed_dataset_test


    with open(path, 'wb') as f:
        for data_pt in data:
            f.write(json.dumps(data_pt) + '\n')

def load_processed_dataset(config, data_type):
    if data_type == 'train':
        path = config.processed_dataset_train
    elif data_type == 'valid':
        path = config.processed_dataset_valid
    elif data_type == 'test':
        path = config.processed_dataset_test

    data_list = []

    data_pts = 0
    answer_in_context = 0
    total = 0
    candidate_in_context = 0

    with open(path, 'rb') as f:
        for idx, line in enumerate(f):
            obj = json.loads(line)
            
            total_len = 0

            summ = []

            if isinstance(obj['summary'][0], list):
                for doc in obj['summary']:
                    summ.extend(doc)
                    total_len += len(doc)
                    if total_len > config.max_context_iterations:
                        break
            else:
                summ = obj['summary']

            obj['summary'] = lower_all(summ)

            if config.multiple_choice:
                data_pts += len(obj['candidates'])
            else:
                data_pts += 1

            if 'answer2' not in obj:
                obj['answer2'] = obj['answer1']

            if len(obj['ques']) == 1:
                continue

            if 'orig_answer' not in obj:
                obj['orig_answer'] = " ".join(obj['answer1'])

            if config.multiple_choice:
                if 'orig_cand' not in obj and 'candidates' in obj:
                    obj['orig_cand'] = []
                    for cand in obj['candidates']:
                        obj['orig_cand'].append(" ".join(cand))

                assert len(obj['orig_cand']) == len(obj['candidates'])

            if 'data_id' not in obj:
                obj['data_id'] = idx

            data_list.append(obj)

            if config.eval_num > 0:
                if data_type == 'valid' and data_pts > config.eval_num:
                    print("Evaluating on %d examples with %d candidates" % 
                            (len(data_list), data_pts))
                    break

    return DataSet(data_list, data_type)

def shuffle_cs(data):
    cs_list = []
    cs_lengths = []
    for d in data.data:
      cs_list.extend(d["commonsense"]) 
      cs_lengths.append(len(d['commonsense']))

    out_data = []

    for i, d in enumerate(data.data):
        out_obj = copy.deepcopy(data.data[i])
        out_obj['commonsense'] = random.sample(cs_list,
                random.choice(cs_lengths))

        out_data.append(out_obj)

    return DataSet(out_data, data.data_type)
