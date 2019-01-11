import os
import tensorflow as tf
import numpy as np
from math import exp
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.bleu.bleu import Bleu
from summary_handler import SummaryHandler
from data import GenModelVocab, translate, translate_spans
import time
import copy
import json

import sys

import random

import json

meteor_obj = Meteor()
rouge_obj = Rouge()
cider_obj = Cider()
bleu_obj = Bleu(4)

def gen_mc_preds(config, valid_data, vocab, model, sess):
    i = 0

    to_eval = []

    rankings = {}

    print("making predictions...!")
    for i in tqdm(range(len(valid_data.data))):

        datapt = valid_data.data[i]

        for j, candidate in enumerate(datapt['candidates']):
            if len(to_eval) == config.batch_size:
                fd = model.encode(to_eval, False)
                _, preds = model.eval(sess, fd)

                for k, eval_pt in enumerate(to_eval):
                    if eval_pt['datapt'] in rankings:
                        rankings[eval_pt['datapt']].append(preds[k])

                    else:
                        rankings[eval_pt['datapt']] = [preds[k]]

                    to_eval = []

            tokens = candidate

            copy_datapt = copy.deepcopy(datapt)
            
            copy_datapt['answer1'] = copy_datapt['answer2'] = tokens

            copy_datapt['datapt'] = i

            to_eval.append(copy_datapt)

    real_length = len(to_eval)
    assert real_length < config.batch_size
    if real_length != 0:
        while len(to_eval) < config.batch_size:
            to_eval.append(to_eval[0])

        fd = model.encode(to_eval, False)
        _, preds = model.eval(sess, fd)

        for k, eval_pt in enumerate(to_eval):
            if k >= real_length:
                break

            if eval_pt['datapt'] in rankings:
                rankings[eval_pt['datapt']].append(preds[k])

            else:
                rankings[eval_pt['datapt']] = [preds[k]]

            to_eval = []

    print('done!')

    predictions = {}
    ranking_keys = list(rankings.keys())
    for data_pt in tqdm(ranking_keys):
        pred_idx = np.argmax(rankings[data_pt])
        try:
            pred = valid_data.data[data_pt]['orig_cand'][pred_idx]
        except:
            print("ERROR!!!")
            print("len of valid data: %d" % len(valid_data.data))
            print("trying to access datapt: %d" % data_pt)
            print("len of cand list: %d" %
                    len(valid_data.data[data_pt]['orig_cand']))
            print("trying to index: %d" % pred_idx)
            pred = valid_data.data[data_pt]['orig_cand'][0]

        predictions[valid_data.data[data_pt]['data_id']] = pred
    return predictions

def eval_multiple_choice_dataset(config, valid_data, vocab, model, sess):
    predictions = gen_mc_preds(config, valid_data, vocab, model, sess)

    total = 0
    correct = 0

    correctness_dict = {}

    for datapt in valid_data.data:
        data_id = datapt['data_id']
        pred = predictions[data_id]
        ans = datapt['orig_answer']

        total += 1
        if pred == ans:
            correct += 1
            correctness_dict[data_id] = 1
        else:
            correctness_dict[data_id] = 0

    with open("%s-wikihop-preds.json" % config.model_name, 'w') as f:
        json.dump(predictions, f)

    return float(correct) / total

def eval_dataset(config, valid_data, vocab, model, sess):
    model_preds = []
    ground_truths = []
    memories = []
    questions = []
    commonsense = []

    total_eval_loss = []

    batch_obj = enumerate(valid_data.get_batches(config.batch_size,
        shuffle=False, pad_to_full_batch=True))
    
    if config.show_eval_progress:
        batch_obj = tqdm(batch_obj)

    for i,eval_batches in batch_obj:
        is_training = False

        if config.sample != -1 and i > config.sample:
            break

        ground_truths.extend(map(
            lambda item: [item['answer1'], item['answer2']],
            eval_batches))
        memories.extend(map(lambda item: item['summary'],
            eval_batches))
        questions.extend(map(lambda item: item['ques'],
            eval_batches))
        if config.load_commonsense:
            commonsense.extend(map(lambda item: item['commonsense'],
                eval_batches))


        fd = model.encode(eval_batches, is_training)
        oovs = model.get_batch_oov()

        eval_loss, preds = model.eval(sess, fd)

        model_preds.extend(map(lambda p: translate(p, vocab, oovs),
            preds))

        total_eval_loss.append(eval_loss)

    model_preds = model_preds[:valid_data.num_examples]
    ground_truths = ground_truths[:valid_data.num_examples]

    eval_loss = avg(total_eval_loss)

    bleu1, bleu4, meteor, rouge, cider,\
        bleu1_scores, bleu4_scores, meteor_scores, rouge_scores, cider_scores =\
            eval_set(model_preds, ground_truths)

    if config.to_print_nums == -1:
        to_print_indices = range(len(model_preds))

    else:
        to_print_indices = random.sample(range(len(model_preds)),
            config.to_print_nums)

    for idx in to_print_indices:
        print("Data %d" % idx)
        print("Summary: ", " ".join(memories[idx]))
        if config.load_commonsense:
            print("Commonsense: ")
            for path in commonsense[idx]:
                for concept in path:
                    print concept, "->", 
                print ""
        print("Question: ", " ".join(questions[idx]))
        print("Answer1: ", " ".join(ground_truths[idx][0]))
        print("Answer2: ", " ".join(ground_truths[idx][1]))
        print("Predicted: ", " ".join(model_preds[idx]))
        print("Bleu1: %.3f, Bleu4: %.3f, Rouge-L: %.3f, Meteor: %.3f, CIDEr: %.3f" %
                (bleu1_scores[idx], bleu4_scores[idx], rouge_scores[idx],
                    meteor_scores[idx], cider_scores[idx]))

        print ""
        print("=" * 80)
        print ""

    return bleu1, bleu4, meteor, rouge, cider, eval_loss, model_preds

def eval_set(predictions, ground_truth):
    word_target = ground_truth # nested list of answers, each expressed as list of tokens
    word_response = predictions # nested list of preds, each expressed as list of word

    assert len(word_target) == len(word_response)

    word_target_dict = dict(enumerate(
        map(lambda item: map(lambda s: " ".join(s), item),
            word_target)))
    word_response_dict = dict(enumerate(map(lambda item: [" ".join(item)],
        word_response)))

    bleu_score, bleu_scores = bleu_obj.compute_score(
            word_target_dict, word_response_dict)
    bleu1_score, _, _, bleu4_score = bleu_score
    bleu1_scores, _, _, bleu4_scores = bleu_scores
    meteor_score, meteor_scores = meteor_obj.compute_score(
            word_target_dict, word_response_dict) 
    rouge_score, rouge_scores = rouge_obj.compute_score(
            word_target_dict, word_response_dict) 
    cider_score, cider_scores = cider_obj.compute_score(
            word_target_dict, word_response_dict) 

    return bleu1_score, bleu4_score, meteor_score, rouge_score, cider_score,\
        bleu1_scores, bleu4_scores, meteor_scores, rouge_scores, cider_scores

def write_summaries(sess, summary_handler, loss, eval_loss, bleu_1, bleu_4,
        meteor, rouge, cider, iteration):
    scores = {}
    scores['ITERATION'] = iteration
    scores['LOSS'] = loss
    scores['PERPLEXITY'] = exp(loss) 
    scores['EVAL_PERPLEXITY'] = exp(eval_loss) 
    scores['BLEU_1'] = bleu_1
    scores['BLEU_4'] = bleu_4
    scores['METEOR'] = meteor
    scores['ROUGE'] = rouge
    scores['CIDER'] = cider

    summary_handler.write_summaries(sess, scores) 

def avg(lst):
    avg = sum(lst) / len(lst)
    return avg
 
def gpu_config():
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'
    return config

def restore_from_last_ckpt(config, model, sess):
    out_dir = os.path.join(config.out_root, config.model_name)
    ckpt_dir = os.path.join(out_dir, 'ckpts')

    ckpts = []
    for file in os.listdir(ckpt_dir):
        if file.endswith("index"):
            ckpts.append(file[:file.rfind(".")])

    ckpts.sort()

    last_ckpt = os.path.join(ckpt_dir, ckpts[-1])

    steps = last_ckpt[last_ckpt.find("step") + 4:]
    steps = int(steps[:steps.find("_")])

    epochs = last_ckpt[last_ckpt.find("epoch") + 6:]
    epochs = int(epochs[:epochs.find("_")])

    print("Restoring from %s" % last_ckpt)
    print("At epoch %d, step %d" % (epochs, steps))

    model.restore_from(sess, last_ckpt)

    print("Done")

    return epochs, steps, out_dir, ckpt_dir

def restore_from_step(config, model, sess, at_step):
    out_dir = os.path.join(config.out_root, config.model_name)
    ckpt_dir = os.path.join(out_dir, 'ckpts')

    ckpts = []
    for file in os.listdir(ckpt_dir):
        if file.endswith("index"):
            ckpts.append(file[:file.rfind(".")])

    for ckpt in ckpts:
        steps = ckpt[ckpt.find("step") + 4:]
        steps = int(steps[:steps.find("_")])

        if steps == at_step:
            print("Restoring from %s" % ckpt)
            ckpt_path = os.path.join(ckpt_dir, ckpt)
            model.restore_from(sess, ckpt_path)
            print("Done")

            return

    raise ValueError("No step found!")

def debug(config, *msg):
    if config.debug:
        print(msg)
