import os
import tensorflow as tf
import numpy as np
from math import exp
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from summary_handler import SummaryHandler
from read_data import *
from data import GenModelVocab, translate, save_vocab, restore_vocab,\
    translate_spans, GloVEVocab
import time

from tensorflow.python.client import timeline

from elmo import Batcher, BidirectionalLanguageModel

from utils import *

import sys

import random

from commonsense_utils.commonsense_data import COMMONSENSE_REL_LOOKUP

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

VALID_VERSIONS = ['baseline_nqa', 'commonsense_nqa', 'baseline_wh',
    'commonsense_wh']

VALID_MODES = ['train', 'test', 'get-vocab', 'build_dataset', 'build_wikihop_dataset']

def import_model(config):
    global UsedModel
    if config.version == 'baseline_nqa':
        from model_baseline_nqa import\
            NarrativeQAGatedMultiBidafWithSelfAttnAndELMoCUDNN as\
            UsedModel
    elif config.version == 'commonsense_nqa':
        from model_commonsense_nqa import \
            GatedMultiBidaf_ResSA_ELMo_PtrGen_CS_1_fix_path_concat_project_w_rels\
            as UsedModel
    elif config.version == 'baseline_wh':
        from model_baseline_wh import \
            GatedMultiBidafWithSelfAttnAndELMoCUDNN_MC as UsedModel
    elif config.version == 'commonsense_wh':
        from model_commonsense_wh import \
            GatedMultiBidaf_ResSA_ELMo_PtrGen_CS_path_cp_rels_MC as\
            UsedModel
    else:
        raise ValueError("Model version DNE!")

def main(config):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable cpp error msgs
    if config.mode == 'train':
        _train(config)
    elif config.mode == 'test':
        _test(config)
    elif config.mode == 'get-vocab':
        _print_vocab(config)
    elif config.mode == 'build_dataset':
        _build_datasets(config)
    elif config.mode == 'build_wikihop_dataset':
        _build_wikihop_datasets(config)
    elif config.mode == 'generate_answers':
        _generate_answers(config)
    else:
        raise ValueError("Invalid Mode!")

def _print_vocab(config):
    train_data = load_processed_dataset(config, 'train')
    vocabs = train_data.get_word_lists()
    valid_data = load_processed_dataset(config, 'valid')
    vocabs.update(valid_data.get_word_lists())
    for v in vocabs:
        print(v.encode('utf-8'))

def _build_datasets(config):
    train_data_processed = create_processed_dataset(config, 'train')
    valid_data_processed = create_processed_dataset(config, 'valid')
    test_data_processed = create_processed_dataset(config, 'test')

    save_processed_dataset(config, train_data_processed, 'train')
    save_processed_dataset(config, valid_data_processed, 'valid')
    save_processed_dataset(config, test_data_processed, 'test')

def _build_wikihop_datasets(config):
    train_data_processed, dev_data_processed = create_processed_wikihop_dataset_cs(config)
    save_wikihop_processed_dataset(config, train_data_processed, 'train')
    save_wikihop_processed_dataset(config, dev_data_processed, 'valid')

def _train(config):
    import_model(config)

    train_data = load_processed_dataset(config, 'train')
    train_commonsense_list = []

    vocab_freq = train_data.get_word_lists()
    relations_vocab = COMMONSENSE_REL_LOOKUP.values()

    valid_data = load_processed_dataset(config, 'valid')

    print("Data loaded!")
    vocab = GenModelVocab(vocab_freq, config.embedding_size,
            forced_word_list=relations_vocab,
            cs=train_commonsense_list,
            threshold=config.min_occurence)

    print("Vocab built! Size (%d)" % vocab.size())

    model = UsedModel(config, vocab)

    #create session 
    gpu_configuration = gpu_config()
    sess = tf.Session(config=gpu_configuration)
    with sess.as_default():
        model.build_graph()
        print("Graph built!")
        model.add_train_op()
        print("Train op added!")
    sess.run(tf.global_variables_initializer())
    print("Variables initialized")

    if config.continue_training:
        start_e, steps, out_dir, ckpt_dir = restore_from_last_ckpt(
                config, model, sess)

        # backup new argv
        with open(os.path.join(out_dir, 'argv.txt'), 'a') as f:
            f.write("\n")
            f.write(" ".join(sys.argv))

        print("Continue training after epoch %d, step %d" % (start_e, steps))

    else:
        if config.model_name == 'default':
            c_time = time.strftime("%m_%d_%H_%M_%S", time.localtime())
            commonsense = ""
            if config.load_commonsense:
                commonsense = "_with_cs" 

            if config.shuffle_cs:
                commonsense = "_shuffle_CS"
 
            config.model_name = UsedModel.__name__ + commonsense + "_%s" % c_time

        if config.debug:
            config.checkpoint_size = 10

        if not config.debug:
            # XXX factor into class eventually
            out_dir = os.path.join(config.out_root, config.model_name)
            if os.path.exists(out_dir):
                raise ValueError("Output directory already exists!")
            else:
                os.makedirs(out_dir)

            # back up src file
            os.system("cp -r src %s" % os.path.join(out_dir, 'src'))
            # back up argv
            with open(os.path.join(out_dir, "argv.txt"), 'w') as f:
                f.write(" ".join(sys.argv))

            # back up environ
            with open(os.path.join(out_dir, 'recreate_environ.sh'), 'w') as f:
                for var, val in os.environ.items():
                    f.write("export %s=\"%s\"\n" % (var, val))

            os.system("chmod +x %s" % os.path.join(out_dir, 'recreate_environ.sh'))

            ckpt_dir = os.path.join(out_dir, "ckpts")

            vocab_loc = os.path.join(out_dir, "vocab.pkl")
            save_vocab(vocab, vocab_loc)

            print("Initialized output at %s" % out_dir)

        steps = 0
        start_e = -1

        print("Started training!")

    #construct graph handler
    if not config.multiple_choice:
        summary_handler = SummaryHandler(
                os.path.join(config.summary_save_path, config.model_name),
                ['LOSS', 'PERPLEXITY', 'EVAL_PERPLEXITY', 'BLEU_1', 'BLEU_4',
                    'METEOR', 'ROUGE', 'CIDER'])
    else:
        summary_handler = SummaryHandler(
                os.path.join(config.summary_save_path, config.model_name),
                ['LOSS', 'ACCURACY'])

    for e in range(config.num_epochs):
        total_loss = []
        for batches in tqdm(train_data.get_batches(config.batch_size,
            front_heavy = (e == 0))):

            if steps != 0 or not config.start_eval:
                steps += 1

            if steps > 10 and config.debug:
                exit(0)

            is_training = True

            fd = model.encode(batches, is_training)
            oovs = model.get_batch_oov()

            loss = model.train_step(sess, fd)
            total_loss.append(loss)

            if steps % config.checkpoint_size == 0:
                if not config.multiple_choice:
                    bleu1, bleu4, meteor, rouge, cider, eval_loss, preds = eval_dataset(
                            config, valid_data, vocab, model, sess)

                    print("Result at step %d:" % steps)
                    print("Bleu1: ", bleu1)
                    print("Bleu4: ", bleu4)
                    print("Meteor: ", meteor)
                    print("Rouge-L: ", rouge)
                    print("CIDEr: ", cider)

                    if not config.debug:
                        write_summaries(sess, summary_handler, avg(total_loss),
                                eval_loss, bleu1, bleu4, meteor, rouge, cider, steps)

                        if start_e > 0:
                            epoch = e + start_e
                        else:
                            epoch = e

                        model.save_to(sess, os.path.join(ckpt_dir,
                            "epoch_%04d_step%08d_bleu1(%f)_bleu4(%f)_meteor(%f)_rogue(%f)" %
                            (epoch, steps, bleu1, bleu4, meteor, rouge)))
                        print("Model saved!")
                else:
                    accuracy = eval_multiple_choice_dataset(
                            config, valid_data, vocab, model, sess)
                    print("Result at step %d: %f" % (steps, accuracy))

                    if not config.debug:
                        summary_handler.write_summaries(sess,
                                {
                                    'ITERATION': steps,
                                    'LOSS': avg(total_loss),
                                    'ACCURACY': accuracy
                                })

                        if start_e > 0:
                            epoch = e + start_e
                        else:
                            epoch = e

                        model.save_to(sess, os.path.join(ckpt_dir,
                            'epoch_%04d_step%08d_acc(%f)' % (epoch, steps,
                                accuracy)))

    summary_handler.close_writer()   

def _test(config):
    print("Evaluating!")
    import_model(config)

    out_dir = os.path.join(config.out_root, config.model_name)

    vocab_loc = os.path.join(out_dir, "vocab.pkl")
    if os.path.exists(vocab_loc): # vocab exists!
        vocab = restore_vocab(vocab_loc)
    else:
        raise Exception("Not valid output directory! No vocab found!")

    print("Vocab built! Size (%d)" % vocab.size())

    if config.use_dev:
        valid_data = load_processed_dataset(config, 'valid')
    else:
        print("Using test!")
        valid_data = load_processed_dataset(config, 'test')

    if config.shuffle_cs:
        valid_data = shuffle_cs(valid_data)

    print("Data loaded!")

    #construct model
    model = UsedModel(config, vocab)

    gpu_configuration = gpu_config()
    sess = tf.Session(config=gpu_configuration)
    with sess.as_default():
        model.build_graph()
        print("Graph built!")
        model.add_train_op()
        print("Train op added!")

    sess.run(tf.global_variables_initializer())

    if config.use_ckpt is not None:
        model.restore_from(sess, os.path.join(out_dir, 'ckpts', config.use_ckpt))
    elif config.at_step is not None:
        restore_from_step(config, model, sess, config.at_step)
    else:
        raise ValueError("Must specify a ckpt to restore from!")


    if not config.multiple_choice:
        bleu1, bleu4, meteor, rouge, cider, eval_loss, preds = eval_dataset(
                config, valid_data, vocab, model, sess)

        #print("Result for %s" % config.use_ckpt)
        #print("Bleu1: ", bleu1)
        #print("Bleu4: ", bleu4)
        #print("Meteor: ", meteor)
        #print("Rouge-L: ", rouge)
        #print("CIDEr: ", cider)

        filename = config.model_name + '_preds.txt'
        with open(filename, 'w') as pred_file:
            for p in preds:
                pred_file.write(" ".join(p) + "\n")
    else:
        accuracy = eval_multiple_choice_dataset( config, valid_data, vocab,
                model, sess)
        print("Result for %s: %f" % (config.use_ckpt, accuracy))

    print("Done!")

def _generate_answers(config):

    def save_answers(prefix, dset):
        ref0 = open(prefix + '_ref0.txt', 'w')
        ref1 = open(prefix + '_ref1.txt', 'w')

        punc_list = ['.', '!', '?']

        for item in dset.data:
            ref0.write(" ".join(filter(lambda w: w not in punc_list,
                item['answer1'])) + '\n')
            ref1.write(" ".join(filter(lambda w: w not in punc_list,
                item['answer2'])) + '\n')

        ref0.close()
        ref1.close()


    val_data = load_processed_dataset(config, 'valid')
    test_data = load_processed_dataset(config, 'test')

    save_answers('val', val_data)
    save_answers('test', test_data)
