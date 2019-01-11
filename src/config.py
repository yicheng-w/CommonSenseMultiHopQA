import os
import tensorflow as tf
from main import main as m

from sys import argv

flags = tf.app.flags


flags.DEFINE_string("data_dir", "raw_data/narrativeqa/", "Data dir")
flags.DEFINE_string("commonsense_file", "data/cn_relations_orig.txt", "ConceptNet file")
flags.DEFINE_string("summary_save_path", "logs/", "Log files")
flags.DEFINE_string("model_name", "default", "Log files")

flags.DEFINE_string("version", '', 'model version to use')
flags.DEFINE_boolean("shuffle_commonsense", False, 'Loads pickle file w commonsense')
flags.DEFINE_boolean("load_commonsense", False, 'Loads pickle file w commonsense')

flags.DEFINE_integer("max_num_paths", 40, 'Commonsense addition')

flags.DEFINE_string("mode", "train", "Mode")
flags.DEFINE_integer("num_epochs", 20, "Epochs")
flags.DEFINE_integer("batch_size", 36, "Batch size")
flags.DEFINE_integer("checkpoint_size", 500, "Checkpoint size")

flags.DEFINE_integer("max_iterations", 37, "Max question iterations")
flags.DEFINE_integer("max_target_iterations", 20, "max answer iterations")
flags.DEFINE_integer("max_context_iterations", 1000, "Max summary size")
flags.DEFINE_integer("num_stop_words", 45, "Num stop words for tfidf")
flags.DEFINE_integer("max_oovs", 1000, "max number of oov words")

flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
flags.DEFINE_float("clipping_threshold", 5.0, "Clipping threshold")
flags.DEFINE_integer("embedding_size", 256, "Embedding size")
flags.DEFINE_integer("hidden_size_encoder", 128, "Encoder hidden size")
flags.DEFINE_integer("decoder_attn_size", 256, "Decoder attention size")
flags.DEFINE_integer("num_layers_encoder", 1, "Encoder num layers")
flags.DEFINE_integer("num_layers_decoder", 1, "Decoder num layers")

flags.DEFINE_float("dropout_rate", 0.4, "Dropout")
flags.DEFINE_boolean("debug", False, "debug mode")
flags.DEFINE_integer("min_occurence", 20,
    "minimum occurence to be counted in the vocab")
flags.DEFINE_boolean("start_eval", False, 'start w/ an eval')
flags.DEFINE_integer("sample", -1, '# of batches to use in eval, -1 means all')
flags.DEFINE_integer("to_print_nums", 0,
    'how many examples to display, -1 means all of them')
flags.DEFINE_boolean("use_dev", True, "evaluate on dev set")
flags.DEFINE_integer("re_eval_step", 1,
    "steps in re-evaluation, default is 1, i.e., reval all")
flags.DEFINE_boolean("show_eval_progress", True, "Show progress of eval")

flags.DEFINE_string("use_ckpt", None, "ckpt to use")
flags.DEFINE_integer("at_step", None, 'eval at step')
flags.DEFINE_integer("num_attn_hops", 3, "numbers of hops for attention")

flags.DEFINE_boolean("continue_training", False,
    "Continue training a trained model?")

flags.DEFINE_string("elmo_options_file",
    "lm_data/nqa/elmo_2x4096_512_2048cnn_2xhighway_options.json",
    "ELMo options file")
flags.DEFINE_string("elmo_weight_file",
    "lm_data/nqa/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
    "ELMo weights file")
flags.DEFINE_string("elmo_token_embedding_file",
    "lm_data/nqa/elmo_token_embeddings-nqa.hdf5", 
    "Pre-computed ELMo token weights")
flags.DEFINE_string("elmo_vocab_file",
    "lm_data/nqa/narrative_qa_vocab.txt",
    "Vocab file for ELMo")

flags.DEFINE_string("out_root", "out", 'output directory')

flags.DEFINE_string("processed_dataset_train", None,
    'process dataset train')
flags.DEFINE_string("processed_dataset_valid", None,
    'process dataset valid')
flags.DEFINE_string("processed_dataset_test", None,
    'process dataset test')

flags.DEFINE_boolean("multiple_choice", False, "is this dataset mc?")

flags.DEFINE_boolean("shuffle_cs", False, "shuffle cs?")
flags.DEFINE_integer('eval_num', -1, 'evaluate on subset of dev')

flags.DEFINE_integer("tfidf_layer", 1, "layers of tfidf selection")


def main(_):
    print(argv)
    config = flags.FLAGS
    m(config)

if __name__ == '__main__':
    tf.app.run()
