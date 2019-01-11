# Commonsense for Generative Multi-Hop Question Answering Tasks (EMNLP 2018)

This repository contains the code and setup instructions for our EMNLP 2018 paper
"Commonsense for Generative Multi-Hop Question Answering Tasks". See full paper
[here](https://arxiv.org/abs/1809.06309).

## Environment Setup

We trained our models with python 2 and TensorFlow 1.3, a full list of python
packages is listed in `requirements.txt`

## Downloading Data

First, to setup the directory structure, please run `setup.sh` to create the
appropriate directories.

We download the raw data for NarrativeQA and WikiHop. For NarrativeQA, we
download from github, starting at the root of the directory, run
```
cd raw_data
git clone https://github.com/deepmind/narrativeqa.git
```

For WikiHop, we download the QAngaroo dataset
[here](https://drive.google.com/file/d/1ytVZ4AhubFDOEL7o7XrIRIyhU8g9wvKA/view),
and extract the zip file into the `raw_data` directory.

We use pre-computed ELMo representations. Download our pre-computed ELMo
representation
[here](https://drive.google.com/file/d/1pwzyEa0ogrXAMDmkFWOwH_eCSk8bP7ud/view),
and extract into the folder `lm_data`.

We also use a local version of ConceptNet's relations. Download the relations
file from
[here](https://drive.google.com/file/d/14nb2lM_KrWReSHlEaXVg9KE1WrcAV2Lj/view)
and put it in the folder `data`.

## Build Processed Datasets

We need to build processed datasets with extracted commonsense information. For
NarrativeQA, we run:
```
python src/config.py \
    --mode build_dataset \
    --data_dir raw_data/narrativeqa \
    --load_commonsense \
    --commonsense_file data/cn_relations_orig.txt \
    --processed_dataset_train data/narrative_qa_train.jsonl \
    --processed_dataset_valid data/narrative_qa_valid.jsonl \
    --processed_dataset_test data/narrative_qa_test.jsonl
```

To build processed datasets with extracted commonsense for WikiHop, we run:
```
python src/config.py \
    --mode build_wikihop_dataset \
    --data_dir raw_data/qangaroo_v1.1 \
    --load_commonsense \
    --commonsense_file data/cn_relations_orig.txt \
    --processed_dataset_train data/wikihop_train.jsonl \
    --processed_dataset_valid data/wikihop_valid.jsonl 
```

## Training & Evaluation

### Training

To train models for NarrativeQA, run:
```
python src/config.py \
    --version {commonsense_nqa, baseline_nqa} \
    --model_name <model_name> \
    --processed_dataset_train data/narrative_qa_train.jsonl \
    --processed_dataset_valid data/narrative_qa_valid.jsonl \
    --batch_size 24 \
    --max_target_iterations 15 \
    --dropout_rate 0.2 
```

To train models for WikiHop, run:
```
python src/config.py \
    --version {commonsense_wh, baseline_wh} \
    --model_name <model_name> \
    --elmo_options_file lm_data/wh/elmo_2x4096_512_2048cnn_2xhighway_options.json \
    --elmo_weight_file lm_data/wh/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 \
    --elmo_token_embedding_file lm_data/wh/elmo_token_embeddings.hdf5 \
    --elmo_vocab_file lm_data/wh/wikihop_vocab.txt \
    --processed_dataset_train data/wikihop_train.jsonl \
    --processed_dataset_valid data/wikihop_valid.jsonl \
    --multiple_choice \
    --max_target_iterations 4 \
    --max_iterations 8 \
    --batch_size 16 \
    --max_target_iterations 4 \
    --max_iterations 8 \
    --max_context_iterations 1300 \
    --dropout_rate 0.2
```

### Evaluation

To evaluate NarrativeQA, we need to first generate official answers on the test
set. To do so, run:
```
python src/config.py \
    --mode generate_answers \
    --processed_dataset_valid data/narrative_qa_valid.jsonl \
    --processed_dataset_test data/narrative_qa_test.jsonl 
```

This will create the reference files `val_ref0.txt`, `val_ref1.txt`,
`test_ref0.txt` and `test_ref1.txt`.

To evaluate a model on NarrativeQA, run:
```
python src/config.py \
    --mode test \
    --version {commonsense_nqa, baseline_nqa} \
    --model_name <model_name> \
    --use_ckpt <ckpt_name> \
    --use_test \ # only use this flag if you want to evaluate on test set
    --processed_dataset_train data/narrative_qa_train.jsonl \
    --processed_dataset_valid data/narrative_qa_valid.jsonl \
    --processed_dataset_test data/narrative_qa_test.jsonl \
    --batch_size 24 \
    --max_target_iterations 15 \
    --dropout_rate 0.2 
```
which generates the output (a new file named <model_name>\_preds.txt). Then run
```
python src/eval_generation.py <ref0> <ref1> <output>
```
where `ref0` and `ref1` are the generated reference files for the automatic
metrics.

To evaluate a model on WikiHop, run:
```
python src/config.py \
    --mode test \
    --version {commonsense_wh, baseline_wh} \
    --model_name <model_name> \
    --use_ckpt <ckpt_name> \
    --elmo_options_file lm_data/wh/elmo_2x4096_512_2048cnn_2xhighway_options.json \
    --elmo_weight_file lm_data/wh/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 \
    --elmo_token_embedding_file lm_data/wh/elmo_token_embeddings.hdf5 \
    --elmo_vocab_file lm_data/wh/wikihop_vocab.txt \
    --processed_dataset_train data/wikihop_train.jsonl \
    --processed_dataset_valid data/wikihop_valid.jsonl \
    --multiple_choice \
    --max_target_iterations 4 \
    --max_iterations 8 \
    --batch_size 16 \
    --max_target_iterations 4 \
    --max_iterations 8 \
    --max_context_iterations 1300 \
    --dropout_rate 0.2 
```
This outputs the test accuracy and generates an output file containing the
model's predictions.

## Download and Run Pre-Trained Models

We release some pretrained models for both the NarrativeQA and WikiHop datasets.
The results are listed below:

**NarrativeQA**

Model | Dev (R-L/B-1/B-4/M/C) | Test (R-L/B-1/B-4/M/C)
------|------------------------------------------|--------------------
Baseline|48.10/45.83/20.62/20.28/163.87|46.15/44.55/21.16/19.60/159.51
Commonsense|51.70/49.28/23.18/22.17/179.13|50.15/48.44/24.01/21.76/178.95

These NarrativeQA models resulted from further tuning after the paper's
publication and have better performance than those presented in the paper.

**WikiHop**

Model | Dev Acc (%) | Test Acc (%)
------|-------------|--------------
Baseline|56.2%|57.5%
Commonsense|58.5%|57.9%

These WikiHop results are after tuning on the official/full WikiHop validation
set, these numbers will appear in an upcoming arxiv update available
[here](https://arxiv.org/abs/1809.06309).

Download our pretrained models here:
- [NarrativeQA Commonsense Model](https://drive.google.com/file/d/1V6G2sTvOiyEtsnVCBV34DzyhTCR58TXM/view)
- [NarrativeQA Baseline Model](https://drive.google.com/file/d/1DsjrNB9z8J2n7oLecRTx3jjyMvHkCnoj/view)
- [WikiHop Commonsense Model](https://drive.google.com/file/d/1ldJJ5cA0hthreC3v3Ux6l0cOFTMDvNHM/view)
- [WikiHop Baseline Model](https://drive.google.com/file/d/1LlgH1gaK96MApg5wsfVCg3LTq_N0_3C8/view)

Download and extract them to the `out` repo, and see above for how to evaluate
these models.

## Bibtex

```
@inproceedings{bauerwang2019commonsense,
  title={Commonsense for Generative Multi-Hop Question Answering Tasks},
  author={Lisa Bauer*, Yicheng Wang* and Mohit Bansal},
  booktitle={Proceedings of the Empirical Methods in Natural Language Processing},
  year={2018}
}
```
