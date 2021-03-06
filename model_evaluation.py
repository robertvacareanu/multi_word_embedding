from sklearn.linear_model import LogisticRegression
from mwe_function_model import LSTMMultiply, LSTMMweModel, FullAdd, MultiplyMean, MultiplyAdd, CNNMweModel, LSTMMweModelPool, GRUMweModel
from task_model import MWESkipGramTaskModel, MWEMeanSquareErrorTaskModel, MWESentenceSkipGramTaskModel
import numpy as np
import argparse
from dataset import TratzDataset
from vocabulary import make_label_vocab, AbstractVocabulary
from embeddings import SkipGramEmbeddings
from utils import read_tratz, init_random
import pandas as pd
import torch
import json
from evaluate import Evaluation2


def bootstrapping_test(evaluation1, evaluation2, total=100000):
    r1 = evaluation1.evaluate()
    r2 = evaluation2.evaluate()
    print(r1)
    print(r2)
    print(r1[-1])
    print(r2[-1])
    gold = r1[1]
    model1_score = 0
    model2_score = 0
    for _ in range(total):
        indices = np.random.choice(gold.shape[0], gold.shape[0])
        cp1 = r1[0][indices]
        cp2 = r2[0][indices]
        s1 = np.count_nonzero(cp1 == gold[indices])
        s2 = np.count_nonzero(cp2 == gold[indices])
        if s1 > s2:
            model1_score += 1
        else:
            model2_score += 1
    return (model1_score, model2_score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate the difference between two models")
    
    parser.add_argument('--model1', type=str, required=True, help="Path to the parameters of model1")
    parser.add_argument('--model1-config', type=str, required=True, help="Path to the first model's config file")
    parser.add_argument('--model2', type=str, required=True, help="Paths to the parameters of model2")
    parser.add_argument('--model2-config', type=str, required=False, help="Path to the second model's config file")
    parser.add_argument('--train-path', type=str, required=False, default='w2v', help="Path to the train file")
    parser.add_argument('--eval-path', type=str, required=False, default='w2v', help="Path to the file on which to evaluate")
    parser.add_argument('--embeddings-path1', type=str, required=False, default='w2v', help="Path to the embeddings of model1")
    parser.add_argument('--embeddings-path2', type=str, required=False, default='w2v', help="Path to the embeddings of model2")
    parser.add_argument('--vocabulary-path1', type=str, required=False, default='w2v', help="Path to the vocabulary of model1")
    parser.add_argument('--vocabulary-path2', type=str, required=False, default='w2v', help="Path to the vocabulary of model1")

    result = parser.parse_args()
    print(result)
    mwe_function_map = {'LSTMMultiply': LSTMMultiply, 'LSTMMweModel': LSTMMweModel,
                    'FullAdd': FullAdd, 'MultiplyMean': MultiplyMean, 'MultiplyAdd': MultiplyAdd,
                    'CNNMweModel': CNNMweModel, 'LSTMMweModelPool': LSTMMweModelPool, 'GRUMweModel': GRUMweModel}

    embeddings1 = SkipGramEmbeddings.from_saved_file(result.embeddings_path1)
    embeddings2 = SkipGramEmbeddings.from_saved_file(result.embeddings_path2)
    config1 = json.load(open(result.model1_config))
    config2 = json.load(open(result.model2_config))
    model1 = mwe_function_map[config1['model']['name']](config1['model']['attributes'])
    model1.load_state_dict(torch.load(result.model1))
    model2 = mwe_function_map[config2['model']['name']](config2['model']['attributes'])
    model2.load_state_dict(torch.load(result.model2))
    init_random(1)
    te = LogisticRegression(multi_class="multinomial", penalty='l2', C=0.5, solver="sag", n_jobs=20)
    evaluation1 = Evaluation2(result.train_path, result.eval_path, model1, embeddings1, result.vocabulary_path1, te=te) 
    evaluation2 = Evaluation2(result.train_path, result.eval_path, model2, embeddings2, result.vocabulary_path2, te=te) 

    eval_result = bootstrapping_test(evaluation1, evaluation2)
    print(eval_result)

    """
python -i model_evaluation.py \
--model1 /data/nlp/corpora/multi_word_embedding/data/wikipedia/tokenized_bigrams_oc_nctrain_ncvocab/models/b10/mwe_f_ft_2_200_50000_2.pt \
--model2 /data/nlp/corpora/multi_word_embedding/data/wikipedia/tokenized_bigrams_oc_random_ncvocab/models/b10/mwe_f_ft_2_200_50000_finetuned \
--model1-config /net/kate/storage/data/nlp/corpora/multi_word_embedding/data/wikipedia/configs/tokenized_bigrams_oc_nctrain_ncvocab/config_model_sgmot_oc_ft_2_200_50000.json \
--model2-config /net/kate/storage/data/nlp/corpora/multi_word_embedding/data/wikipedia/configs/tokenized_bigrams_oc_nctrain_ncvocab/config_model_sgmot_oc_ft_2_200_50000.json \
--train-path /data/nlp/corpora/multi_word_embedding/data/tratz/coarse_grained_lexical_full/train.tsv \
--eval-path /data/nlp/corpora/multi_word_embedding/data/tratz/coarse_grained_lexical_full/val.tsv \
--embeddings-path1 /data/nlp/corpora/multi_word_embedding/data/wikipedia/tokenized_bigrams_oc_nctrain_ncvocab/embeddings/embeddings_ft_w2_e200_sg_50000.pt \
--embeddings-path2 /data/nlp/corpora/multi_word_embedding/data/wikipedia/tokenized_bigrams_oc_nctrain_ncvocab/embeddings/embeddings_ft_w2_e200_sg_50000.pt \
--vocabulary-path1 /data/nlp/corpora/multi_word_embedding/data/wikipedia/tokenized_bigrams_oc_nctrain_ncvocab/vocab/vocab_nc_vocab_50000.json \
--vocabulary-path2 /data/nlp/corpora/multi_word_embedding/data/wikipedia/tokenized_bigrams_oc_nctrain_ncvocab/vocab/vocab_nc_vocab_50000.json
    """








"""
PATHS:
/data/nlp/corpora/multi_word_embedding/data/models_updated/supervised/mwe_f_normal_complete_rs1
/data/nlp/corpora/multi_word_embedding/data/models_updated/supervised/mwe_f_normal_complete_rs2
/data/nlp/corpora/multi_word_embedding/data/models_updated/supervised/mwe_f_normal_complete_rs3


/data/nlp/corpora/multi_word_embedding/data/models_updated/supervised/mwe_f_sentencewise_complete_rs1
/data/nlp/corpora/multi_word_embedding/data/models_updated/supervised/mwe_f_sentencewise_complete_rs2
/data/nlp/corpora/multi_word_embedding/data/models_updated/supervised/mwe_f_sentencewise_complete_rs3




/data/nlp/corpora/multi_word_embedding/data/models_updated/unsupervised/mwe_f_normal_complete_rs1
/data/nlp/corpora/multi_word_embedding/data/models_updated/unsupervised/mwe_f_normal_complete_rs2
/data/nlp/corpora/multi_word_embedding/data/models_updated/unsupervised/mwe_f_normal_complete_rs3


/data/nlp/corpora/multi_word_embedding/data/models_updated/unsupervised/mwe_f_sentencewise_frs_complete_rs1
/data/nlp/corpora/multi_word_embedding/data/models_updated/unsupervised/mwe_f_sentencewise_frs_complete_rs2
/data/nlp/corpora/multi_word_embedding/data/models_updated/unsupervised/mwe_f_sentencewise_frs_complete_rs3









python train.py --config-file /data/nlp/corpora/multi_word_embedding/data/models_updated/base_supervised_normal_training_config_cutoff5.json --random-seed 2 --heldout-data /data/nlp/corpora/multi_word_embedding/data/wikipedia/tokenized_bigrams_oc_nctrain_ncvocab/en_corpus_tokenized_bigrams_complete_test --which-cuda 1 --save-path /data/nlp/corpora/multi_word_embedding/data/models_updated/supervised/mwe_f_normal_complete_rs2 > supervised_normal2
python train.py --config-file /data/nlp/corpora/multi_word_embedding/data/models_updated/base_supervised_sentencewise_training_config_cutoff5.json --random-seed 3 --heldout-data /data/nlp/corpora/multi_word_embedding/data/wikipedia/tokenized_bigrams_oc_nctrain_ncvocab/en_corpus_tokenized_bigrams_complete_test --which-cuda 2 --save-path /data/nlp/corpora/multi_word_embedding/data/models_updated/supervised/mwe_f_sentencewise_complete_rs3 > supervised_sentencewise3

"""