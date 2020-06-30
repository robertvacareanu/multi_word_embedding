import torch
import torch.nn as nn
import numpy as np
import gensim
from mwe_function_model import LSTMMultiply, FullAdd, AddMultiply, LSTMMweModel
from vocabulary import AbstractVocabulary
from baseline import RandomLSTM, Average, Max
import pandas as pd
from typing import List
import argparse


def cosine_similarity(a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def get_word_e(vocabulary, embeddings, word, device):
    return embeddings.center_embeddings(vocabulary.to_input_tensor([[word]], device)).squeeze(dim=0).squeeze(dim=0).cpu().numpy()


def similar_words_by_vector(vocabulary, embeddings, vector, device, k=5):
    values = list(map(lambda word: (word, cosine_similarity(get_word_e(vocabulary, embeddings, word, device), vector)), list(vocabulary.element2id.keys())[1:]))
    cos_vals = np.array([x[1] for x in values])
    words = np.array([x[0] for x in values])
    sorted_vals = np.argpartition(cos_vals, -k)[-k:]
    return list(zip(list(words[sorted_vals]), list(cos_vals[sorted_vals])))


def get_mwe_e(mwe, vocabulary, embeddings, words, device):
    words_vectorized = vocabulary.to_input_tensor([words], device)
    words_emb = embeddings.center_embeddings(words_vectorized)
    words_mwe = mwe(words_emb, torch.tensor([len(words)]))
    return words_mwe.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entry point of the application.")

    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved model to test")
    # Currently is used as path to the pre-computed vocabulary embeddings
    parser.add_argument("--embeddings-path", type=str, default='sg_embeddings_250', help="Path to the pre-trained embeddings to use")
    parser.add_argument("--vocabulary-path", type=str, default='vocab/vocab_250.json', help="Path to the stored vocabulary")
    parser.add_argument("--test-path", type=str, default='test_file', help="Path to the file containing the words to test")
    result = parser.parse_args()

    device = torch.device('cpu')

    mwe_function = LSTMMultiply(300, 500)
    mwe_function.load_state_dict(torch.load(result.model_path, map_location=device))
    sg_embeddings = torch.load(result.embeddings_path, map_location=device)
    vocabulary = AbstractVocabulary.load(result.vocabulary_path)

    # result = []
    for line in open(result.test_path):
        query = line if '\n' not in line else line[:-1]
        result = similar_words_by_vector(vocabulary, sg_embeddings, get_mwe_e(mwe_function, vocabulary, sg_embeddings, query.split(' '), device), device)
        print(f"For query={query}, we got the following results:")
        for r in result:
            print(f"\t{r[0]}\t{r[1]}")




