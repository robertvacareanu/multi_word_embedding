import torch
import pandas as pd
import numpy as np
import re
import random
from nltk.tokenize import word_tokenize
from nltk import ngrams
import string
import copy
from multiprocessing import Pool
import json

def pad_sentence(sentences, pad_token):
    """
    Pad list of sentences to the longest sentence in the batch
    :param sentences: list of sentences, where each sentence is represented by a list of words (str)
    :param pad_token: padding token
    :return: list of sentences where sentences shorter than the max lengths are padded out with the pad_token
    """
    max_sentence = max([len(s) for s in sentences])
    sentences_padded = [ x +[pad_token] *(max_sentence -len(x)) for x in sentences]
    return sentences_padded


def read_corpus(file_path):
    """
    Read the corpus
    :param file_path: path to the corpus file. It is assumed that the file contains lines of text separated by
    newline (\n)
    :return: List[List[str]]: list of sentences (words)
    """
    data = []
    for line in open(file_path):
        sentence = filter(None, line.strip().split(' '))
        data.append(sentence)
    return data


"""
def read_wikipedia_corpus(path):
    for line in open(path):
        words = line.lower()
        words = word_tokenize(words)
        words = list(words)
        yield words
"""
def read_wikipedia_corpus(path):
    for line in open(path):
        yield line.replace('\n', '').split(' ') # was already tokenized
        # yield word_tokenize(line.lower())


def read_tratz(path):
    dataset = pd.read_csv(path, header=None, quoting=3, sep='\t', names=[
                          'constituent1', 'constituent2', 'label'])
    dataset['mwe'] = dataset['constituent1'] + ' ' + dataset['constituent2']
    dataset = dataset.drop(columns=['constituent1', 'constituent2'])[
        ['mwe', 'label']].values
    for label in dataset[:, 1]:
        yield label



def make_corpus(file_path, save_path, mwes_path, number_of_lines=10000):
    """
    Make a corpus file (from a tokenized corpus), containing only sentences with mwe (therefore, with _). Needed to work with the dataset (EntitySentenceDataset), which expects each sentence
    to contain a mwe (merged using '_'). This design choice was made due to how Shwartz is constructing her training data. Her script for constructing the data can be found at
    (https://github.com/vered1986/NC_embeddings/blob/8dec4e2f7918ab7606abf61b9d90e4f2786a9652/source/training/distributional/preprocessing/extract_ngrams_and_windows.py)
    :param file_path: path to the corpus file
    :param save_path: path to where to save the generated corpus file
    :param mwes_path: path to the file containing the mwes. We keep only the sentences with mwes that are in the file at mwes_path
    """
    added_lines = 0
    mwes = [x.replace('\t', '_').strip() for x in open(mwes_path, 'r')]
    with open(save_path, 'w+') as f_out:
        with open(file_path, 'r') as f_in:
            for line in f_in:
                if added_lines < number_of_lines:
                    if '_' in line: # add only the lines with a mwe in it
                        words = line.split(' ')
                        mwe_word = list(filter(lambda x: '_' in x, words))[0]
                        if mwe_word.strip() in mwes and len(words) > 1: # add only if it is in mwes list and only if the lines that also have some context in it.
                            f_out.write(line)
                            added_lines+=1
                else:
                    return


def init_random(seed):
    """
    Init torch, torch.cuda and numpy with same seed. For reproducibility.
    :param seed: Random number generator seed
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Seeds used by Shwartz
    # torch.manual_seed(133)
    # torch.cuda.manual_seed(133)
    # np.random.seed(1337)
    # random.seed(13370)


def cosine_similarity(a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def similar_words_by_vector(vocabulary, embeddings, vector, device, k=5):
    values = list(map(lambda word: (word, cosine_similarity(embeddings.center_embeddings(vocabulary.to_input_tensor([[word]], device)).squeeze(dim=0).squeeze(dim=0).cpu().numpy(), vector)), list(vocabulary.word2id.keys())[1:]))
    cos_vals = np.array([x[1] for x in values])
    words = np.array([x[0] for x in values])
    sorted_vals = np.argpartition(cos_vals, -k)[-k:]
    return list(zip(list(words[sorted_vals]), list(cos_vals[sorted_vals])))


def get_mwe_e(mwe, vocabulary, embeddings, words):
    words_vectorized = vocabulary.to_input_tensor([words], torch.device('cpu'))
    words_emb = embeddings.center_embeddings(words_vectorized)
    words_mwe = mwe(words_emb, torch.tensor([len(words)]))
    return words_mwe.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()


def format_number(number, decimals=5):
    return (f'%.{decimals}f' % number)


def flatten(x):
    return [z for y in x for z in y]
