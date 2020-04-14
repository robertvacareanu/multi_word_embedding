import json
import pandas as pd
import torch
import numpy as np
import gensim
import tempfile

from nltk.tokenize import word_tokenize
from typing import List
from collections import Counter

from utils import pad_sentence

def make_word_vocab(path, reader):
    vocab = AbstractVocabulary(None)
    for sentence in reader(path):
        for word in sentence:
            vocab.add(word, 1)
    return vocab


def make_label_vocab(path, reader):
    vocab = AbstractVocabulary(None)
    for label in reader(path):
        vocab.add(label, 1)
    return vocab


def make_dm_word_vocab(path):
    """
    Make vocabulary for direct minimization using the vocab built by Shwartz. It adds both the mwe and its constituents
    Path to file containing all the entities. Each constituent separated by '\t'
    """
    vocab = AbstractVocabulary(None)
    with open(path, 'r') as in_f:
        for line in in_f:
            line = line.replace('\n', '')
            mwe = line.replace('\t', '_')
            mwe_constituents = line.split('\t')
            vocab.add(mwe, 1)
            for constituent in mwe_constituents:
                vocab.add(constituent, 1)
    return vocab


def make_sgot_word_vocab(corpus, mwes, corpus_reader):
    vocab = AbstractVocabulary(None)
    for sentence in corpus_reader(corpus):
        for word in sentence:
            vocab.add(word, 1)
    with open(mwes, 'r') as in_f:
        for line in in_f:
            line = line.replace('\n', '')
            mwe = line.replace('\t', '_')
            mwe_constituents = line.split('\t')
            vocab.add(mwe, 1)
            for constituent in mwe_constituents:
                vocab.add(constituent, 1)
    return vocab


def make_word_vocab_from_gensim_file(file, count_threshold=5):
    """
    Make the vocabulary from the gensim model. Useful when already using some word embeddings with gensim -- 
    TODO needs preprocessing first. Better, give the path to the model, then write a temporary file containing all the necessary things, then create the vocabulary.
    :param file: Path to
    :param count_threshold: Cutoff value for words in the vocabulary
    :return:
    """
    element2id = {}
    element2id['<pad>'] = 0
    element2id['<unk>'] = 1
    counts = [0] * len(element2id)
    with open(file) as f:
        # skip the first element, which should be '<unk>'
        for _ in range(1):
            line = next(f)
            token, _, _, _ = line.split("\t")
            if token != '<unk>':
                raise ValueError("First should be unk")

        for line in f:
            token, count, _, _ = line.split('\t')
            if int(count) > count_threshold:
                element2id[token] = len(element2id)
                counts += [int(count)]

    AbstractVocabulary({'element2id': element2id, 'counts': counts}).save(
        f'vocab_{count_threshold}.json')


class AbstractVocabulary(object):
    """
    Class representing a vocabulary.
    Maps from element to index
    """

    def __init__(self, saved_version):
        """
        Args:
        :param saved_version (dict): dictionary containing both word2id and counts
        """
        if saved_version:
            self.element2id = saved_version['element2id']
            self.counts = saved_version['counts']
        else:
            self.element2id = dict()
            self.element2id['<pad>'] = 0
            self.element2id['<unk>'] = 1
            self.counts = [0] * len(self.element2id)
        self.unk_id = self.element2id['<unk>']
        self.pad_id = 0
        self.pad_token = '<pad>'
        self.id2element = {v: k for k, v in self.element2id.items()}

    def __getitem__(self, element):
        """
        Retrieve word's index or return unk_token index in case of out of focabulary
        :param element (usually str): element to look up
        :return: index (int): index of the element
        """
        return self.element2id.get(element, self.unk_id)

    def __setitem__(self, key, value):
        raise ValueError("Vocabulary is readonly")

    def __contains__(self, item):
        """
        Returns whether the @item is contained in element2id dictionary or not
        :param item:
        :return:
        """
        return item in self.element2id

    def __len__(self):
        return len(self.element2id)

    def id2element(self, element_id):
        """
        :param element_id: (int) the id of the element for which to return the value
        :return: the element associated with index element_id in the dictionary
        """
        return self.id2element[element_id]

    def get_pad(self):
        return self.id2element[0]

    # Add count with default value as 1 to use in conjuction with a Counter
    def add(self, potential_element, count):
        if potential_element not in self:
            element_id = self.element2id[potential_element] = len(self)
            self.id2element[element_id] = potential_element
            self.counts += [count]
            return element_id
        else:
            self.counts[self[potential_element]] += count
            return self[potential_element]

    # Given a list of elements, return a list of indices by looking into element2id
    def elements2indices(self, elements: List):
        if type(elements[0]) == list:
            return [[self[el] for el in sub_elements] for sub_elements in elements]
        else:
            return [self[el] for el in elements]

    def to_input_tensor(self, elements: List, device: torch.device):
        """
        Convert list of sentences (sentence = list of words) into tensor with necessary padding for shorter sentences
        :param sentences: (List[List]) list of sentences
        :param device: on which device to return the result
        :return: tensor of (batch, max_sentence_length)
        """
        elements_ids = self.elements2indices(elements)
        if type(elements[0]) == list:
            elements_ids = pad_sentence(elements_ids, self['<pad>'])
        return torch.tensor(elements_ids, dtype=torch.long, device=device)

    def to_input_array(self, elements: List):
        """
        Convert list of sentences (sentence = list of words) into tensor with necessary padding for shorter sentences
        :param sentences: (List[List]) list of sentences
        :return: array of (batch, max_sentence_length)
        """
        elements_ids = self.elements2indices(elements)
        if type(elements[0]) == list:
            elements_ids = pad_sentence(elements_ids, self['<pad>'])
        return np.array(elements_ids, dtype=int)

    def save(self, filepath):
        """
        :param filepath: (str) where to save the vocabulary
        """
        json.dump(dict(word2id=self.element2id, counts=self.counts),
                  open(filepath, "w+"), indent=2)

    @staticmethod
    def load(filepath):
        """
        :param filepath: (str) from where to load the vocabulary
        :return Vocabulary loaded
        """
        loaded = json.load(open(filepath, "r"))
        return AbstractVocabulary(dict(element2id=loaded['word2id'], counts=loaded['counts']))
