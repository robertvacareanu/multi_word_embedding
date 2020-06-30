import json
from typing import List

import pandas as pd
import torch
from nltk.tokenize import word_tokenize

from utils import pad_sentence


def read_wikipedia_corpus(path):
    for line in open(path):
        words = line.lower()
        words = word_tokenize(words)
        words = list(words)
        yield words


def read_tratz(path):
    dataset = pd.read_csv(path, header=None, quoting=3, sep='\t', names=['constituent1', 'constituent2', 'label'])
    dataset['mwe'] = dataset['constituent1'] + ' ' + dataset['constituent2']
    dataset = dataset.drop(columns=['constituent1', 'constituent2'])[['mwe', 'label']].values
    for label in dataset[:, 1]:
        yield label


def make_word_vocab(path, reader):
    vocab = AbstractVocabulary(None)
    for sentence in reader(path):
        for word in sentence:
            vocab.add(word)
    return vocab


def make_label_vocab(path, reader):
    vocab = AbstractVocabulary(None)
    for label in reader(path):
        vocab.add(label)
    return vocab


"""
Class representing a vocabulary.
Maps from element to index
"""


class AbstractVocabulary(object):
    def __init__(self, saved_version):
        if saved_version:
            self.element2id = saved_version['element2id']
            self.counts = saved_version['counts']
        else:
            self.element2id = dict()
            self.element2id['<pad>'] = 0
            self.element2id['<unk>'] = 1
            self.counts = [0] * len(self.element2id)
        self.unk_id = self.element2id['<unk>']
        self.id2element = {v: k for k, v in self.element2id.items()}

    def id2element(self, element_id):
        """

        :param element_id: (int) the id of the element for which to return the value
        :return: the element associated with index element_id in the dictionary
        """
        return self.id2element[element_id]

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

    def get_pad(self):
        return '<pad>'

    def add(self, potential_element):
        if potential_element not in self:
            element_id = self.element2id[potential_element] = len(self)
            self.id2element[element_id] = potential_element
            self.counts += [1]
            return element_id
        else:
            self.counts[self[potential_element]] += 1
            return self[potential_element]

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

    def save(self, filepath):
        """
        :param filepath: (str) where to save the vocabulary
        """
        json.dump(dict(word2id=self.element2id, counts=self.counts), open(filepath, "w+"), indent=2)

    @staticmethod
    def load(filepath):
        """
        :param filepath: (str) from where to load the vocabulary
        :return Vocabulary loaded
        """
        loaded = json.load(open(filepath, "r"))
        return AbstractVocabulary(dict(word2id=loaded['elements2id'], counts=loaded['counts']))

