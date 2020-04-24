import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from sklearn.metrics import recall_score, precision_score, f1_score
from dataset import TratzDataset
from embeddings import SkipGramEmbeddings
from utils import init_random, read_tratz
from mwe_function_model import LSTMMultiply
from vocabulary import make_label_vocab, AbstractVocabulary
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, svm
from baseline import Average, Max

class TratzEvaluation(nn.Module):

    def __init__(self, mwe, embedding_function, label_vocab):
        super().__init__()
        self.mwe = mwe
        self.embedding_function = embedding_function
        self.label_vocab = label_vocab
        self.ll = nn.Linear(in_features=embedding_function.embedding_size, out_features=len(label_vocab))

    def forward(self, input_data):
        """

        :param input_data: (batch_size, mwe_length=2) contains the multi-word entities for which to generate class label
        :return: (batch_size, len(label_vocab)) contains the class label for each multi_word entity
        """

        with torch.no_grad():
            # (batch_size, mwe_len=2, embedding_size)
            input_emb = self.embedding_function.center_embeddings(input_data)
            # (batch_size, embedding_size)
            input_emb = self.mwe(input_emb, [2]*input_data.shape[0])

        predictions = self.ll(input_emb)

        return predictions


class Evaluation2(object):

    def __init__(self, train_path, dev_path, model, embedding_function, vocabulary_path, batch_size=64, epochs=10, embedding_device = torch.device('cpu'),
                 device=torch.device('cpu'), te=LogisticRegression(multi_class="multinomial", solver="sag", n_jobs=20)):
        self.evaluation_model = model
        self.embedding_function = embedding_function
        self.label_vocab = make_label_vocab(train_path, read_tratz)
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.embedding_device = embedding_device
        # self.cpu_device = torch.device('cpu')

        self.te = te
        self.train_dataset = TratzDataset(train_path)
        self.dev_dataset = TratzDataset(dev_path)
        self.vocabulary = AbstractVocabulary.load(vocabulary_path)

    def evaluate(self):
        # TODO Maybe take into consideration all? (now only considers those with size 2 -- not clear in the paper)
        train_x = [self.train_dataset[x][0].split(" ") for x in range(len(self.train_dataset))]
        train_y = [self.train_dataset[x][1] for x in range(len(self.train_dataset))]
        dev_x = [self.dev_dataset[x][0].split(" ") for x in range(len(self.dev_dataset))]
        dev_y = [self.dev_dataset[x][1] for x in range(len(self.dev_dataset))]

        train_x = self.vocabulary.to_input_tensor(train_x, self.embedding_device)
        train_y = self.label_vocab.to_input_tensor(train_y, self.embedding_device).cpu().detach().numpy()
        dev_x = self.vocabulary.to_input_tensor(dev_x, self.embedding_device)
        dev_y = self.label_vocab.to_input_tensor(dev_y, self.embedding_device).cpu().detach().numpy()
        xyz = self.embedding_function.center_embeddings(train_x)
        train_x_mwe = self.evaluation_model(self.embedding_function.center_embeddings(train_x).to(self.device), [2]*train_x.shape[0]).cpu().detach().numpy()
        dev_x_mwe = self.evaluation_model(self.embedding_function.center_embeddings(dev_x).to(self.device), [2]*dev_x.shape[0]).cpu().detach().numpy()

        self.te.fit(train_x_mwe, train_y)
        prediction = self.te.predict(dev_x_mwe)
        return prediction, dev_y, precision_score(prediction, dev_y, average='micro'), recall_score(prediction, dev_y, average='micro'), f1_score(prediction, dev_y, average='micro'), precision_score(prediction, dev_y, average='macro'), recall_score(prediction, dev_y, average='macro'), f1_score(prediction, dev_y, average='weighted')
