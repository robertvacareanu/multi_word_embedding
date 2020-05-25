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

        
class Evaluation3(object):
    def __init__(self, mwe_f, embedding_f, te, params):
        self.mwe_f = mwe_f
        self.embedding_f = embedding_f
        self.te = te
        self.params = params

    def evaluate(self):
        self.te.train(TratzDataset(self.params['train_path'], self.params['dev_path'], make_label_vocab(self.params['train_path'], read_tratz)))

class Evaluation(object):

    def __init__(self, train_path, dev_path, model, embedding_function, vocabulary_path, batch_size=64, epochs=100,
                 device=torch.device('cpu')):
        self.evaluation_model = model
        self.embedding_function = embedding_function
        self.label_vocab = make_label_vocab(train_path, read_tratz)
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.model = model

        self.te = TratzEvaluation(self.model, self.embedding_function, self.label_vocab).to(device)
        self.train_dataset = TratzDataset(train_path)
        self.dev_dataset = TratzDataset(dev_path)
        self.optimizer = torch.optim.Adam(self.te.ll.parameters(), lr=0.001, weight_decay=0.00001)
        self.vocabulary = AbstractVocabulary.load(vocabulary_path)

    def train(self):
        
        generator = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=16,
                                                collate_fn=lambda nparr: np.vstack(nparr))

        for _ in range(self.epochs):
            for batch in generator:
                self.optimizer.zero_grad()
                mwe = [x.split(' ') for x in batch[:, 0]]
                mwe = self.vocabulary.to_input_tensor(mwe, self.device)
                labels = batch[:, 1]
                predictions = self.te.forward(mwe)
                loss = F.cross_entropy(predictions, self.label_vocab.to_input_tensor(labels, self.device))

                loss.backward()

                self.optimizer.step()

    def evaluate(self):
        self.train()
        torch.save(self.te, 'model.pt')
        self.te.eval()
        saved_loss = 0
        correct_items = 0

        gl = np.array([])
        pl = np.array([])

        for batch in torch.utils.data.DataLoader(self.dev_dataset, batch_size=self.batch_size, num_workers=16, shuffle=False,
                                                 collate_fn=lambda nparr: np.vstack(nparr)):
            mwe = [x.split(' ') for x in batch[:, 0]]
            mwe = self.vocabulary.to_input_tensor(mwe, self.device)

            labels = batch[:, 1]
            predictions = self.te.forward(mwe)

            gold_labels = self.label_vocab.to_input_tensor(labels, self.device)
            gl = np.concatenate([gl, gold_labels.cpu().numpy()])
            loss = F.cross_entropy(predictions, gold_labels)
            saved_loss += loss.item()
            predicted_labels = torch.max(predictions, 1)[1]
            pl = np.concatenate([pl, predicted_labels.cpu().numpy()])

            correct_items += torch.sum(predicted_labels == gold_labels)

        return saved_loss, correct_items, len(self.dev_dataset), correct_items/len(self.dev_dataset), precision_score(gl, pl, average='micro'), recall_score(gl, pl, average='micro'), f1_score(gl, pl, average='micro'), precision_score(gl, pl, average='macro'), recall_score(gl, pl, average='macro'), f1_score(gl, pl, average='macro')

# Somehow merge Evaluation2 and Evaluation1 under same object
# Maybe receive as parameter a dictionary for usual stuff such as: train, dev paths, vocabulary path, batch size, epochs, device_name etc
# Receive embedding function and evaluation_model and which model to use
# ------------------------------------ total: 3 parameters (mwe, emb_f, tratz_evaluation_model, dictionary)
# 
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

        # print(train_x_mwe.shape)
        # print(dev_x_mwe.shape)
        # print(train_y.shape)
        # print(dev_y.shape)
        # exit()
        # print(f"{train_x_mwe.shape}, {train_y.shape}")
        self.te.fit(train_x_mwe, train_y)

        prediction = self.te.predict(dev_x_mwe)
        # report = metrics.classification.classification_report(dev_y, predict)
        return prediction, dev_y, precision_score(dev_y, prediction, average='micro'), recall_score(dev_y, prediction, average='micro'), f1_score(dev_y, prediction, average='micro'), precision_score(dev_y, prediction, average='macro'), recall_score(dev_y, prediction, average='macro'), f1_score(dev_y, prediction, average='weighted')


# Should be updated with the new way evaluation is performed or deleted
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entry point of the application.")

    parser.add_argument("--learning-rate", type=float, required=False, default=0.1,
                        help="Learning rate to pass to the optimizer")
    parser.add_argument("--embeddings-path", type=str, required=True, help="Path to the saved embeddings file")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model file")
    parser.add_argument("--train-file", type=str, required=True,
                        help="Path to the file containing the data to train on")
    parser.add_argument("--dev-file", type=str, required=True,
                        help="Path to the file containing the data to evaluate on")
    parser.add_argument("--batch-size", type=int, required=False, default=64,
                        help="Number of examples to consider per batch")
    parser.add_argument("--num-epochs", type=int, required=False, default=10)
    parser.add_argument("--hidden-size", type=int, required=False, default=500)
    parser.add_argument("--weight-decay", type=float, required=False, default=0.001)

    result = parser.parse_args()
    init_random(1)
    device = torch.device('cpu')
    model = LSTMMultiply(300, result.hidden_size)
    model.load_state_dict(torch.load(result.model_path, map_location=device))
    sg_embeddings = torch.load(result.embeddings_path, map_location=device)
    evaluation = Evaluation(result.train_file, result.dev_file, model, sg_embeddings, 'vocab/vocab_250.json')

    print(evaluation.evaluate())