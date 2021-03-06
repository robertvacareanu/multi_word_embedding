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
        self.train_path = train_path
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

        # train_x1 = [self.train_dataset[x][0].split(" ") for x in range(len(self.train_dataset))]
        # train_y1 = [self.train_dataset[x][1] for x in range(len(self.train_dataset))]
        # dev_x1 = [self.dev_dataset[x][0].split(" ") for x in range(len(self.dev_dataset))]
        # dev_y1 = [self.dev_dataset[x][1] for x in range(len(self.dev_dataset))]

        train_x = self.vocabulary.to_input_tensor(train_x, self.embedding_device)
        train_y = self.label_vocab.to_input_tensor(train_y, self.embedding_device).cpu().detach().numpy()
        dev_x = self.vocabulary.to_input_tensor(dev_x, self.embedding_device)
        dev_y = self.label_vocab.to_input_tensor(dev_y, self.embedding_device).cpu().detach().numpy()
        xyz = self.embedding_function.center_embeddings(train_x)
        train_x_mwe = self.evaluation_model(self.embedding_function.center_embeddings(train_x).to(self.device), [2]*train_x.shape[0]).cpu().detach().numpy()
        dev_x_mwe = self.evaluation_model(self.embedding_function.center_embeddings(dev_x).to(self.device), [2]*dev_x.shape[0]).cpu().detach().numpy()
        

        # pth = self.train_path.split('/')[-1].split('.')[0]
        # x = {}
        # x['train'] = {}
        # x['test'] = {}
        # x['train']['x'] = {}
        # x['test']['x'] = {}
        # x['train']['y'] = {}
        # x['test']['y'] = {}
        # for i, tx in enumerate(train_x1):
            # x['train']['x']['_'.join(tx)] = train_x_mwe[i]
        # for i, tx in enumerate(dev_x1):
            # x['test']['x']['_'.join(tx)] = dev_x_mwe[i]
        # for tx, lx in zip(train_x1, train_y1):
            # x['train']['y']['_'.join(tx)] = lx
        # for tx, lx in zip(dev_x1, dev_y1):
            # x['test']['y']['_'.join(tx)] = lx
        # import pickle
        # with open('zz_withoutcontext1', 'wb') as fin:
            # pickle.dump(x, fin)
        # exit()

        # print(train_x_mwe.shape)
        # print(dev_x_mwe.shape)
        # print(train_y.shape)
        # print(dev_y.shape)
        # exit()
        # print(f"{train_x_mwe.shape}, {train_y.shape}")
        self.te.fit(train_x_mwe, train_y)
        prediction = self.te.predict(dev_x_mwe)
        # print(prediction.tolist())
        # print(dev_y.tolist())
        # report = metrics.classification.classification_report(dev_y, predict)
        return prediction, dev_y, precision_score(dev_y, prediction, average='micro'), recall_score(dev_y, prediction, average='micro'), f1_score(dev_y, prediction, average='micro'), precision_score(dev_y, prediction, average='macro'), recall_score(dev_y, prediction, average='macro'), f1_score(dev_y, prediction, average='weighted')


    def evaluateWithContext(self, new_train_x, new_train_y, context, test_sentences='/work/rvacarenu/research/linnaeus/test_sentences4'):
        def prepareContext(line, embeddings, vocabulary, embedding_device, device, window_size=2):
            words = line.split(' ')
            if len(list(filter(lambda x: '_' in x, words))) == 0:
                print(line)
                exit()
            entity = list(filter(lambda x: '_' in x, words))[0]
            index = words.index(entity)
            span = (index, index+1) # Because the mwe are merged with '_', making them, essentially, as a single word

            lc = words[max(0, span[0]-window_size):span[0]]
            rc = words[span[1]:span[1]+window_size]
            entity = entity.split('_')
            left_sentence_vectorized = torch.tensor(vocabulary.to_input_array(words[0:span[0]] + entity))
            right_sentence_vectorized = torch.tensor(vocabulary.to_input_array(list(reversed(entity + words[span[1]:]))))
            # print(left_sentence_vectorized.shape)
            # print(right_sentence_vectorized.shape)
            # print(embeddings.center_embeddings(left_sentence_vectorized.to(embedding_device)).shape)
            # print(embeddings.center_embeddings(right_sentence_vectorized.to(embedding_device)).shape)
            left_part_embeddings = embeddings.center_embeddings(left_sentence_vectorized.to(embedding_device)).to(device).unsqueeze(0)
            right_part_embeddings = embeddings.center_embeddings(right_sentence_vectorized.to(embedding_device)).to(device).unsqueeze(0)
            return left_part_embeddings, right_part_embeddings, len(left_sentence_vectorized), len(right_sentence_vectorized)

        train_x = ['_'.join(self.train_dataset[x][0].split(" ")) for x in range(len(self.train_dataset))]
        train_y = [self.train_dataset[x][1] for x in range(len(self.train_dataset))]
        dev_x = ['_'.join(self.dev_dataset[x][0].split(" ")) for x in range(len(self.dev_dataset))]
        dev_y = [self.dev_dataset[x][1] for x in range(len(self.dev_dataset))]


        train_y = self.label_vocab.to_input_tensor(train_y, self.embedding_device).cpu().detach().numpy()
        dev_y = self.label_vocab.to_input_tensor(dev_y, self.embedding_device).cpu().detach().numpy()
        # /work/rvacarenu/research/linnaeus/test_sentences

        if new_train_x is None and new_train_y is None and context is None:
            context = open(test_sentences).readlines()
            context = dict([[x.split('\t')[0], [z.strip() for z in x.split('\t')[1:] if z.strip() != '' and '_' in z]] for x in context])

            # construct new train, where each sentence will get the same label
            new_train_x = []
            # new_train_x_lstm2 = []
            new_train_y = []
            for tx, lx in zip(train_x, train_y):
                if context[tx] != [''] and len(context[tx]) > 0:
                    for sentence in context[tx][:10]:
                        if len(list(filter(lambda x: '_' in x, sentence.split(' ')))) == 0:
                            print(tx)
                            print(sentence)
                            print(len(context[tx]))
                            print(context[tx][:5])
                            exit()
                        left, right, ll, rl = prepareContext(sentence, self.embedding_function, self.vocabulary, self.embedding_device, self.device)
                        new_train_x.append(((self.evaluation_model.forward(left, [ll], which_lstm=1) + self.evaluation_model.forward(right, [rl], which_lstm=2)) / 2).cpu().detach().numpy())
                        new_train_y.append(lx)
                else:
                    mwe = self.vocabulary.to_input_tensor([tx.split('_')], self.embedding_device)
                    mwe = self.embedding_function.center_embeddings(mwe).to(self.device)
                    new_train_x.append(self.evaluation_model.forward(mwe, [2], which_lstm=0).cpu().detach().numpy())
                    new_train_y.append(lx)

            new_train_x = np.array(new_train_x).squeeze(1)
            new_train_y = np.array(new_train_y)#.squeeze(1)
        # print(new_train_x)
        # print(new_train_y)

        # print(np.array(new_train_x).shape)
        # print(np.array(new_train_y).shape)
        # exit()
        from datetime import datetime
        print(datetime.now())
        self.te.fit(new_train_x, new_train_y)
        print(datetime.now())



        # Prediction
        prediction = []
        # print(len(list(zip(dev_x, dev_y))))
        for dx, lx in zip(dev_x, dev_y):
            current_dx = []
            if context[dx] != [''] and len(context[dx]) > 0:
                for sentence in context[dx][:10]:
                    left, right, ll, rl = prepareContext(sentence, self.embedding_function, self.vocabulary, self.embedding_device, self.device)
                    current_dx.append(((self.evaluation_model.forward(left, [ll], which_lstm=1) + self.evaluation_model.forward(right, [rl], which_lstm=2)) / 2).cpu().detach().numpy())
                current_dx = np.array(current_dx)#.squeeze(1)
                current_dx = current_dx.squeeze(1)
                prediction_current_dx = self.te.predict(current_dx)
                proba = self.te.predict_proba(current_dx)
                proba = (1 - (1-proba).prod(axis=0))

                prediction.append(self.te.classes_[proba.argmax()])
            else:
                mwe = self.vocabulary.to_input_tensor([dx.split(' ')], self.embedding_device)
                mwe = self.embedding_function.center_embeddings(mwe).to(self.device)
                mwe_dx = self.evaluation_model.forward(mwe, [2], which_lstm=0).cpu().detach().numpy()
                # print(f"B:{mwe_dx.shape}")
                # print(mwe_dx.shape)
                # print(f"bpredicted: {self.te.predict(mwe_dx)}")
                prediction.append(self.te.predict(mwe_dx)[0])
        
        print(f1_score(dev_y, prediction, average="weighted"))

        return np.array(prediction), np.array(dev_y), new_train_x, new_train_y, context, f1_score(dev_y, prediction, average='weighted')


    # Different aggregators
    def evaluateWithContext2(self, new_train_x, new_train_y, context, test_sentences='/work/rvacarenu/research/linnaeus/test_sentences6'):
        def prepareContext(line, embeddings, vocabulary, embedding_device, device, window_size=2):
            words = line.split(' ')
            if len(list(filter(lambda x: '_' in x, words))) == 0:
                print(line)
                exit()
            entity = list(filter(lambda x: '_' in x, words))[0]
            index = words.index(entity)
            span = (index, index+1) # Because the mwe are merged with '_', making them, essentially, as a single word

            lc = words[max(0, span[0]-window_size):span[0]]
            rc = words[span[1]:span[1]+window_size]
            entity = entity.split('_')
            left_sentence_vectorized = torch.tensor(vocabulary.to_input_array(words[0:span[0]] + entity))
            right_sentence_vectorized = torch.tensor(vocabulary.to_input_array(list(reversed(entity + words[span[1]:]))))
            # print(left_sentence_vectorized.shape)
            # print(right_sentence_vectorized.shape)
            # print(embeddings.center_embeddings(left_sentence_vectorized.to(embedding_device)).shape)
            # print(embeddings.center_embeddings(right_sentence_vectorized.to(embedding_device)).shape)
            left_part_embeddings = embeddings.center_embeddings(left_sentence_vectorized.to(embedding_device)).to(device).unsqueeze(0)
            right_part_embeddings = embeddings.center_embeddings(right_sentence_vectorized.to(embedding_device)).to(device).unsqueeze(0)
            return left_part_embeddings, right_part_embeddings, len(left_sentence_vectorized), len(right_sentence_vectorized)

        train_x = ['_'.join(self.train_dataset[x][0].split(" ")) for x in range(len(self.train_dataset))]
        train_y = [self.train_dataset[x][1] for x in range(len(self.train_dataset))]
        dev_x = ['_'.join(self.dev_dataset[x][0].split(" ")) for x in range(len(self.dev_dataset))]
        dev_y = [self.dev_dataset[x][1] for x in range(len(self.dev_dataset))]

        train_y = self.label_vocab.to_input_tensor(train_y, self.embedding_device).cpu().detach().numpy()
        dev_y = self.label_vocab.to_input_tensor(dev_y, self.embedding_device).cpu().detach().numpy()
        # /work/rvacarenu/research/linnaeus/test_sentences

        if new_train_x is None and new_train_y is None and context is None:
            context = open(test_sentences).readlines()
            context = dict([[x.split('\t')[0], [z.strip() for z in x.split('\t')[1:] if z.strip() != '' and '_' in z]] for x in context])

            # construct new train, where each sentence will get the same label
            new_train_x = []
            # new_train_x_lstm2 = []
            new_train_y = []
            for tx, lx in zip(train_x, train_y):
                if context[tx] != [''] and len(context[tx]) > 0:
                    current_training_point = []
                    for sentence in context[tx]:
                        if len(list(filter(lambda x: '_' in x, sentence.split(' ')))) == 0:
                            print(tx)
                            print(sentence)
                            print(len(context[tx]))
                            print(context[tx][:5])
                            exit()
                        left, right, ll, rl = prepareContext(sentence, self.embedding_function, self.vocabulary, self.embedding_device, self.device)
                        current_training_point.append(((self.evaluation_model.forward(left, [ll], which_lstm=1) + self.evaluation_model.forward(right, [rl], which_lstm=2)) / 2).cpu().detach().numpy())
                    # Aggregation happens here:
                    new_train_x.append(np.mean(current_training_point, axis=0))
                else:
                    mwe = self.vocabulary.to_input_tensor([tx.split('_')], self.embedding_device)
                    mwe = self.embedding_function.center_embeddings(mwe).to(self.device)
                    new_train_x.append(self.evaluation_model.forward(mwe, [2], which_lstm=0).cpu().detach().numpy())
                    # new_train_y.append(lx)
                new_train_y.append(lx)
            new_train_x = np.array(new_train_x).squeeze(1)
            new_train_y = np.array(new_train_y)#.squeeze(1)

        from datetime import datetime
        print(datetime.now())
        self.te.fit(new_train_x, new_train_y)
        print(datetime.now())



        # Prediction
        prediction = []
        new_dev_x = []
        # print(len(list(zip(dev_x, dev_y))))
        for dx, lx in zip(dev_x, dev_y):
            current_dx = []
            if context[dx] != [''] and len(context[dx]) > 0:
                current_point = []
                for sentence in context[dx]:
                    left, right, ll, rl = prepareContext(sentence, self.embedding_function, self.vocabulary, self.embedding_device, self.device)
                    current_point.append(((self.evaluation_model.forward(left, [ll], which_lstm=1) + self.evaluation_model.forward(right, [rl], which_lstm=2)) / 2).cpu().detach().numpy())

                new_dev_x.append(np.mean(current_point, axis=0))
            else:
                mwe = self.vocabulary.to_input_tensor([dx.split(' ')], self.embedding_device)
                mwe = self.embedding_function.center_embeddings(mwe).to(self.device)
                mwe_dx = self.evaluation_model.forward(mwe, [2], which_lstm=0).cpu().detach().numpy()
                # print(f"B:{mwe_dx.shape}")
                # print(mwe_dx.shape)
                # print(f"bpredicted: {self.te.predict(mwe_dx)}")
                new_dev_x.append(self.evaluation_model.forward(mwe, [2], which_lstm=0).cpu().detach().numpy())
                # prediction.append(self.te.predict(mwe_dx)[0])
        new_dev_x = np.array(new_dev_x)
        # print(new_dev_x.shape)
        new_dev_x = new_dev_x.squeeze(1)
        # print(new_dev_x.shape)
        # exit()
        prediction = self.te.predict(new_dev_x)
        print(f1_score(dev_y, prediction, average="weighted"))

        return np.array(prediction), np.array(dev_y), new_train_x, new_train_y, context, f1_score(dev_y, prediction, average='weighted')


    def evaluate2(self, dump_path):
        # THIS function is used only to dump to a file the predictions for a specified model
        # TODO Maybe take into consideration all? (now only considers those with size 2 -- not clear in the paper)
        train_x = [self.train_dataset[x][0].split(" ") for x in range(len(self.train_dataset))]
        train_y = [self.train_dataset[x][1] for x in range(len(self.train_dataset))]

        train_x1 = [self.train_dataset[x][0].split(" ") for x in range(len(self.train_dataset))]
        train_y1 = [self.train_dataset[x][1] for x in range(len(self.train_dataset))]

        train_x = self.vocabulary.to_input_tensor(train_x, self.embedding_device)
        train_y = self.label_vocab.to_input_tensor(train_y, self.embedding_device).cpu().detach().numpy()
        train_x_mwe = self.evaluation_model(self.embedding_function.center_embeddings(train_x).to(self.device), [2]*train_x.shape[0]).cpu().detach().numpy()

        pth = self.train_path.split('/')[-1].split('.')[0]
        x = {}
        x['x'] = {}
        x['y'] = {}
        for i, tx in enumerate(train_x1):
            x['x']['_'.join(tx)] = train_x_mwe[i]
        for tx, lx in zip(train_x1, train_y1):
            x['y']['_'.join(tx)] = lx
        return x
        import pickle
        with open(dump_path, 'wb') as fin:
            pickle.dump(x, fin)
        return

    # Only used to dump to file
    def evaluateWithContext22(self, new_train_x, new_train_y, context, dump_path, test_sentences='/work/rvacarenu/research/linnaeus/test_sentences6'):
        def prepareContext(line, embeddings, vocabulary, embedding_device, device, window_size=2):
            words = line.split(' ')
            if len(list(filter(lambda x: '_' in x, words))) == 0:
                print(line)
                exit()
            entity = list(filter(lambda x: '_' in x, words))[0]
            index = words.index(entity)
            span = (index, index+1) # Because the mwe are merged with '_', making them, essentially, as a single word

            lc = words[max(0, span[0]-window_size):span[0]]
            rc = words[span[1]:span[1]+window_size]
            entity = entity.split('_')
            left_sentence_vectorized = torch.tensor(vocabulary.to_input_array(words[0:span[0]] + entity))
            right_sentence_vectorized = torch.tensor(vocabulary.to_input_array(list(reversed(entity + words[span[1]:]))))
            # print(left_sentence_vectorized.shape)
            # print(right_sentence_vectorized.shape)
            # print(embeddings.center_embeddings(left_sentence_vectorized.to(embedding_device)).shape)
            # print(embeddings.center_embeddings(right_sentence_vectorized.to(embedding_device)).shape)
            left_part_embeddings = embeddings.center_embeddings(left_sentence_vectorized.to(embedding_device)).to(device).unsqueeze(0)
            right_part_embeddings = embeddings.center_embeddings(right_sentence_vectorized.to(embedding_device)).to(device).unsqueeze(0)
            return left_part_embeddings, right_part_embeddings, len(left_sentence_vectorized), len(right_sentence_vectorized)

        train_x = ['_'.join(self.train_dataset[x][0].split(" ")) for x in range(len(self.train_dataset))]
        train_y = [self.train_dataset[x][1] for x in range(len(self.train_dataset))]
        
        train_x1 = ['_'.join(self.train_dataset[x][0].split(" ")) for x in range(len(self.train_dataset))]
        train_y1 = [self.train_dataset[x][1] for x in range(len(self.train_dataset))]

        dev_x = ['_'.join(self.dev_dataset[x][0].split(" ")) for x in range(len(self.dev_dataset))]
        dev_y = [self.dev_dataset[x][1] for x in range(len(self.dev_dataset))]

        train_y = self.label_vocab.to_input_tensor(train_y, self.embedding_device).cpu().detach().numpy()
        dev_y = self.label_vocab.to_input_tensor(dev_y, self.embedding_device).cpu().detach().numpy()
        # /work/rvacarenu/research/linnaeus/test_sentences
        context = open(test_sentences).readlines()
        context = dict([[x.split('\t')[0], [z.strip() for z in x.split('\t')[1:] if z.strip() != '' and '_' in z]] for x in context])

        pth = self.train_path.split('/')[-1].split('.')[0]
        x = {}
        x['x'] = {}
        x['y'] = {}
        for tx, lx in zip(train_x1, train_y1):
            if context[tx] != [''] and len(context[tx]) > 0:
                current_training_point = []
                for sentence in context[tx]:
                    if len(list(filter(lambda x: '_' in x, sentence.split(' ')))) == 0:
                        print(tx)
                        print(sentence)
                        print(len(context[tx]))
                        print(context[tx][:5])
                        exit()
                    left, right, ll, rl = prepareContext(sentence, self.embedding_function, self.vocabulary, self.embedding_device, self.device)
                    current_training_point.append(((self.evaluation_model.forward(left, [ll], which_lstm=1) + self.evaluation_model.forward(right, [rl], which_lstm=2)) / 2).cpu().detach().numpy())
                x['x'][tx] = current_training_point
            else:
                mwe = self.vocabulary.to_input_tensor([tx.split('_')], self.embedding_device)
                mwe = self.embedding_function.center_embeddings(mwe).to(self.device)
                x['x'][tx] = [self.evaluation_model.forward(mwe, [2], which_lstm=0).cpu().detach().numpy()]
                # new_train_x.append(self.evaluation_model.forward(mwe, [2], which_lstm=0).cpu().detach().numpy())
            x['y'][tx] = lx
        return x
        import pickle
        with open(dump_path, 'wb') as fin:
            pickle.dump(x, fin)
        return 0


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