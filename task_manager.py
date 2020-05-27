import argparse

import json

import numpy as np
import time
import torch
import torch.optim
import tqdm
import os
from torch.utils import data
from datetime import datetime

from dataset import WordWiseSGMweDataset, SkipGramMinimizationDataset, JointTrainingSGMinimizationDataset, SentenceWiseSGMweDataset, DirectMinimizationDataset
from embeddings import SkipGramEmbeddings, RandomInitializedEmbeddings
from evaluate import Evaluation, Evaluation2
from mwe_function_model import LSTMMultiply, LSTMMweModel, FullAdd, MultiplyMean, MultiplyAdd, CNNMweModel, LSTMMweModelPool, GRUMweModel, AttentionWeightedModel
from task_model import MWESkipGramTaskModel, MWEMeanSquareErrorTaskModel, MWESentenceSkipGramTaskModel, MWEJointTraining
from utils import init_random, format_number, read_wikipedia_corpus, flatten
from vocabulary import AbstractVocabulary, make_word_vocab
from baseline import Max, Average, RandomLSTM
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


class TaskManager():
    def __init__(self):
        # self.task_model = task_model_dict['train_task']
        # self.pretrain_task_model = task_model_dict['pretrain']
        # self.batch_construction = task_model_dict['batch']
        self.debug = False
    def pre_train(self):
        pass

    def train(self):
        pass
    
    """
    Train a single epoch with the given task_model
    :param task_model - something that is capable of returning a loss value, calculated using the batch
    :param generator - something that returns batches
    :param optimizer - something capable of modifying the weights (of the task_model, but depends on how it was constructed) based on the loss value
    :param batch_construction - transforms the batch into the expected input of the task_model
    :returns the mean loss over the entire epoch
    """
    def step(self, task_model, generator, optimizer, batch_construction):
        running_epoch_loss = []
        with tqdm.tqdm(total=len(generator)) as progress_bar:
            for batch in generator:
                progress_bar.update(1)

                # Zero the gradients
                optimizer.zero_grad()

                # Get the loss
                loss = task_model.forward(*batch_construction(batch)) # Prepare each batch based on what type of training is used
                loss_item = loss.item()

                if self.debug:
                    print(loss_item)
                # running_batch_loss.append(loss_item)

                # Get gradients
                loss.backward()

                # Grad clipping, same as in the paper we are comparing with
                torch.nn.utils.clip_grad_norm_(task_model.parameters(), 5)

                # Take gradient step
                optimizer.step()

                running_epoch_loss.append(loss_item)

        # print(f"REL: {round(np.mean(running_epoch_loss), 5)} - {[round(x, 5) for x in running_epoch_loss[:20]]} {round(running_epoch_loss[-1], 5)}")
        return running_epoch_loss
            
    def evaluateOnTratz(self, params, mwe_f, sg_embeddings, embedding_device, device):
        scores = []

        models = [
            LogisticRegression(multi_class="multinomial", penalty='l2', C=0.5, solver="sag", n_jobs=20, max_iter=500,),
            LogisticRegression(multi_class="multinomial", penalty='l2', C=1,   solver="sag", n_jobs=20, max_iter=500,),
            LogisticRegression(multi_class="multinomial", penalty='l2', C=2,   solver="sag", n_jobs=20, max_iter=500,),
            LogisticRegression(multi_class="multinomial", penalty='l2', C=5,   solver="sag", n_jobs=20, max_iter=500,),
            LogisticRegression(multi_class="multinomial", penalty='l2', C=10,  solver="sag", n_jobs=20, max_iter=500,),
            LinearSVC(penalty='l2', dual=False, C=0.5, max_iter=500,),
            LinearSVC(penalty='l2', dual=False, C=1,   max_iter=500,),
            LinearSVC(penalty='l2', dual=False, C=2,   max_iter=500,),
            LinearSVC(penalty='l2', dual=False, C=5,   max_iter=500,),
            LinearSVC(penalty='l2', dual=False, C=10,  max_iter=500,)
            
            # LogisticRegression(multi_class="multinomial", solver="sag", n_jobs=20)
        ]
        for model in models:
            evaluation = Evaluation2(params['evaluation']['evaluation_train_file'],
                            params['evaluation']['evaluation_dev_file'], 
                            mwe_f, sg_embeddings, params['vocabulary_path'], embedding_device=embedding_device, device=device, te=model)
            score = float(evaluation.evaluate()[-1])
            scores.append(score)
        
        return np.max(scores), np.argmax(scores)

    def evaluateOnTratzBestModel(self, params, mwe_f, sg_embeddings, embedding_device, device):
        scores = []

        models = [
            LogisticRegression(multi_class="multinomial", penalty='l2', C=0.5, solver="sag", n_jobs=20, max_iter=500,),
            LogisticRegression(multi_class="multinomial", penalty='l2', C=1,   solver="sag", n_jobs=20, max_iter=500,),
            LogisticRegression(multi_class="multinomial", penalty='l2', C=2,   solver="sag", n_jobs=20, max_iter=500,),
            LogisticRegression(multi_class="multinomial", penalty='l2', C=5,   solver="sag", n_jobs=20, max_iter=500,),
            LogisticRegression(multi_class="multinomial", penalty='l2', C=10,  solver="sag", n_jobs=20, max_iter=500,),
            LinearSVC(penalty='l2', dual=False, C=0.5, max_iter=500,),
            LinearSVC(penalty='l2', dual=False, C=1,   max_iter=500,),
            LinearSVC(penalty='l2', dual=False, C=2,   max_iter=500,),
            LinearSVC(penalty='l2', dual=False, C=5,   max_iter=500,),
            LinearSVC(penalty='l2', dual=False, C=10,  max_iter=500,)
            
            # LogisticRegression(multi_class="multinomial", solver="sag", n_jobs=20)
        ]
        # TODO replace this with more generic (train,val,test as params). Used this right now to avoid rewriting the config files
        val_path_splitted = params['evaluation']['evaluation_dev_file'].split('/')
        val_path = '/'.join(val_path_splitted[:-1]) + '/val.tsv'
        for model in models:
            evaluation = Evaluation2(params['evaluation']['evaluation_train_file'],
                            val_path, 
                            mwe_f, sg_embeddings, params['vocabulary_path'], embedding_device=embedding_device, device=device, te=model)
            score = float(evaluation.evaluate()[-1])
            scores.append(score)
        
        best_model = np.argmax(scores)
        print(f"Best model on dev: {best_model} with {np.max(scores)}")
        
        evaluation = Evaluation2(params['evaluation']['evaluation_train_file'],
                            params['evaluation']['evaluation_dev_file'], 
                            mwe_f, sg_embeddings, params['vocabulary_path'], embedding_device=embedding_device, device=device, te=models[best_model])
        return float(evaluation.evaluate()[-1]), best_model

    def evaluateOnHeldoutDataset(self, params, task_model, generator, batch_construction):
        running_epoch_loss = []
        for batch in generator:
            loss = task_model.forward(*batch_construction(batch)) # Prepare each batch based on what type of training is used
            loss_item = loss.item()
            running_epoch_loss.append(loss_item)
            
        # print(f"REvL: {round(np.mean(running_epoch_loss), 5)} - {[round(x, 5) for x in running_epoch_loss[:20]]} {round(running_epoch_loss[-1], 5)}\n")
        
        return np.mean(running_epoch_loss)