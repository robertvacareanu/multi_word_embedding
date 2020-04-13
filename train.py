import argparse

import json

import numpy as np
import time
import torch.optim
import os
from torch.utils import data
from datetime import datetime

from dataset import EntitySentenceDataset, EntitySentenceDirectMinimizationDataset, JointTrainingSGMinimizationDataset
from embeddings import SkipGramEmbeddings, RandomInitializedEmbeddings
from evaluate import Evaluation, Evaluation2
from mwe_function_model import LSTMMultiply, LSTMMweModel, FullAdd, MultiplyMean, MultiplyAdd, CNNMweModel, LSTMMweModelPool, GRUMweModel
from task_model import MWESkipGramTaskModel, MWEMeanSquareErrorTaskModel, MWESentenceSkipGramTaskModel, MWEJointTraining
from utils import init_random, format_number, read_wikipedia_corpus, flatten
from vocabulary import AbstractVocabulary, make_word_vocab
from baseline import Max, Average, RandomLSTM
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

"""
All the orchestration happens here
To support multiple types of training regimes:
- Skipgram minimization
- Direct minimization
- Skipgram minimization over full sentences
All of them with different parameter expectations

And different mapping functions:
- LSTMs, CNNs, etc

A lot of dictionaries that map from config name to function had to be used
"""
class MWETrain(object):

    def __init__(self, params):
        self.params = params
        print(datetime.now())
        # dictionary containing the mappings from string to mwe function.
        mwe_function_map = {'LSTMMultiply': LSTMMultiply, 'LSTMMweModel': LSTMMweModel,
                            'FullAdd': FullAdd, 'MultiplyMean': MultiplyMean, 'MultiplyAdd': MultiplyAdd,
                            'CNNMweModel': CNNMweModel, 'LSTMMweModelPool': LSTMMweModelPool, 'GRUMweModel': GRUMweModel, 'Max': Max,
                            'Average': Average, 'RandomLSTM': RandomLSTM}

        # Different training regimes may need different data format (or more data)
        # This maps from a training regimes to its corresponding dataset
        dataset_function_map = {
            'DirectMinimization': EntitySentenceDirectMinimizationDataset, 
            'SkipGramMinimization': EntitySentenceDataset, 
            'SkipGramSentenceMinimization': EntitySentenceDataset,
            'JointTrainingSkipGramMinimization': JointTrainingSGMinimizationDataset}

        # Maps from a training regime to a function that knows how to prepare the expected output
        # DirectMinimization - Minimize the mean squared eXrror (squared L2 norm) between the learned embedding and the predicted embedding
        # SkipGramMinimization - A training objective similar to SkipGram, where the predicted embedding is then used to predict the surrounding words
        self.minimization_types = {'DirectMinimization': self.prepare_direct_minimzation,
                                   'SkipGramMinimization': self.prepare_skipgram_minimization,
                                   'SkipGramSentenceMinimization': self.prepare_sentence_skipgram_minimization,
                                   'JointTrainingSkipGramMinimization': self.prepare_joint_training}
        
        # The training regimes
        self.task_model_types = {
            'DirectMinimization': MWEMeanSquareErrorTaskModel, 
            'SkipGramMinimization': MWESkipGramTaskModel, 
            'SkipGramSentenceMinimization': MWESentenceSkipGramTaskModel, 
            'JointTrainingSkipGramMinimization': MWEJointTraining}

        self.learning_rate = params['learning_rate']

        # Create or load vocabulary
        if params['vocabulary_path'] is None:
            print("Compute vocabulary")
            self.vocabulary = make_word_vocab(
                params['train_file'], read_wikipedia_corpus)
            self.vocabulary.save("vocab.json")
        else:
            print("Load vocabulary")
            self.vocabulary = AbstractVocabulary.load(
                params['vocabulary_path'])

        print("Vocabulary - done")

        self.device = torch.device(f"cuda:{params['which_cuda']}" if torch.cuda.is_available(
        ) and params['which_cuda'] is not None else "cpu")
        self.num_epochs = params['num_epochs']
        self.save_path = params['save_path']
        self.batch_size = params['batch_size']
        self.hidden_size = params['model']['attributes']['hidden_size']
        # self.entity_path = params['entity_path']

        if params['train_objective'] == 'JointTrainingSkipGramMinimization':
            self.sg_embeddings = RandomInitializedEmbeddings(self.vocabulary)
        else:
            if params['load_embeddings']:
                print(f"Load embeddings from {params['load_embeddings']}")
                self.sg_embeddings = SkipGramEmbeddings.from_saved_file(
                    params['load_embeddings'])
                # self.sg_embeddings = RandomInitializedEmbeddings(self.vocabulary)
            else:
                sg_embeddings = SkipGramEmbeddings.from_embedding_file(
                    params['embeddings_path'], vocabulary).to(device)
                if params['save_embeddings']:
                    torch.save(self.sg_embeddings.state_dict(),
                                params['save_embeddings'])
                    exit()


        print("Embeddings - done")
        # init_random(1)
        self.mwe_f = mwe_function_map[params['model']['name']](
            params['model']['attributes']).to(self.device)
        if params['pretrained_model'] is not None:
            self.mwe_f.load_state_dict(torch.load(f"{params['pretrained_model']}.pt"))
        self.task_model = self.task_model_types[params['train_objective']](
            self.sg_embeddings, self.mwe_f).to(self.device)
        self.task_model.train()

        # self.optimizer = torch.optim.Adam(
        # self.task_model.parameters(), lr=self.learning_rate, weight_decay=params['weight_decay'])
        self.optimizer = torch.optim.Adagrad(
            self.task_model.parameters(), lr=self.learning_rate, weight_decay=params['weight_decay'])
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=3, min_lr=0.0005, cooldown=0, )
        
        self.dataset = dataset_function_map[params['train_objective']](
            params['train_file'], params['window_size'])#, params['train_objective']=='SkipGramSentenceMinimization')
        self.generator = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, num_workers=10, shuffle=False,
                                                     collate_fn=lambda nparr: nparr)
        print(len(self.generator))

        negative_sampling_distribution = torch.tensor(
            self.vocabulary.counts).float().to(self.device)
        negative_sampling_distribution = negative_sampling_distribution.pow(
            3 / 4)
        self.negative_sampling_distribution = negative_sampling_distribution.div(
            negative_sampling_distribution.sum(dim=0))
        self.number_of_negative_examples = params['number_of_negative_examples']
        self.number_of_iter = 1000
        self.debug = False

    def generate_negative_examples(self, batch_size, words_to_avoid):
        """
        :param batch_size: The number of samples in a batch
        :param words_to_avoid: The words for which to generate negative examples (context), of shape (batch_size * context_size)
        :return: a tensor containing the negative examples, of shape (batch_size * context_size, number_of_negative_examples)
        """
        # Generate negative samples
        negative_examples = torch.multinomial(self.negative_sampling_distribution, num_samples=self.number_of_negative_examples *
                                              words_to_avoid.shape[0], replacement=True).reshape(-1, self.number_of_negative_examples).to(self.device) # (batch * context_size, number_of_negative_examples)
        # Compare negative samples with the words that are indeed in the context (compare 2d tensor to 2d), a (batch_size, number_of_negative_samples) tensor to a (batch_size, 1) tensor
        indices_to_change = (negative_examples == (words_to_avoid.unsqueeze(dim=1))) == True
        while torch.any(indices_to_change):
            replacement = torch.multinomial(self.negative_sampling_distribution, num_samples=torch.nonzero(
                indices_to_change).shape[0], replacement=True).to(self.device)
            negative_examples[indices_to_change] = replacement
            indices_to_change = (negative_examples == (words_to_avoid.unsqueeze(dim=1))) == True

        return negative_examples

    def train(self):
        begin_time = time.time()
        running_loss = []
        train_iter = 0
        f = open('output', 'w+')
        performance = None
        epochs_since_last_improvement = 0
        for epoch in range(self.num_epochs):
            running_epoch_loss = []
            if epochs_since_last_improvement < self.params['early_stopping']:
                for batch in self.generator:
                    train_iter += 1

                    # Zero the gradients
                    self.optimizer.zero_grad()

                    torch.set_printoptions(threshold=5000)

                    # Get the loss
                    loss = self.task_model.forward(
                        *self.minimization_types[self.params['train_objective']](batch)) # Prepare each batch based on what type of training is used
                    loss_item = loss.item()
                    running_epoch_loss.append(loss_item)
                    if self.debug:
                        print(loss_item)
                    # running_batch_loss.append(loss_item)

                    # Get gradients
                    loss.backward()

                    # Grad clipping, same as in the paper we are comparing with
                    torch.nn.utils.clip_grad_norm_(
                        self.task_model.parameters(), 5)

                    # Take gradient step
                    self.optimizer.step()

                    # running_epoch_loss.append(loss_item)
            else:
                print(
                    f"No improvements for {epochs_since_last_improvement}. Training stopped. Report saved at: {self.params['save_path']}_report")
                if not os.path.exists(f'{self.params["save_path"]}_report'):
                    with open(f'{self.params["save_path"]}_report', 'w+') as report_f:
                        # write header
                        report_f.write(
                            f"Performance\tTime\tEval\tSeed\tTime\n")

                with open(f'{self.params["save_path"]}_report', 'a+') as report_f:
                    # write lines
                    report_f.write(
                        f"{performance}\t{format_number(time.time() - begin_time)}\t{self.params['evaluation']['evaluation_dev_file']}\t{self.params['random_seed']}\t{datetime.now()}\n")

                return

            if epochs_since_last_improvement < self.params['early_stopping']:
                # Prepare for evaluation
                # Zero the gradients
                self.optimizer.zero_grad()
                # Freeze the network for evaluation
                for param in self.task_model.mwe_f.parameters():
                    param.requires_grad = False

                self.task_model.eval()
                scores = []
                models = [
                    LogisticRegression(multi_class="multinomial", penalty='l2', C=0.5, solver="sag", n_jobs=20),
                    # LogisticRegression(multi_class="multinomial", penalty='l2', C=1,   solver="sag", n_jobs=20),
                    # LogisticRegression(multi_class="multinomial", penalty='l2', C=2,   solver="sag", n_jobs=20),
                    # LogisticRegression(multi_class="multinomial", penalty='l2', C=5,   solver="sag", n_jobs=20),
                    # LogisticRegression(multi_class="multinomial", penalty='l2', C=10,  solver="sag", n_jobs=20),
                    # LinearSVC(penalty='l2', dual=False, C=0.5,),
                    # LinearSVC(penalty='l2', dual=False, C=1,  ),
                    # LinearSVC(penalty='l2', dual=False, C=2,  ),
                    # LinearSVC(penalty='l2', dual=False, C=5,  ),
                    # LinearSVC(penalty='l2', dual=False, C=10, )
                    # 
                    # LogisticRegression(multi_class="multinomial", solver="sag", n_jobs=20)
                ]
                for model in models:
                    evaluation = Evaluation2(self.params['evaluation']['evaluation_train_file'],
                                            self.params['evaluation']['evaluation_dev_file'], 
                                            self.task_model.mwe_f, self.sg_embeddings, self.params['vocabulary_path'], device=self.device,
                                            te=model)
                    score = float(evaluation.evaluate()[-1])
                    scores.append(score)
                # print(score)
                # exit()
                score = np.max(scores)
                print(f'Max was with: {np.argmax(scores)}')
                # Unfreeze the network after evaluation
                for param in self.task_model.mwe_f.parameters():
                    param.requires_grad = True

                if performance is None:
                    performance = score
                    print(f"Save new best: {score}")
                    torch.save(self.task_model.mwe_f.state_dict(), f"{self.save_path}.pt")
                    if self.params['train_objective'] == 'JointTrainingSkipGramMinimization':
                        self.task_model.embedding_function.to_saved_file(f"{self.save_path}_embeddings.pt")
                elif performance < score:
                    epochs_since_last_improvement = 0
                    performance = score
                    print(f"Save new best: {score}")
                    torch.save(self.task_model.mwe_f.state_dict(), f"{self.save_path}.pt")
                    if self.params['train_objective'] == 'JointTrainingSkipGramMinimization':
                        self.task_model.embedding_function.to_saved_file(f"{self.save_path}_embeddings.pt")
                    # 
                else:
                    epochs_since_last_improvement += 1

                # self.scheduler.step(metrics=-score)
                print(
                    f"{epoch} - {format_number(score)}; {self.optimizer.param_groups[0]['lr']}. Current score {format_number(np.mean(running_epoch_loss))}. Took a total of {format_number(time.time() - begin_time)}s; {self.optimizer.param_groups[0]['lr']}. Model norm: {format_number(np.sum([torch.sum(torch.abs(x)) for x in self.task_model.mwe_f.parameters()]))}. Best is {format_number(performance)}. Epochs {epochs_since_last_improvement}")

                self.task_model.train()

        f.close()

    def save(self, path):
        torch.save(self.task_model.mwe_f.state_dict(), f"{path}")

    def prepare_direct_minimzation(self, batch):
        # Seeds used.
        # 2020-02-04 06:52:06,814 - INFO - allennlp.common.params - random_seed = 13370
        # 2020-02-04 06:52:06,814 - INFO - allennlp.common.params - numpy_seed = 1337
        # 2020-02-04 06:52:06,814 - INFO - allennlp.common.params - pytorch_seed = 133
        # 2020-02-04 06:52:06,851 - INFO - allennlp.common.checks - Pytorch version: 1.2.0

        # Calculate the embeddings
        multi_word_entities = [x.split(" ") for x in batch]
        # Sort by length
        multi_word_entities_indices = sorted(range(len(
            multi_word_entities)), key=lambda x: len(multi_word_entities[x]), reverse=True)
        mwe_len = torch.tensor([len(s) for s in multi_word_entities])[
            multi_word_entities_indices]

        # Words to indices using the vocabulary
        multi_word_entities = self.vocabulary.to_input_tensor(
            multi_word_entities, self.device)

        # Sort 
        mwe_vectorized = multi_word_entities[multi_word_entities_indices]

        # Get the learnt embeddings for each mwe
        learned_multi_word_entities = np.array(
            [x.replace(" ", "_") for x in batch])

        learned_multi_word_entities = self.vocabulary.to_input_tensor(
            learned_multi_word_entities, self.device)
        learned_multi_word_entities = learned_multi_word_entities[multi_word_entities_indices]
        if torch.any(learned_multi_word_entities == self.vocabulary.unk_id) or torch.any(mwe_vectorized == self.vocabulary.unk_id):
            print(
                "Some of the mwes (constituents + learned) in the current batch are unknown. Is everything alright?")
        return mwe_vectorized, mwe_len, learned_multi_word_entities

    def prepare_skipgram_minimization(self, batch):
        batch = np.array(batch, dtype=object)
        multi_word_entities = batch[:,1].tolist()
        multi_word_entities_indices = sorted(range(len(
            multi_word_entities)), key=lambda x: len(multi_word_entities[x]), reverse=True)
        mwe_len = torch.tensor([len(s) for s in multi_word_entities])[multi_word_entities_indices]
        multi_word_entities = self.vocabulary.to_input_tensor(
            multi_word_entities, self.device)
        mwe_vectorized = multi_word_entities[multi_word_entities_indices]

        words = batch[:, 0].tolist()
        # print(words)
        words_vectorized = self.vocabulary.to_input_tensor(words, self.device)[
            multi_word_entities_indices] # (batch_size, window_size*2)

        not_pads_idx = words_vectorized.reshape(-1) != 0
        
        negative_examples = self.generate_negative_examples(
            self.params['batch_size'], words_vectorized.reshape(-1)[not_pads_idx]) # (batch_size * context_size, number_of_negative_examples)


        return mwe_vectorized, mwe_len, words_vectorized, negative_examples

    def prepare_words_skipgram_minimization(self, batch):
        batch = np.array(batch, dtype=object)
        center_words = batch[:, 0].tolist() # batch_size, 
        outside_words = batch[:, 1].tolist()
        
        center_words_vectorized = self.vocabulary.to_input_tensor(center_words, self.device) # (batch_size * sentence_length, 1)
        outside_words_vectorized = self.vocabulary.to_input_tensor(outside_words, self.device) # (batch_size * sentence_length, 2*window_length)
        
        not_pads_idx = outside_words_vectorized.reshape(-1) != 0

        negative_examples = self.generate_negative_examples(
                    self.params['batch_size'], outside_words_vectorized.reshape(-1)[not_pads_idx]) # (batch_size * context_size, number_of_negative_examples); Sidenote: batch_size * context_size is the same as batch_size * 2 * window_length, but keeping only the non-pads

        # batch_size * sentence_length -> means that we are taking a number of <batch_size> sentences, where each sentence may have a different number of words
        # (batch_size * sentence_length, 1), (batch_size * sentence_length, 2*window_size), (batch_size * sentence_length  * 2 * window_length, negative_samples)        
        return center_words_vectorized, outside_words_vectorized, negative_examples

    def prepare_joint_training(self, batch):
        batch = np.array(batch, dtype=object)
        words = np.array(flatten(batch[:,0].tolist()))

        mwes = np.array(batch[:,1].tolist())

        mwe_params = list(self.prepare_skipgram_minimization(mwes.tolist()))
        words_params = list(self.prepare_words_skipgram_minimization(words.tolist()))

        return words_params, mwe_params

    def prepare_sentence_skipgram_minimization(self, batch):
        batch = np.array(batch)
        multi_word_entities = list(batch[:, 2])
        mwe_vectorized = self.vocabulary.to_input_tensor(multi_word_entities, self.device)
        left_part_vectorized  = self.vocabulary.to_input_tensor(list(batch[:, 0]), self.device)
        right_part_vectorized = self.vocabulary.to_input_tensor(list(batch[:, 3]), self.device)
        right_context = self.vocabulary.to_input_tensor(list(batch[:, 1]), self.device)
        rc_not_pad = right_context.reshape(-1) != 0
        left_context  = self.vocabulary.to_input_tensor(list(batch[:, 4]), self.device)
        lc_not_pad = left_context.reshape(-1) != 0

        negative_examples_left = self.generate_negative_examples(self.params['batch_size'], left_context.reshape(-1)[lc_not_pad]) # (batch_size * context, number_of_negative_samples)
        negative_examples_right = self.generate_negative_examples(self.params['batch_size'], right_context.reshape(-1)[rc_not_pad]) # (batch_size * context, number_of_negative_samples)
        return left_part_vectorized, right_part_vectorized, right_context, left_context, {'lpv_len': torch.tensor([len(x) for x in list(batch[:, 0])]), 
        'rpv_len': torch.tensor([len(x) for x in list(batch[:, 3])]), 
        'rc_len' : torch.tensor([len(x) for x in list(batch[:, 1])]), 
        'lc_len' : torch.tensor([len(x) for x in list(batch[:, 4])])}, negative_examples_left, negative_examples_right

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Entry point of the application.")

    parser.add_argument('--config-file', type=str, required=True,
                        help="Path to the file containig the config")
    parser.add_argument('--model-config-file', type=str, required=False,
                        help="Path to the file containig the config of the model")

    parser.add_argument("--learning-rate", type=float, required=False,
                        help="Learning rate to pass to the optimizer")
    parser.add_argument("--batch-size", type=int, required=False,
                        help="Number of examples to consider per batch")
    parser.add_argument("--num-epochs", type=int, required=False)
    parser.add_argument("--number-of-negative-examples",
                        type=int, required=False)
    parser.add_argument("--save-path", type=str, required=False)
    parser.add_argument("--hidden-size", type=int, required=False,
                        help="Hidden size for the LSTM model. The hidden size of the LSTM model can be different than of the word embeddings by using a projection matrix at the end. Good for increasing the capacity")
    parser.add_argument("--which-cuda", type=int, required=False,
                        help="Which cuda to use. Leave empty to use cpu")
    parser.add_argument("--weight-decay", type=float,
                        required=False, help="Weight decay for the optimizer")
    parser.add_argument("--model-type", type=str, required=False,
                        help="Type of the model. One of {'LSTMMultiply', 'LSTMMweModel', 'FullAdd', 'MultiplyMean', 'AddMultiply', 'CNNMweModel', 'LSTMMweModelPool'}")
    parser.add_argument("--num-models", type=int, required=False,
                        help="How many LSTMs to use. Applicable only with 'LSTMMweModelPool' model type")
    parser.add_argument("--early-stopping", type=int, required=False,
                        help="How many epochs of no improvements to wait until deciding to stop")
    parser.add_argument("--random-seed", type=int, required=False,
                        help="Random seed to use. Default: 1")
    parser.add_argument("--pretrained-model", type=str, required=False, default=None, help="Path to the pretrained model. Used to fine tune")

    result = parser.parse_args()
    config = json.load(open(result.config_file))
    model_config = json.load(open(config['model']))
    config = {**config, **model_config}
    result = vars(result)
    print(result)
    print(config)
    # Override based on cli arguments
    for update_param in ['learning_rate', 'batch_size', 'num_epochs', 'number_of_negative_examples', 'save_path', 'which_cuda', 'weight_decay', 'early_stopping', 'random_seed', 'pretrained_model']:
        if result[update_param] is not None:
            config[update_param] = result[update_param]

    if result['hidden_size'] is not None:
        config['model']['attributes']['hidden_size'] = result['hidden_size']

    if result['model_type'] is not None:
        config['model']['name'] = result['model_type']

    if result['num_models'] is not None:
        config['model']['attributes']['num_models'] = result['num_models']
    print(config)
    print(f"Init random seed with {config['random_seed']}")
    init_random(seed=config['random_seed'])
    if result['pretrained_model'] is not None:
        print(f"Loading pretrained model from {result['pretrained_model']}")
    train_obj = MWETrain(config)
    train_obj.train()
