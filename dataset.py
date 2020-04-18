from torch.utils import data
import torch
import pandas as pd
import numpy as np
import itertools


"""
File containing the necessary logic for loading the data
"""


class SkipGramMinimizationDataset(data.Dataset):

    def __init__(self, filepath, negative_examples_distribution, vocabulary, window_size=5):
        self.path = filepath
        self.negative_examples_distribution = negative_examples_distribution / negative_examples_distribution.sum()

        self.negative_examples_distribution_values = np.arange(len(self.negative_examples_distribution))
        self.vocabulary = vocabulary
        self.window_size = window_size
        with open(filepath) as f:
            count = sum(1 for _ in f)
            self.length = count
        self.dataset = pd.read_csv(self.path, sep='\n', header=None, quoting=3).values
        # self.cache_size = 20
        # self.cache = {}

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        sentence = self.dataset[index][0]
        words = sentence.split(' ')
        return words

    def generate_negative_examples(self, number_of_negative_examples, words_to_avoid):

        ## Commented below is the code for computing more negative examples and then caching them. It doesn't improve performance
        # tuple_words_to_avoid = tuple(words_to_avoid)
        # if tuple_words_to_avoid in self.cache and len(self.cache[tuple_words_to_avoid]) > 0:
        #     negative_examples = self.cache[tuple_words_to_avoid].pop(0)
        #     return negative_examples
        # else:
        #     negative_examples = np.random.choice(self.negative_examples_distribution_values, 
        #                         size=self.cache_size * number_of_negative_examples * words_to_avoid.shape[0], 
        #                         replace=True,
        #                         p=self.negative_examples_distribution).reshape(self.cache_size, -1, number_of_negative_examples)
        #     words_to_avoid = np.expand_dims(words_to_avoid, axis=1)
        #     words_to_avoid = np.expand_dims(words_to_avoid, axis=0)

        #     # Compare negative samples with the words that are indeed in the context. 
        #     # negative_examples: (cache_size, batch_size, number_of_negative_examples)
        #     # words_to_avoid: (1,batch_size,1)
        #     indices_to_change = (negative_examples == words_to_avoid)

        #     # indices_to_change will have the same shape as negative_examples: (cache_size, batch_size, number_of_negative_examples)
        #     # Replace such that no words_to_avoid appear in negative_examples (when is not allowed to, i.e. the context words itself shouldn't appear in his negative examples list)
        #     while np.any(indices_to_change):
        #         replacement = np.random.choice(self.negative_examples_distribution_values, np.nonzero(indices_to_change)[0].shape[0], replace=True, p=self.negative_examples_distribution)
        #         negative_examples[indices_to_change] = replacement
        #         indices_to_change = (negative_examples == words_to_avoid)

        #     self.cache[tuple_words_to_avoid] = list(negative_examples)
        #     negative_examples = self.cache[tuple_words_to_avoid].pop(0)
        #     return negative_examples

        negative_examples = np.random.choice(self.negative_examples_distribution_values, 
                                size=number_of_negative_examples * words_to_avoid.shape[0], 
                                replace=True,
                                p=self.negative_examples_distribution).reshape(-1, number_of_negative_examples)
        words_to_avoid = np.expand_dims(words_to_avoid, axis=1)
        # Compare negative samples with the words that are indeed in the context. 
        indices_to_change = (negative_examples == words_to_avoid)
        # indices_to_change will have the same shape as negative_examples: (batch_size, number_of_negative_examples)
        while np.any(indices_to_change):
            replacement = np.random.choice(self.negative_examples_distribution_values, np.nonzero(indices_to_change)[0].shape[0], replace=True, p=self.negative_examples_distribution)
            negative_examples[indices_to_change] = replacement
            indices_to_change = (negative_examples == words_to_avoid)
        
        return negative_examples


# TODO better add a new one, where you return full sentence or not (or lp, e, rp)
class WordWiseSGMweDataset(SkipGramMinimizationDataset):
    """
    Dataset implementation for feeding sentences with entities

    TODO Implement batch:
        - If the size is larger than threshold (say 1 000 000), then read 1 000 000 initially. Then, when the index to fetch gets larger than 1 000 000 read the next million and so on
        The only disadvantage to this 
    """
    def __init__(self, params):#filepath, window_size=5, full_sentence=False):
        super().__init__(params['train_file'], params['negative_examples_distribution'], params['vocabulary'], params['window_size'])
        self.number_of_negative_examples = params['number_of_negative_examples']

    def __getitem__(self, index: int):
        words = super().__getitem__(index)
        entity = list(filter(lambda x: '_' in x, words))[0]  # Should always exist. If it does not, then the method make_corpus from utils.py was not used
        index = words.index(entity)
        span = (index, index+1) # Because the mwe are merged with '_', making them, essentially, as a single word

        lc = words[max(0, span[0]-self.window_size):span[0]]
        rc = words[span[1]:span[1]+self.window_size]
        entity = entity.split('_')

        entity_vectorized = self.vocabulary.to_input_array(entity)
        # Add the necessary pad. Every context should be of length 2*window_size
        context = lc + rc
        context_vectorized = self.vocabulary.to_input_array(context + (2*self.window_size - len(context)) * [self.vocabulary.pad_token])
        negative_examples_vectorized = self.generate_negative_examples(self.number_of_negative_examples, context_vectorized[:len(context)]) # skip pads

        return [entity_vectorized, context_vectorized, negative_examples_vectorized, len(entity)] 


class SentenceWiseSGMweDataset(SkipGramMinimizationDataset):
    """
    This dataset corresponds with the following training procedure:
    Using as example the sentence: "John moved to New York after living for two years in Atlantic City"
    With the multi-word entity: "New York", and window_size = 2
    We want to predicct "after living" using "John moved to New York" and "moved to" using "after living for two years in Atlantic City"

    Different datasets are needed not only because different training regimes need different data, but also because
    in some cases (such as this one), we cannot pad in advance, because we don't know the max sentence length
    """
    def __init__(self, params):#filepath, window_size=5, full_sentence=False):
        super().__init__(params['train_file'], params['negative_examples_distribution'], params['vocabulary'], params['window_size'])
        self.number_of_negative_examples = params['number_of_negative_examples']

    def __getitem__(self, index: int):
        words = super().__getitem__(index)
        entity = list(filter(lambda x: '_' in x, words))[0]  # Should always exist. If it does not, then the method make_corpus from utils.py was not used
        index = words.index(entity)
        span = (index, index+1) # Because the mwe are merged with '_', making them, essentially, as a single word

        lc = words[max(0, span[0]-self.window_size):span[0]]
        rc = words[span[1]:span[1]+self.window_size]
        entity = entity.split('_')

        entity_vectorized = self.vocabulary.to_input_array(entity)

        left_sentence_vectorized = self.vocabulary.to_input_array(words[0:span[0]] + entity)
        right_context_vectorized = self.vocabulary.to_input_array(rc + (self.window_size - len(rc)) * [self.vocabulary.pad_token])
        right_sentence_vectorized = self.vocabulary.to_input_array(entity + words[span[1]:])
        left_context_vectorized = self.vocabulary.to_input_array(lc + (self.window_size - len(lc)) * [self.vocabulary.pad_token])


        # Negative for left sentence corresponds to the negatives of the right context. Same thing, different wordings.
        negative_for_left_sentence_vectorized = self.generate_negative_examples(self.number_of_negative_examples, right_context_vectorized.reshape(-1)[right_context_vectorized.reshape(-1)!=0])
        negative_for_right_sentence_vectorized = self.generate_negative_examples(self.number_of_negative_examples, left_context_vectorized.reshape(-1)[left_context_vectorized.reshape(-1)!=0])
        result = [left_sentence_vectorized, right_context_vectorized, negative_for_left_sentence_vectorized, len(left_sentence_vectorized),
                        entity_vectorized, len(entity),
                        right_sentence_vectorized, left_context_vectorized, negative_for_right_sentence_vectorized, len(right_sentence_vectorized)]
        return result 



class DirectMinimizationDataset(data.Dataset):
    """
    Dataset implementation for feeding the mwe
    """
    def __init__(self, params):
        super().__init__()
        self.path = params['train_file']
        with open(self.path) as f:
            count = sum(1 for _ in f)
            self.length = count
        self.dataset = pd.read_csv(self.path, sep='\n', header=None, quoting=3, names=['mwe'])['mwe'].str.replace('\t', ' ').values

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        word = self.dataset[index]
        return word


class TratzDataset(data.Dataset):
    """
    Dataset implementation for Tratz dataset ("Stephen Tratz  2011 Semantically-enriched Parsing for Natural Language Understanding")
    """
    def __init__(self, path):
        """
        :param path: to a file containing
        """
        super().__init__()
        dataset = pd.read_csv(path, header=None, quoting=3, sep='\t', names=['constituent1', 'constituent2', 'label'])
        dataset['mwe'] = dataset['constituent1'] + ' ' + dataset['constituent2']
        self.dataset = dataset.drop(columns=['constituent1', 'constituent2'])[['mwe', 'label']].values
        self.dataset = self.dataset[np.array([len(x[0].split(" ")) == 2 for x in self.dataset])]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]


class WordWiseSGDataset(SkipGramMinimizationDataset):
    """
    Dataset implementation for feeding words to a Skip-Gram like training
    Other datasets return 1d arrays because on every sentence there is only one token of interest that is center (the mwe), and 2*window_size context
    Here, every word can be center and context, so there will be <sentence_length> center words, and at most <sentence_length> * 2 * window_length context words
    """
    def __init__(self, params):
        super().__init__(params['train_file'], params['negative_examples_distribution'], params['vocabulary'], params['window_size'])
        self.number_of_negative_examples = params['number_of_negative_examples']

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        words = super().__getitem__(index)
        sentence = list(itertools.chain.from_iterable(map(lambda x: x.split('_') if '_' in x else [x], words)))
        center_words_vectorized = self.vocabulary.to_input_array([sentence[x] for x in range(len(sentence))])

        context_words = [sentence[max(0, x-self.window_size):x] + sentence[x+1:x+self.window_size+1] for x in range(len(sentence))]
        context_words = [cw + (2*self.window_size - len(cw)) * [self.vocabulary.pad_token] for cw in context_words]
        context_words_vectorized = self.vocabulary.to_input_array(context_words)

        non_pads_idx = context_words_vectorized != 0

        negative_examples_vectorized = self.generate_negative_examples(self.number_of_negative_examples, context_words_vectorized[non_pads_idx].reshape(-1)).reshape(-1, self.number_of_negative_examples)

        # sentence_length is the length of the sentence after transforming the mwe into multiple words
        # (sentence_length), (sentence_length, 2*window_size), (context_size, number_of_negative_examples) 
        # context_size = sentence_length * 2 * window_size - (window_size * (window_size+1)). In other words, it is the number of non_pads in the context_words_vectorized
        return [center_words_vectorized, context_words_vectorized, negative_examples_vectorized]
 

class JointTrainingSGMinimizationDataset(data.Dataset):
    """
    Dataset implementation for feeding the mwe
    """
    def __init__(self, params):
        super().__init__()
        with open(params['train_file']) as f:
            count = sum(1 for _ in f)
            self.length = count
        # self.number_of_negative_examples = params['number_of_negative_examples']
        self.words_sg = WordWiseSGDataset(params)
        self.mwe_sg = WordWiseSGMweDataset(params)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        return [self.words_sg[index], self.mwe_sg[index]] # to use the same collate_fn for the generator in train.py script
