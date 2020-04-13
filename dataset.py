from torch.utils import data
import pandas as pd
import numpy as np
import itertools

"""
File containing the necessary logic for loading the data
"""

class EntitySentenceDataset(data.Dataset):
    """
    Dataset implementation for feeding sentences with entities

    TODO Implement batch:
        - If the size is larger than threshold (say 1 000 000), then read 1 000 000 initially. Then, when the index to fetch gets larger than 1 000 000 read the next million and so on
        The only disadvantage to this 
    """
    def __init__(self, filepath, window_size=5, full_sentence=False):
        super().__init__()
        self.path = filepath
        self.window_size = window_size
        self.full_sentence = full_sentence
        with open(filepath) as f:
            count = sum(1 for _ in f)
            self.length = count
        self.dataset = pd.read_csv(self.path, sep='\n', header=None, quoting=3).values

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        sentence = self.dataset[index][0]
        words = sentence.split(' ')
        words = list(map(lambda w: w.replace('\n', ''), words))
        entity = list(filter(lambda x: '_' in x, words))[0]  # Should always exist. If it does not, then the method make_corpus from utils.py was not used
        index = words.index(entity)
        span = (index, index+1) # Because the mwe are merged with '_', making them, essentially, as a single word
        lc = words[max(0, span[0]-self.window_size):span[0]]
        rc = words[span[1]:span[1]+self.window_size]
        entity = entity.split('_')
        if self.full_sentence: 
            # context = lc + rc
            # entity = ' '.join(entity)
            # left_sentence, right_context, entity, right_sentence, left_context
            result = [words[0:span[0]] + entity, rc, entity, entity + words[span[1]:], lc]
            # result = [lc + entity, rc, entity, entity + rc, lc]
        else: 
            context = lc + rc

            # Avoid the cases where it has the same size and it concatenates instead of using object
            result = np.zeros(2, dtype=object)
            result[0] = context
            result[1] = entity
            result = [context, entity]
            # entity = ' '.join(entity.split('_')) # from mwe merged with '_' to mwe merged with ' '
            # result = [np.array(list(zip([entity] * len(context), context)))]
            # result = np.concatenate(result, axis=0) # 2d nd array, where on the first column is the center word and on the second column are the context words
        
        return result 


class EntitySentenceDirectMinimizationDataset(data.Dataset):
    """
    Dataset implementation for feeding the mwe
    """
    def __init__(self, filepath, window_size=5):
        super().__init__()
        self.path = filepath
        with open(filepath) as f:
            count = sum(1 for _ in f)
            self.length = count
        self.dataset = pd.read_csv(self.path, sep='\n', header=None, quoting=3, names=['mwe'])['mwe'].str.replace('\t', ' ').values

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        word = self.dataset[index]
        return np.array([word]) # to use the same collate_fn for the generator in train.py script


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


class WordSGMinimizationDataset(data.Dataset):
    """
    Dataset implementation for feeding the mwe
    """
    def __init__(self, filepath, window_size=5):
        super().__init__()
        self.path = filepath
        self.window_size = window_size
        with open(filepath) as f:
            count = sum(1 for _ in f)
            self.length = count
        self.dataset = pd.read_csv(self.path, sep='\n', header=None, quoting=3).values

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        sentence = self.dataset[index][0].split(' ')
        sentence = list(itertools.chain.from_iterable(map(lambda x: x.split('_') if '_' in x else [x], sentence)))

        splitted = [[[sentence[x]], sentence[max(0, x-self.window_size):x] + sentence[x+1:x+self.window_size+1]] for x in range(len(sentence))]
        return splitted # to use the same collate_fn for the generator in train.py script
 


class JointTrainingSGMinimizationDataset(data.Dataset):
    """
    Dataset implementation for feeding the mwe
    """
    def __init__(self, filepath, window_size=5):
        super().__init__()
        self.path = filepath
        self.window_size = window_size
        with open(filepath) as f:
            count = sum(1 for _ in f)
            self.length = count
        self.dataset = pd.read_csv(self.path, sep='\n', header=None, quoting=3).values

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        sentence = self.dataset[index][0]
        words = sentence.split(' ')
        words = list(map(lambda w: w.replace('\n', ''), words))
        entity = list(filter(lambda x: '_' in x, words))[0]  # Should always exist. If it does not, then the method make_corpus from utils.py was not used
        index = words.index(entity)
        span = (index, index+1) # Because the mwe are merged with '_', making them, essentially, as a single word
        lc = words[max(0, span[0]-self.window_size):span[0]]
        rc = words[span[1]:span[1]+self.window_size]
        entity = entity.split('_')

        sentence = list(itertools.chain.from_iterable(map(lambda x: x.split('_') if '_' in x else [x], words)))
        splitted = [[[words[x]], words[max(0, x-self.window_size):x] + words[x+1:x+self.window_size+1]] for x in range(len(words))]

        context = lc + rc
        mwe = [context, entity]

        return splitted, mwe # to use the same collate_fn for the generator in train.py script
