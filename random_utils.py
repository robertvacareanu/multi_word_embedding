import pandas as pd
import numpy as np
import gensim
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity


def predict_majority_head(train_path, test_path):
    train = pd.read_csv(train_path, sep='\t', header=None, quoting=3, names=['c1', 'c2', 'label'])
    test = pd.read_csv(test_path, sep='\t', header=None, quoting=3, names=['c1', 'c2', 'label'])
    t = train.groupby(by=['c2', 'label'])['label'].count().sort_values().groupby(level=0).tail(1)
    prediction_dict = dict(list(t.index.values))
    gold = test['label'].values
    labels = train['label'].values
    # scores = [f1_score(gold, [prediction_dict[x] if x in prediction_dict else np.random.choice(labels) for x in test['c2']], average='weighted') * 100 for z in range(1000)]
    scores = [f1_score(gold, [prediction_dict[x] if x in prediction_dict else 'x' for x in test['c2']], average='weighted') * 100 for z in range(10)]
    print(np.mean(scores))
    print(np.std(scores))

def load_glove_model(f):
    glove = {}
    with open(f) as fin:
        for line in fin:
            split_lines = line.split(" ")
            word = split_lines[0]
            word_embedding = np.array([float(value) for value in split_lines[1:]])
            glove[word] = word_embedding
    return glove

def get_most_similar(possibilities, possibilities_emb, query, model):
    query_emb = model[query] if query in model else model['unk']
    m = cosine_similarity(possibilities_emb, [query_emb]).argmax()
    return possibilities[m]


def predict_majority_head_similarity(train_path, test_path, model):
    train = pd.read_csv(train_path, sep='\t', header=None, quoting=3, names=['c1', 'c2', 'label'])
    test = pd.read_csv(test_path, sep='\t', header=None, quoting=3, names=['c1', 'c2', 'label'])
    t = train.groupby(by=['c2', 'label'])['label'].count().sort_values().groupby(level=0).tail(1)
    prediction_dict = dict(list(t.index.values))
    possibilities = list(prediction_dict.keys())
    possibilities_emb = np.vstack([model[x] if x in model else model['unk'] for x in possibilities])
    gold = test['label'].values
    labels = train['label'].values
    # scores = [f1_score(gold, [prediction_dict[x] if x in prediction_dict else np.random.choice(labels) for x in test['c2']], average='weighted') * 100 for z in range(1000)]
    scores = [f1_score(gold, [prediction_dict[x] if x in prediction_dict else get_most_similar(possibilities, possibilities_emb, x, model) for x in test['c2']], average='weighted') * 100 for z in range(10)]
    print(np.mean(scores))
    print(np.std(scores))
