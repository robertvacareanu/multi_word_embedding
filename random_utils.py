import pandas as pd
import numpy as np
from sklearn.metrics import f1_score


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
