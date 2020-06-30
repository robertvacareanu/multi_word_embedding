import pandas as pd
import numpy as np
import gensim
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity


def predict_majority_head(train_path, test_path):
    train = pd.read_csv(train_path, sep='\t', header=None, quoting=3, names=['c1', 'c2', 'label'])
    l,c = np.unique(train['label'].values, return_counts=True)
    label = l[c.argmax()]

    test = pd.read_csv(test_path, sep='\t', header=None, quoting=3, names=['c1', 'c2', 'label'])
    t = train.groupby(by=['c2', 'label'])['label'].count().sort_values().groupby(level=0).tail(1)
    t2= train.groupby(by=['c1', 'label'])['label'].count().sort_values().groupby(level=0).tail(1)
    prediction_dict = dict(list(t.index.values))
    prediction2_dict= dict(list(t2.index.values))
    gold = test['label'].values
    labels = train['label'].values
    def get_prediction(x):
        if x in prediction_dict:
            return prediction_dict[x]
        elif x in prediction2_dict:
            return prediction2_dict[x]
        else:
            return label
    # scores = [f1_score(gold, [prediction_dict[x] if x in prediction_dict else np.random.choice(labels) for x in test['c2']], average='weighted') * 100 for z in range(1000)]
    scores = [f1_score(gold, [prediction_dict[x] if x in prediction_dict else label for x in test['c2']], average='weighted') * 100 for z in range(10)]
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

# embeddings_path = "/data/nlp/corpora/glove/glove.840B.300d.txt"
# model = load_glove_model(embeddings_path)

# model = gensim.models.KeyedVectors.load("/work/rvacarenu/code/mwe/NC_embeddings/output/distributional/fasttext_sg/200d_oc/win2/wv.bin")

# print('<unk>' in glove)
# print('<UNK>' in glove)
# print('unk' in glove)
# print('UNK' in glove)
# 
predict_majority_head('/work/rvacarenu/code/mwe/NC_embeddings/source/evaluation/classification/data/tratz_fine_grained_random/train.tsv', '/work/rvacarenu/code/mwe/NC_embeddings/source/evaluation/classification/data/tratz_fine_grained_random/test.tsv')
predict_majority_head('/work/rvacarenu/code/mwe/NC_embeddings/source/evaluation/classification/data/tratz_coarse_grained_random/train.tsv', '/work/rvacarenu/code/mwe/NC_embeddings/source/evaluation/classification/data/tratz_coarse_grained_random/test.tsv')
predict_majority_head('/work/rvacarenu/code/mwe/NC_embeddings/source/evaluation/classification/data/tratz_fine_grained_lexical/train.tsv', '/work/rvacarenu/code/mwe/NC_embeddings/source/evaluation/classification/data/tratz_fine_grained_lexical/test.tsv')
predict_majority_head('/work/rvacarenu/code/mwe/NC_embeddings/source/evaluation/classification/data/tratz_coarse_grained_lexical/train.tsv', '/work/rvacarenu/code/mwe/NC_embeddings/source/evaluation/classification/data/tratz_coarse_grained_lexical/test.tsv')

# predict_majority_head_similarity('/work/rvacarenu/code/mwe/NC_embeddings/source/evaluation/classification/data/tratz_fine_grained_random/train.tsv', '/work/rvacarenu/code/mwe/NC_embeddings/source/evaluation/classification/data/tratz_fine_grained_random/test.tsv', model)
# predict_majority_head_similarity('/work/rvacarenu/code/mwe/NC_embeddings/source/evaluation/classification/data/tratz_coarse_grained_random/train.tsv', '/work/rvacarenu/code/mwe/NC_embeddings/source/evaluation/classification/data/tratz_coarse_grained_random/test.tsv', model)
# predict_majority_head_similarity('/work/rvacarenu/code/mwe/NC_embeddings/source/evaluation/classification/data/tratz_fine_grained_lexical/train.tsv', '/work/rvacarenu/code/mwe/NC_embeddings/source/evaluation/classification/data/tratz_fine_grained_lexical/test.tsv', model)
# predict_majority_head_similarity('/work/rvacarenu/code/mwe/NC_embeddings/source/evaluation/classification/data/tratz_coarse_grained_lexical/train.tsv', '/work/rvacarenu/code/mwe/NC_embeddings/source/evaluation/classification/data/tratz_coarse_grained_lexical/test.tsv', model)


# BASELINES

    # Most similar fastext
        # FINE-GRAINED RANDOM
        # 54.721640464738 +- 0.0
        # COARSE-GRAINED RANDOM
        # 60.83826337358649 +- 0.0
        # FINE-GRAINED LEXICAL
        # 0.0 +- 0.0
        # COARSE-GRAINED LEXICAL
        # 0.0 +- 0.0

    # Most similar glove
        # FINE-GRAINED RANDOM
        # 54.721640464738 +- 0.0
        # COARSE-GRAINED RANDOM
        # 60.85150496612558 +- 0.0
        # FINE-GRAINED LEXICAL
        # 0.0 +- 0.0
        # COARSE-GRAINED LEXICAL
        # 0.0 +- 0.0

    # Predict majority
        # FINE-GRAINED RANDOM
        # 53.07881373801998 +- 0.0
        # COARSE-GRAINED RANDOM
        # 58.96859684443011 +- 0.0
        # FINE-GRAINED LEXICAL
        # 5.464312778916051 +- 0.0
        # COARSE-GRAINED LEXICAL
        # 6.448511627650726 +- 0.0

    # Predict random
        # FINE-GRAINED RANDOM
        # 52.92612918241657 +- 0.11262168719515299
        # COARSE-GRAINED RANDOM
        # 59.063195492617034 +- 0.15276441435068538
        # FINE-GRAINED LEXICAL
        # 6.498085573506013 +- 0.8561751988820646
        # COARSE-GRAINED LEXICAL
        # 11.675270976937975 +- 1.1011438526014723

    # Predict non-existent
        # FINE-GRAINED RANDOM
        # 54.72164046473843 +- 0.0
        # COARSE-GRAINED RANDOM
        # 60.86040763533979 +- 0.0
        # FINE-GRAINED LEXICAL
        # 0.0 +- 0.0
        # COARSE-GRAINED LEXICAL
        # 0.0 +- 0.0
