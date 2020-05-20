import torch
import pandas as pd
import numpy as np
import re
import random
from nltk.tokenize import word_tokenize
from nltk import ngrams
import string
import copy
from multiprocessing import Pool
import json

def pad_sentence(sentences, pad_token):
    """
    Pad list of sentences to the longest sentence in the batch
    :param sentences: list of sentences, where each sentence is represented by a list of words (str)
    :param pad_token: padding token
    :return: list of sentences where sentences shorter than the max lengths are padded out with the pad_token
    """
    max_sentence = max([len(s) for s in sentences])
    sentences_padded = [ x +[pad_token] *(max_sentence -len(x)) for x in sentences]
    return sentences_padded


def read_corpus(file_path):
    """
    Read the corpus
    :param file_path: path to the corpus file. It is assumed that the file contains lines of text separated by
    newline (\n)
    :return: List[List[str]]: list of sentences (words)
    """
    data = []
    for line in open(file_path):
        sentence = filter(None, line.strip().split(' '))
        data.append(sentence)
    return data


"""
def read_wikipedia_corpus(path):
    for line in open(path):
        words = line.lower()
        words = word_tokenize(words)
        words = list(words)
        yield words
"""
def read_wikipedia_corpus(path):
    for line in open(path):
        yield line.replace('\n', '').split(' ') # was already tokenized
        # yield word_tokenize(line.lower())


def read_tratz(path):
    dataset = pd.read_csv(path, header=None, quoting=3, sep='\t', names=[
                          'constituent1', 'constituent2', 'label'])
    dataset['mwe'] = dataset['constituent1'] + ' ' + dataset['constituent2']
    dataset = dataset.drop(columns=['constituent1', 'constituent2'])[
        ['mwe', 'label']].values
    for label in dataset[:, 1]:
        yield label



def make_corpus(file_path, save_path, mwes_path, number_of_lines=10000):
    """
    Make a corpus file (from a tokenized corpus), containing only sentences with mwe (therefore, with _). Needed to work with the dataset (EntitySentenceDataset), which expects each sentence
    to contain a mwe (merged using '_'). This design choice was made due to how Shwartz is constructing her training data. Her script for constructing the data can be found at
    (https://github.com/vered1986/NC_embeddings/blob/8dec4e2f7918ab7606abf61b9d90e4f2786a9652/source/training/distributional/preprocessing/extract_ngrams_and_windows.py)
    :param file_path: path to the corpus file
    :param save_path: path to where to save the generated corpus file
    :param mwes_path: path to the file containing the mwes. We keep only the sentences with mwes that are in the file at mwes_path
    """
    added_lines = 0
    mwes = [x.replace('\t', '_').strip() for x in open(mwes_path, 'r')]
    with open(save_path, 'w+') as f_out:
        with open(file_path, 'r') as f_in:
            for line in f_in:
                if added_lines < number_of_lines:
                    if '_' in line: # add only the lines with a mwe in it
                        words = line.split(' ')
                        mwe_word = list(filter(lambda x: '_' in x, words))[0]
                        if mwe_word.strip() in mwes and len(words) > 1: # add only if it is in mwes list and only if the lines that also have some context in it.
                            f_out.write(line)
                            added_lines+=1
                else:
                    return


# def save(model, version, size):
#     vocabulary = AbstractVocabulary.load(f'/net/kate/storage/work/rvacarenu/research/multi_word_embedding/data/wikipedia/{version}/vocab/vocab_en_corpus_tokenized_bigrams_{size}.json')
#     center_context_emb = from_embedding_file(model, vocabulary)
#     tcce = torch.tensor(center_context_emb)
#     torch.save(tcce, f'/work/rvacarenu/research/multi_word_embedding/data/wikipedia/{version}/embeddings/embeddings_w2v_w2_e100_sg_{size}.pt')
#     print(tcce.shape)


def init_random(seed):
    """
    Init torch, torch.cuda and numpy with same seed. For reproducibility.
    :param seed: Random number generator seed
    """
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = False
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Seeds used by Shwartz
    # torch.manual_seed(133)
    # torch.cuda.manual_seed(133)
    # np.random.seed(1337)
    # random.seed(13370)


def cosine_similarity(a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def similar_words_by_vector(vocabulary, embeddings, vector, device, k=5):
    values = list(map(lambda word: (word, cosine_similarity(embeddings.center_embeddings(vocabulary.to_input_tensor([[word]], device)).squeeze(dim=0).squeeze(dim=0).cpu().numpy(), vector)), list(vocabulary.word2id.keys())[1:]))
    cos_vals = np.array([x[1] for x in values])
    words = np.array([x[0] for x in values])
    sorted_vals = np.argpartition(cos_vals, -k)[-k:]
    return list(zip(list(words[sorted_vals]), list(cos_vals[sorted_vals])))


def get_mwe_e(mwe, vocabulary, embeddings, words):
    words_vectorized = vocabulary.to_input_tensor([words], torch.device('cpu'))
    words_emb = embeddings.center_embeddings(words_vectorized)
    words_mwe = mwe(words_emb, torch.tensor([len(words)]))
    return words_mwe.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()


def format_number(number, decimals=5):
    return (f'%.{decimals}f' % number)


def construct_configs_for_comparison():
    data_sizes = [50000, 100000, 250000, 500000, 1000000]
    # minimization_types = {'DirectMinimization': 'direct_minimization', 'SkipGramMinimization': 'sg_minimization_onlytrain'}
    embeddings_paths = {'embeddings_ft_w2_e200_sg': 'ft_2_200'}
    config = {
        "batch_size": 10,
        "early_stopping": 20,
        "learning_rate": 0.01,
        # "embeddings_path": "/work/rvacarenu/NC_embeddings/output/distributional/word2vec_sg/win2/100d_t/wv.bin",
        # "load_embeddings"
        # "model":"./data/wikipedia/config/config_model_"
        "num_epochs": 500,
        "number_of_negative_examples": 10,
        # "train_file":
        # "train_objective":
        "random_seed": 1, 
        "save_embeddings": False,
        # "save_path":
        # "vocabulary_path": 
        # "window_size": 
        "which_cuda": 2,
        "weight_decay": 0,
        "evaluation" : {
            "evaluation_train_file": "./data/tratz/coarse_grained_lexical_full/train.tsv",
            "evaluation_dev_file": "./data/tratz/coarse_grained_lexical_full/val.tsv"
        }
    }
    config_model = {
        "model": {
            "name": "LSTMMweModel",
            "attributes": {
                "num_layers": 1,
            }
        }
    }
    with open('./data/wikipedia/config/all_configs', 'w+') as all_configs:

        for ep in embeddings_paths:
            epa = embeddings_paths[ep]
            mt = 'SkipGramMinimization'
            mta = 'sg_minimization_onlytrain'
            for ds in data_sizes:
                window_size, embedding_size = epa.split('_')[1], epa.split('_')[2]
                current_config_model = copy.deepcopy(config_model)
                current_config_model['model']['attributes']['embedding_dim'] = int(embedding_size)
                current_config_model['model']['attributes']['hidden_size'] = int(embedding_size)

                with open(f"./data/wikipedia/config/config_model_{mta}_{epa}_{ds}.json", "w+") as config_model_f:
                    json.dump(current_config_model, config_model_f, indent=4)

                current_config = copy.deepcopy(config)
                current_config['load_embeddings'] = f"./data/wikipedia/{mta}/embeddings/{ep}_{ds}.pt"
                current_config['model'] = f"./data/wikipedia/config/config_model_{mta}_{epa}_{ds}.json"
                current_config['train_file'] = f"./data/wikipedia/{mta}/en_corpus_tokenized_bigrams_{ds}"
                current_config['train_objective'] = mt
                current_config['save_path'] = f"./data/wikipedia/{mta}/model/b10/mwe_f_{epa}_{ds}"
                current_config['vocabulary_path'] = f"./data/wikipedia/{mta}/vocab/vocab_en_corpus_tokenized_bigrams_{ds}.json"
                current_config['window_size'] = int(window_size)
                print(f"./data/wikipedia/{mta}/model/b10/mwe_f_{epa}_{ds}")
                with open(f"./data/wikipedia/config/config_{mta}_{epa}_{ds}.json", "w+") as config_f:
                    json.dump(current_config, config_f, indent=4)
                all_configs.write(f"./data/wikipedia/config/config_{mta}_{epa}_{ds}.json\n")
            

            mt = 'DirectMinimization'
            mta = 'direct_minimization'
            ds = 'complete'
            epa = embeddings_paths[ep]
            window_size, embedding_size = epa.split('_')[1], epa.split('_')[2]
            current_config_model = copy.deepcopy(config_model)
            current_config_model['model']['attributes']['embedding_dim'] = int(embedding_size)
            current_config_model['model']['attributes']['hidden_size'] = int(embedding_size)

            with open(f"./data/wikipedia/config/config_model_{mta}_{epa}_{ds}.json", "w+") as config_model_f:
                json.dump(current_config_model, config_model_f, indent=4)

            current_config = copy.deepcopy(config)
            current_config['load_embeddings'] = f"./data/wikipedia/{mta}/embeddings/{ep}_{ds}.pt"
            current_config['model'] = f"./data/wikipedia/config/config_model_{mta}_{epa}_{ds}.json"
            current_config['train_file'] = f"./data/wikipedia/{mta}/nc_vocab.txt"
            current_config['train_objective'] = mt
            current_config['save_path'] = f"./data/wikipedia/{mta}/model/b10/mwe_f_{epa}_{ds}"
            current_config['vocabulary_path'] = f"./data/wikipedia/{mta}/vocab/vocab_en_corpus_tokenized_bigrams_{ds}.json"
            current_config['window_size'] = int(window_size)
            print(f"./data/wikipedia/{mta}/model/b10/mwe_f_{epa}_{ds}")
            with open(f"./data/wikipedia/config/config_{mta}_{epa}_{ds}.json", "w+") as config_f:
                json.dump(current_config, config_f, indent=4)
            all_configs.write(f"./data/wikipedia/config/config_{mta}_{epa}_{ds}.json\n")
            

# def save(model, version, size):
# 	vocabulary = AbstractVocabulary.load(f'/net/kate/storage/work/rvacarenu/research/multi_word_embedding/data/wikipedia/{version}/vocab/vocab_en_corpus_tokenized_bigrams_{size}.json')
# 	center_context_emb = from_embedding_file(model, vocabulary)
# 	tcce = torch.tensor(center_context_emb)
# 	torch.save(tcce, f'/work/rvacarenu/research/multi_word_embedding/data/wikipedia/{version}/embeddings/embeddings_ft_w2_e200_{size}.pt')
# 	print(tcce.shape)


def flatten(x):
    return [z for y in x for z in y]


def from_embedding_file(model, vocabulary):
	counts = np.array(vocabulary.counts)
	words = [w for w in vocabulary.element2id.keys()][counts[counts == 0].shape[0]:]  # skip over first k dummy words with 0 count (<pad>, <unk> etc). Should always be the first ones
	center_unk = model.wv.vectors[model.wv.vocab['<unk>'].index]
	context_unk = model.trainables.syn1neg[model.wv.vocab['<unk>'].index]
	def index_of(word):
		return model.wv.vocab[word].index if word in model.wv.vocab else model.wv.vocab['<unk>'].index
	# words_center_emb = np.vstack([np.zeros(center_unk.shape), center_unk]+[model.wv.vectors[model.wv.vocab[w].index] if condition(w) else center_unk for w in words])
	words_center_emb = np.vstack([np.zeros(center_unk.shape), center_unk]+[model.wv.vectors[index_of(w)] if np.all(model.wv.vectors[index_of(w)])!=0 else center_unk for w in words])
	words_context_emb = np.vstack([np.zeros(context_unk.shape), context_unk]+[model.trainables.syn1neg[index_of(w)] if np.all(model.trainables.syn1neg[index_of(w)]) != 0 else center_unk for w in words])
	"""
	mean_words_in = np.average(np.vstack([model[w] for w in words if w in model]), axis=0)
	words_emb = np.vstack([np.zeros(mean_words_in.shape), mean_words_in]+[model[w] if w in model else mean_words_in for w in words])
	"""
	center_context_emb = np.concatenate((np.expand_dims(words_center_emb, axis=0),np.expand_dims(words_context_emb, axis=0)), axis=0)
	return center_context_emb

# Better, just create 2 evaluation2 objects and use the return (returns prediciton, gold among other things)
def bootstrapping_test(evaluation1, evaluation2, total=10000):
    from sklearn.linear_model import LogisticRegression
    from mwe_function_model import LSTMMweModel
    import numpy as np
    from dataset import TratzDataset
    from vocabulary import make_label_vocab, AbstractVocabulary
    from embeddings import SkipGramEmbeddings
    from utils import read_tratz
    import pandas as pd
    import torch
    from evaluate import Evaluation2
    r1 = evaluation1.evaluate()
    r2 = evaluation2.evaluate()
    gold = r1[1]
    model1_score = 0
    model2_score = 0
    for _ in range(total):
        indices = np.random.choice(gold.shape[0], gold.shape[0])                                                                                                                                        
        cp1 = r1[0][indices]
        cp2 = r2[0][indices]
        s1 = np.count_nonzero(cp1==gold[indices])
        s2 = np.count_nonzero(cp2==gold[indices])
        if s1>s2:
            model1_score += 1
        else:
            model2_score += 1
    return (model1_score, model2_score)