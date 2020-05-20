import torch
import torch.nn as nn
import numpy as np
import gensim
from mwe_function_model import LSTMMweModel, FullAdd, AddMultiply
from baseline import RandomLSTM, Average, Max
import pandas as pd
from typing import List


def cosine_similarity(a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


mwe_function = LSTMMweModel(300, 300)
mwe_function.load_state_dict(torch.load('model/11111_mwe_f.pt', map_location=torch.device('cpu')))
# gensim.models.keyedvectors.KeyedVectors.load
embeddings = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
# embeddings = torch.load('random_embeddings.pickle', map_location=torch.device('cpu'))
average_baseline = Average()
max_baseline = Max()
lstm_baseline = RandomLSTM(300, 300, 1)
# lstm_baseline.requires_grad_(False)

average_baseline_res = []
max_baseline_res = []
lstm_baseline_res = []
model_res = []

# test_data = pd.read_csv('data/evaluation_data/word2vec_test2', header=None, sep='\n', quoting=3).values.squeeze(axis=1)
# print(test_data.shape)
# for query in test_data:
#     splitted = query.split("_")
#     # if splitted[0] not in embeddings or splitted[1] not in embeddings or splitted[-1] not in embeddings:
#     #     print(query)
#     #     print(splitted)
#     #     exit()
#     # if len(splitted) == 1:
#     #     print(query)
#     #     exit()
#     word_emb = torch.tensor(embeddings[splitted]).unsqueeze(dim=0)
#     entity_emb = embeddings[query]
#     average_baseline_res.append(cosine_similarity(average_baseline.forward(word_emb).squeeze(dim=0).detach().numpy(), entity_emb))
#     max_baseline_res.append(cosine_similarity(max_baseline.forward(word_emb).squeeze(dim=0).detach().numpy(), entity_emb))
#     lstm_baseline_res.append(cosine_similarity(lstm_baseline.forward(word_emb, [2]).squeeze(dim=0).detach().numpy(), entity_emb))
#     model_res.append(cosine_similarity(mwe_function.forward(word_emb, [2]).squeeze(dim=0).detach().numpy(), entity_emb))
#     # print(splitted)
#     # print(sum_baseline_res)
#     # print(lstm_baseline_res)
#     # print(model_res)
#     # exit()
#
# print(np.mean(average_baseline_res))
# print(np.mean(max_baseline_res))
# print(np.mean(lstm_baseline_res))
# print(np.mean(model_res))
#
# print(mwe_function)
# GEN text
# mwe = [x for x in embeddings.index2word if len(x.split("_")) > 1 and all(t in embeddings for t in x.split("_")) and all(b.isalpha() for a in x.split("_") for b in a)]
# len(mwe)
# mwe_np = np.array(mwe)
# with open('data/evaluation_data/word2vec_test2', 'w+') as f:
#     for word in mwe_np[np.random.randint(len(mwe), size=100000)]:
#         f.write(f"{word}\n")
#



#mwe_function.forward(torch.tensor(embeddings['New', 'York']).unsqueeze(dim=0), torch.tensor([2])).squeeze(dim=0)
#embeddings.similar_by_vector(mwe_function.forward(torch.tensor(embeddings['New', 'York']).unsqueeze(dim=0), torch.tensor([2])).squeeze(dim=0).detach().numpy(), 5)
#embeddings.similar_by_vector(mwe_function.forward(torch.tensor(embeddings['social', 'status']).unsqueeze(dim=0), torch.tensor([2])).squeeze(dim=0).detach().numpy(), 5)

#embeddings.outside_embs(torch.tensor([v['social'], v['status']]))
#cosine_similarity(mwe_function.forward(torch.tensor(embeddings.outside_embs(torch.tensor([v['social'], v['status']]))).unsqueeze(dim=0), torch.tensor([2])).squeeze(dim=0).detach().numpy(), embeddings.outside_embs(torch.tensor([v['the']])).squeeze(dim=0).detach().numpy())
# from vocabulary import Vocabulary
# v = Vocabulary.load('vocab.json')

"""

from vocabulary import Vocabulary
v = Vocabulary.load('vocab.json')

with open('gensim_format', 'w+') as f:
    f.write(f"{len(v)} 300\n")
    for word in v.word2id.keys():
        emb = embeddings.outside_embs(torch.tensor([v[word]]))
        f.write(f"{word} ")
        for val in emb.squeeze(dim=0).detach().numpy():
            f.write(f"{val} ")
        f.write("\n")
        
e = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format('gensim_format')

"""

