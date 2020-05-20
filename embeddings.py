import torch
import torch.nn as nn
import gensim
import numpy as np

"""
Functionality handling the embeddings
"""


class SkipGramEmbeddings(nn.Module):
    """
    Skip-gram embeddings
    """
    def __init__(self, center_embeddings, context_embeddings):
        """
        :param embeddings_path: (str) path to file containing the embeddings of the vocabulary. They are expected to
        be in the same order. The static function generate_vocabulary_embeddings generates such a file
        :param vocabulary: (Vocabulary) the vocabulary object of the corpus
        """
        super(SkipGramEmbeddings, self).__init__()

        if center_embeddings.shape != context_embeddings.shape:
            raise ValueError("Size mismatch between center and context embeddings")

        # Doesn't require grad
        self.center_embeddings = nn.Embedding.from_pretrained(center_embeddings.float())
        self.context_embeddings = nn.Embedding.from_pretrained(context_embeddings.float())
        self.embedding_size = center_embeddings.shape[1]

    @staticmethod
    def from_saved_file(saved_path):
        center_context_embeddings = torch.load(saved_path)
        center_embeddings = center_context_embeddings[0]
        context_embeddings = center_context_embeddings[1]
        return SkipGramEmbeddings(center_embeddings, context_embeddings)

    def to_saved_file(self, save_path):
        center_context_embeddings = torch.cat([self.center_embeddings.unsqueeze(dim=0), self.context_embeddings.unsqueeze(dim=0)], dim=0)
        torch.save(center_context_embeddings, save_path)

    # The way model is created depends on the type of embeddings. e.g. for FastText: gensim.models.KeyedVectors.load
    @staticmethod
    def from_embedding_file(model, vocabulary):
        # model = gensim.models.Word2Vec.load(embeddings_path)    
        counts = np.array(vocabulary.counts)
        # print('state_power' in model)
        # print('state' in model)
        # print('power' in model)
        # print('state_power' in model.wv.vocab)
        # print('state' in model.wv.vocab)
        # print('power' in model.wv.vocab)
        # exit()
        words = [w for w in vocabulary.element2id.keys()][counts[counts == 0].shape[0]:]  # skip over first k dummy words with 0 count (<pad>, <unk> etc). Should always be the first ones
        center_unk = model.wv.vectors[model.wv.vocab['<unk>'].index]
        context_unk = model.trainables.syn1neg[model.wv.vocab['<unk>'].index]

        def index_of(word):
            return model.wv.vocab[word].index if word in model.wv.vocab else model.wv.vocab['<unk>'].index

        # words_center_emb = np.vstack([np.zeros(center_unk.shape), center_unk]+[model.wv.vectors[model.wv.vocab[w].index] if condition(w) else center_unk for w in words])
        words_center_emb = np.vstack([np.zeros(center_unk.shape), center_unk]+[model.wv.vectors[index_of(w)] if np.all(model.wv.vectors[index_of(w)])!=0 else center_unk for w in words])
        words_context_emb = np.vstack([np.zeros(context_unk.shape), context_unk]+[model.trainables.syn1neg[index_of(
            w)] if np.all(model.trainables.syn1neg[index_of(w)]) != 0 else center_unk for w in words])
    
        """
        mean_words_in = np.average(np.vstack([model[w] for w in words if w in model]), axis=0)
        words_emb = np.vstack([np.zeros(mean_words_in.shape), mean_words_in]+[model[w] if w in model else mean_words_in for w in words])
        """
        center_context_emb = np.concatenate((np.expand_dims(words_center_emb, axis=0),np.expand_dims(words_context_emb, axis=0)), axis=0)
        return center_context_emb


    def __call__(self, word_tensor: torch.tensor):
        """
        Get the embeddings of the word_tensor. Usually, it is composed only of a batch of sentences.
        :param word_tensor: (torch.tensor), usually with shape (batch_size, max_sentence_length).
        Does not enforce sorting
        :return: a tensor of shape (batch_size, max_sentence_length, embedding_size)
        """
        raise ValueError("This is not meant to use. Access the weights through center_embeddings or context_embeddings")

    def forward(self, *input):
        raise ValueError("This is not meant to use in a forward-backward fashion. The weights are from a pretrained"
                         "model and are frozen")


class RandomInitializedEmbeddings(nn.Module):
    def __init__(self, vocabulary, embedding_size=200) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.embedding_size = embedding_size
        self.context_embeddings = nn.Embedding(num_embeddings=len(vocabulary), embedding_dim=embedding_size)
        self.center_embeddings = nn.Embedding(num_embeddings=len(vocabulary), embedding_dim=embedding_size)
        nn.init.xavier_normal_(self.context_embeddings.weight)
        nn.init.xavier_normal_(self.center_embeddings.weight)
        # self.outside_embs.weight.requires_grad_(False)
        # self.center_embs.weight.requires_grad_(False)

    def to_saved_file(self, save_path):
        center_context_embeddings = torch.cat([self.center_embeddings.weight.unsqueeze(dim=0), self.context_embeddings.weight.unsqueeze(dim=0)], dim=0)
        torch.save(center_context_embeddings, save_path)