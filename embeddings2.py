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
    def __init__(self, center_embs, context_embs):
        """
        :param embeddings_path: (str) path to file containing the embeddings of the vocabulary. They are expected to
        be in the same order. The static function generate_vocabulary_embeddings generates such a file
        :param vocabulary: (Vocabulary) the vocabulary object of the corpus
        """
        super(SkipGramEmbeddings, self).__init__()

        # self.embeddings_path = embeddings_path
        # self.vocabulary = vocabulary
        

        # from_embedding_file is not static for the moment because there are some difficulties on loading the model afterwards
        # center_subset, context_subset, embeddings_size = self.from_embedding_file()

        self.center_embeddings = nn.Embedding.from_pretrained(center_embs.float())
        self.context_embeddings = nn.Embedding.from_pretrained(context_embs.float())
        self.embedding_size = context_embs.shape[1]

    @staticmethod
    def from_saved_file(saved_path):
        return torch.load(saved_path)

    @staticmethod
    def from_embedding_file(embeddings_path, vocabulary):
        model = gensim.models.Word2Vec.load(embeddings_path)
        counts = np.array(vocabulary.counts)
        words = [w for w in vocabulary.element2id.keys()][counts[counts == 0].shape[0]:]  # skip over first k dummy words with 0 count (<pad>, <unk> etc)
        center_unk = model.wv.vectors[model.wv.vocab['unk'].index]
        context_unk = model.syn1neg[model.wv.vocab['unk'].index]

        def index_of(word):
            return model.wv.vocab[word].index if word in model else model.wv.vocab['unk'].index

        # words_center_emb = np.vstack([np.zeros(center_unk.shape), center_unk]+[model.wv.vectors[model.wv.vocab[w].index] if condition(w) else center_unk for w in words])
        words_center_emb = np.vstack([np.zeros(center_unk.shape), center_unk]+[model.wv.vectors[index_of(w)] if np.all(model.wv.vectors[index_of(w)])!=0 else center_unk for w in words])
        words_context_emb = np.vstack([np.zeros(context_unk.shape), context_unk]+[model.syn1neg[index_of(w)] if np.all(model.syn1neg[index_of(w)])!=0 else center_unk for w in words])
        """
        mean_words_in = np.average(np.vstack([model[w] for w in words if w in model]), axis=0)
        words_emb = np.vstack([np.zeros(mean_words_in.shape), mean_words_in]+[model[w] if w in model else mean_words_in for w in words])
        """
        center_subset = torch.tensor(words_center_emb) # pre_trained_embeddings_subset
        context_subset = torch.tensor(words_context_emb)
        # return center_subset, context_subset, center_subset.shape[1]
        sg_embeddings = SkipGramEmbeddings(center_subset, context_subset)
        sg_embeddings.center_embeddings = nn.Embedding.from_pretrained(center_subset.float())
        sg_embeddings.context_embeddings = nn.Embedding.from_pretrained(context_subset.float())
        sg_embeddings.embedding_size = center_subset.shape[1]
        return sg_embeddings

    # @staticmethod
    # def with_gensim_backing():
    #     model = gensim.models.Word2Vec.load(embeddings_path)

    #     return 0

    # @staticmethod
    # def using_gensim(embeddings_path):
    #     """
    #     Create an embedding object using the gensim model as backing embeddings. As opposed to from_embedding_file
    #     which uses a vocabulary
    #     :param embeddings_path:
    #     :return:
    #     """

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
    def __init__(self, vocabulary, embedding_size=300) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.embedding_size = 300
        self.context_embeddings = nn.Embedding(num_embeddings=len(vocabulary), embedding_dim=embedding_size)
        self.center_embeddings = nn.Embedding(num_embeddings=len(vocabulary), embedding_dim=embedding_size)
        # nn.init.xavier_normal_(self.outside_embs.weight)
        # nn.init.xavier_normal_(self.center_embs.weight)
        # self.outside_embs.weight.requires_grad_(False)
        # self.center_embs.weight.requires_grad_(False)

        