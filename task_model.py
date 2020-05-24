import torch
import torch.nn as nn
import numpy as np

"""
Generate embeddings for a multi-word entity (mwe) by training a function f to map multiple embeddings to a single one.
The embeddings for each word (including the words that are part of a mwe) are given. (i.e. they are not learnt or 
fine-tuned).
"""

class MWESkipGramTaskModel(nn.Module):
    """
       //Training a function that should map a multi-word entity to a single embedding by a function f to the input. (maybe delete this line)
    The train procedure is similar with the skip-gram, namely, the resulting word embedding should have good
    capabilities of predicting its neighbors.
    """

    def __init__(self, embedding_function, mwe_f, embedding_device, device, additional_info):
        super().__init__()
        self.embedding_function = embedding_function
        self.mwe_f = mwe_f
        self.embedding_device = embedding_device
        self.device = device
        self.ls = nn.LogSigmoid()

    def forward(self, center_words, center_words_len, outside_words, negative_samples):
        """
        This function is according to the description of Mikolov in "Distributed Representations of Words and Phrases
         and their Compositionality" (@link https://arxiv.org/pdf/1310.4546.pdf) for negative sampling.
        :param center_words: (batch_size, max_entity_length) - represents the center words used for predicting the
        outside words
        :param center_words_len: (batch_size) length of the center_words
        :param outside_words: (batch_size, window_size * 2) - represents the context of each center word. It can be
        separated by splitting in half
        :param negative_samples: (batch_size * context_size, neg_sample_size) - represents negative samples for each word from outside_words
        :return: The loss, as described in the Mikolov paper

        """

        # There will be some pad for outside words, because not all the center words have ALL the words in between -window length,window length (2*window_length, ignoring the center)
        # In this way, we ignore the pads
        not_pads_idx = outside_words.reshape(-1) != 0

        center_words_embeddings = self.embedding_function.center_embeddings(center_words.to(self.embedding_device)).to(self.device)
        # (batch_size, mwe_hidden_size=embedding_size)
        mwe = self.mwe_f(center_words_embeddings, center_words_len)
        # Each computed mwe is used for for 2*window_length words (outside words). Duplicate them and reshape to a 2D tensor where each line represents the embedding of a word
        mwe = mwe.unsqueeze(dim=1).expand(-1, outside_words.shape[1], -1).reshape(-1, mwe.shape[1])[not_pads_idx] # (batch_size * context_size, embedding_size)

        # (batch_size, window_size * 2, embedding_size)
        outside_words_embeddings = self.embedding_function.context_embeddings(outside_words.to(self.embedding_device)).to(self.device)
        # (batch_size * context_size, embedding_size) -- batch_size * context_size here is not necessarily equal to batch_size * window_size * 2, because it may contain pads, which are removed here
        outside_words_embeddings = outside_words_embeddings.reshape(-1, outside_words_embeddings.shape[2])[not_pads_idx]
        # outside_words_embeddings = outside_words_embeddings/torch.sqrt(torch.sum(outside_words_embeddings*outside_words_embeddings, dim=1)).unsqueeze(dim=1)
        # (batch_size * context_size, neg_sample_size, embedding_size)
        negative_samples_embeddings = self.embedding_function.context_embeddings(negative_samples.to(self.embedding_device)).to(self.device)
        # negative_samples_embeddings = negative_samples_embeddings / torch.sqrt(torch.sum(negative_samples_embeddings*negative_samples_embeddings, dim=2)).unsqueeze(dim=2)

        # Formulas according to the paper from Mikolov
        positive_loss = -self.ls(torch.sum(outside_words_embeddings * mwe, dim=1))
        negative_loss = torch.sum(-self.ls(torch.bmm(-negative_samples_embeddings, mwe.unsqueeze(dim=2)).squeeze(dim=2)), dim=1)

        loss = positive_loss + negative_loss
        return loss.mean()# + 0.0002 * np.sum([torch.sum(torch.abs(x)) for x in self.mwe_f.parameters()])


class MWEMeanSquareErrorTaskModel(nn.Module):

    def __init__(self, embedding_function, mwe_f, embedding_device, device, additional_info):
        super().__init__()
        # nn.Embedding.from_pretrained()
        self.embedding_function = embedding_function
        self.mwe_f = mwe_f
        self.device = device
        self.embedding_device = embedding_device
        self.loss = nn.MSELoss()


    def forward(self, center_words, center_words_len, mwe_words):
        """
        The training procedure in this case is to minimize the difference between what is predicted (mwe embedding) and what should have been predicted (the learnt mwe embedding)

        """
        center_words_embeddings = self.embedding_function.center_embeddings(center_words.to(self.embedding_device)).to(self.device)
        # (batch_size, mwe_hidden_size=embedding_size)
        mwe = self.mwe_f(center_words_embeddings, center_words_len)
        mwe_words_embeddings = self.embedding_function.center_embeddings(mwe_words.to(self.embedding_device)).to(self.device)

        loss = self.loss(mwe, mwe_words_embeddings)
        return loss


class MWESentenceSkipGramTaskModel(nn.Module):
    """
    The train procedure is similar with the skip-gram, namely, the resulting word embedding should have good
    capabilities of predicting its neighbors. The mwe_f is usually an RNN, since it builds the vector by going over the whole left context to predict
    the right and vice-versa, but anything capable of mapping a a sequence of vectors to a single vector words 
    """

    def __init__(self, embedding_function, mwe_f, embedding_device, device, additional_info):
        super().__init__()
        self.embedding_function = embedding_function
        self.mwe_f = mwe_f
        self.device = device
        self.embedding_device = embedding_device
        self.flip_right_sentence = additional_info['flip_right_sentence']
        self.ls = nn.LogSigmoid()

    def forward(self, left_sentence_vectorized, right_sentence_vectorized, right_context, left_context, lens, negative_examples_left, negative_examples_right):
        """
        This function is according to the description of Mikolov in "Distributed Representations of Words and Phrases
         and their Compositionality" (@link https://arxiv.org/pdf/1310.4546.pdf) for negative sampling.
        :param center_words: (batch_size, max_entity_length) - represents the center words used for predicting the
        outside words
        :param left_part_vectorized: (batch_size, max_left_part_len) - the left part of the sentence, including the mwe
        :param right_part_vectorized: (batch_size, max_right_part_len) - the right part of the sentence, including the mwe
        :param right_context: (batch_size, window_size) - right context 
        :param left_context: (batch_size, window_size) - left context
        :param lens: dictionary that hold the lengths for left_sentence_vectorized and right_sentence_vectorized
        :param negative_examples_left:  (batch_size * context_size, neg_sample_size) - represents negative samples for each outside_word of left_context
        :param negative_examples_right: (batch_size * context_size, neg_sample_size) - represents negative samples for each outside_word of right_context
        :return: The loss, as described in the Mikolov paper

        """
        batch_size = left_sentence_vectorized.shape[0]

        rc_not_pad = right_context.reshape(-1) != 0
        lpv_lens_sorted, lpv_lens_idx = lens['lpv_len'].sort(descending=True)
        # (batch_size, window_size, embedding_size)
        left_part_embeddings = self.embedding_function.center_embeddings(left_sentence_vectorized.to(self.embedding_device)).to(self.device)
        # (batch_size, embedding_size)
        if self.flip_right_sentence:
            mwe_left = self.mwe_f.forward(left_part_embeddings[lpv_lens_idx], lpv_lens_sorted, which_lstm=1)
        else:
            mwe_left = self.mwe_f.forward(left_part_embeddings[lpv_lens_idx], lpv_lens_sorted)
        # (batch_size, embedding_size)
        mwe_left = torch.zeros_like(mwe_left).to(mwe_left.device).scatter_(0, lpv_lens_idx.unsqueeze(dim=1).expand(-1, mwe_left.shape[1]).to(mwe_left.device), mwe_left) # unsort
        mwe_left = mwe_left.unsqueeze(dim=1).expand(-1, right_context.shape[1], -1).reshape(-1, mwe_left.shape[1])[rc_not_pad]
        right_context_embeddings = self.embedding_function.context_embeddings(right_context.to(self.embedding_device)).to(self.device)
        right_context_embeddings = right_context_embeddings.reshape(-1, right_context_embeddings.shape[2])[rc_not_pad]


        lc_not_pad = left_context.reshape(-1) != 0
        rpv_lens_sorted, rpv_lens_idx = lens['rpv_len'].sort(descending=True)
        right_part_embeddings = self.embedding_function.center_embeddings(right_sentence_vectorized.to(self.embedding_device)).to(self.device)
        if self.flip_right_sentence:
            mwe_right = self.mwe_f(right_part_embeddings[rpv_lens_idx], rpv_lens_sorted, which_lstm=2)
        else:
            mwe_right = self.mwe_f(right_part_embeddings[rpv_lens_idx], rpv_lens_sorted)

        mwe_right = torch.zeros_like(mwe_right).to(mwe_right.device).scatter_(0, rpv_lens_idx.unsqueeze(dim=1).expand(-1, mwe_right.shape[1]).to(mwe_right.device), mwe_right) # unsort
        mwe_right = mwe_right.unsqueeze(dim=1).expand(-1, left_context.shape[1], -1).reshape(-1, mwe_right.shape[1])[lc_not_pad]
        # (batch_size, window_size, embedding_size)
        left_context_embeddings = self.embedding_function.context_embeddings(left_context.to(self.embedding_device)).to(self.device)
        left_context_embeddings = left_context_embeddings.reshape(-1, left_context_embeddings.shape[2])[lc_not_pad]


        
        # (batch_size, number_of_neg_examples, embedding_size)
        negative_samples_left_embeddings = self.embedding_function.context_embeddings(negative_examples_left.to(self.embedding_device)).to(self.device)
        negative_samples_right_embeddings = self.embedding_function.context_embeddings(negative_examples_right.to(self.embedding_device)).to(self.device)

        positive_loss_right = -self.ls(torch.sum(right_context_embeddings * mwe_left, dim=1))
        negative_loss_right = torch.sum(-self.ls(torch.bmm(-negative_samples_right_embeddings, mwe_left.unsqueeze(dim=2)).squeeze(dim=2)), dim=1)

        positive_loss_left = -self.ls(torch.sum(left_context_embeddings * mwe_right, dim=1))
        negative_loss_left = torch.sum(-self.ls(torch.bmm(-negative_samples_left_embeddings, mwe_right.unsqueeze(dim=2)).squeeze(dim=2)), dim=1)


        loss = (positive_loss_right + negative_loss_right).mean() + (positive_loss_left + negative_loss_left).mean()
        return loss

# TODO Abstract
class MWEWordLevelSkipGramTaskModel(nn.Module):
    """
    Task model for word-level skip-gram
    """

    def __init__(self, embedding_function, mwe_f, embedding_device, device, additional_info):
        super().__init__()
        # nn.Embedding.from_pretrained()
        self.embedding_function = embedding_function
        self.mwe_f = mwe_f
        self.ls = nn.LogSigmoid()
        self.mwe_task_model = MWESkipGramTaskModel(embedding_function, mwe_f)
        self.device = device
        self.embedding_device = embedding_device

    def forward(self, params_words, params_mwes):
        """
        This function is according to the description of Mikolov in "Distributed Representations of Words and Phrases
         and their Compositionality" (@link https://arxiv.org/pdf/1310.4546.pdf) for negative sampling.
        :param center_words: (batch_size, max_entity_length) - represents the center words used for predicting the
        outside words
        :param center_words_len: (batch_size) length of the center_words
        :param outside_words: (batch_size, window_size * 2) - represents the context of each center word. It can be
        separated by splitting in half
        :param negative_samples: (batch_size * context_size, neg_sample_size) - represents negative samples for each word from outside_words
        :return: The loss, as described in the Mikolov paper

        """
        center_words, outside_words, negative_examples_words = params_words
        mwe_words, mwe_length, outside_mwe_words, negative_examples_mwe = params_mwes

        # (batch_size * words_in_sentence, embedding_size)
        center_words_embeddings = self.embedding_function.center_embeddings(center_words.unsqueeze(dim=1).to(self.embedding_device)).to(self.device)

        # (batch_size * sentence_length, window_size * 2, embedding_size)
        outside_words_embeddings = self.embedding_function.context_embeddings(outside_words.to(self.embedding_device)).to(self.device)

        # (batch_size * context_size, neg_sample_size, embedding_size)
        negative_samples_embeddings = self.embedding_function.context_embeddings(negative_examples_words.to(self.embedding_device)).to(self.device)
        # negative_samples_embeddings = negative_samples_embeddings / torch.sqrt(torch.sum(negative_samples_embeddings*negative_samples_embeddings, dim=2)).unsqueeze(dim=2)

        # Formulas according to the paper from Mikolov
        positive_loss = -self.ls(torch.sum(outside_words_embeddings * center_words_embeddings, dim=1))
        negative_loss = torch.sum(-self.ls(torch.bmm(-negative_samples_embeddings, center_words_embeddings.unsqueeze(dim=2)).squeeze(dim=2)), dim=1)

        loss_words = positive_loss + negative_loss
        loss_words = loss_words.mean()
        loss_mwes = self.mwe_task_model.forward(mwe_words, mwe_length, outside_mwe_words, negative_examples_mwe)
        
        loss = loss_words + 25 * loss_mwes

        return loss.mean()


class MWEJointTraining(nn.Module):
    """
    Jointly training word embeddings with a function that learns to map multi words to a single embedding
    The train procedure is similar with the skip-gram, namely, the resulting word embedding should have good
    capabilities of predicting its neighbors. 
    """

    def __init__(self, embedding_function, mwe_f, embedding_device, device, additional_info):
        super().__init__()
        # nn.Embedding.from_pretrained()
        self.embedding_function = embedding_function
        self.mwe_f = mwe_f
        self.ls = nn.LogSigmoid()
        self.mwe_task_model = MWESkipGramTaskModel(embedding_function, mwe_f, embedding_device, device, additional_info)
        self.device = device
        self.embedding_device = embedding_device

    def forward(self, params_words, params_mwes):
        # (batch_size * sentence_length), (batch_size * sentence_length, 2*window_size), (context_size, number_of_negative_examples)
        center_words, outside_words, negative_examples_words = params_words
        mwe_words, mwe_length, outside_mwe_words, negative_examples_mwe = params_mwes

        not_pads_idx = outside_words.reshape(-1) != 0

        center_words_embeddings = self.embedding_function.center_embeddings(center_words.to(self.embedding_device)).to(self.device)
        # (batch_size, embedding_size)
        center_words_embeddings = center_words_embeddings.unsqueeze(dim=1).expand(-1, outside_words.shape[1], -1).reshape(-1, center_words_embeddings.shape[1])[not_pads_idx] # (batch_size * context_size, embedding_size)

        # (batch_size, window_size * 2, embedding_size)
        outside_words_embeddings = self.embedding_function.context_embeddings(outside_words.to(self.embedding_device)).to(self.device)
        # (batch_size * context_size, embedding_size) -- batch_size * context_size here is not necessarily equal to batch_size * window_size * 2, because it may contain pads, which are removed here
        outside_words_embeddings = outside_words_embeddings.reshape(-1, outside_words_embeddings.shape[2])[not_pads_idx]
        # outside_words_embeddings = outside_words_embeddings/torch.sqrt(torch.sum(outside_words_embeddings*outside_words_embeddings, dim=1)).unsqueeze(dim=1)
        # (batch_size * context_size, neg_sample_size, embedding_size)
        negative_samples_embeddings = self.embedding_function.context_embeddings(negative_examples_words.to(self.embedding_device)).to(self.device)
        # negative_samples_embeddings = negative_samples_embeddings / torch.sqrt(torch.sum(negative_samples_embeddings*negative_samples_embeddings, dim=2)).unsqueeze(dim=2)

        # Formulas according to the paper from Mikolov
        positive_loss = -self.ls(torch.sum(outside_words_embeddings * center_words_embeddings, dim=1))
        negative_loss = torch.sum(-self.ls(torch.bmm(-negative_samples_embeddings, center_words_embeddings.unsqueeze(dim=2)).squeeze(dim=2)), dim=1)

        loss_words = positive_loss + negative_loss
        # loss_words = loss_words.mean()
        
        loss_mwes = self.mwe_task_model.forward(mwe_words, mwe_length, outside_mwe_words, negative_examples_mwe)
        
        loss = loss_words + loss_mwes

        return loss.mean()


class AutoEncoderPreTraining(nn.Module):
    """
    Train an auto encoder. Used as a pretraining procedure
    :param embedding_function generate embeddings of the input words
    :param mwe_f the function to apply this pretraining to
    """
    def __init__(self, embedding_function, mwe_f, embedding_device, device):
        super().__init__()
        # nn.Embedding.from_pretrained()
        self.embedding_function = embedding_function
        self.encoder = mwe_f
        self.decoder = nn.Linear(in_features=self.embedding_function.embedding_dim, out_features=self.embedding_function.embedding_dim, bias=True)
        self.loss = nn.MSELoss()
        self.device = device
        self.embedding_device = embedding_device

    def forward(self, input):
        """
        :param input (batch_size, max_seq_length)
        """
        # (batch_size * max_seq_length)
        non_pads = input.reshape(-1) != 0

        # (batch_size, max_seq_length, embedding_dim)
        input_vectorized = self.embedding_function(input.to(self.embedding_device)).to(self.device)

        # (batch_size, max_seq_length, embedding_dim) -- added noise with mean 0.1. The average std for the embeddings is 0.5
        encoded_input = self.encoder(input_vectorized + torch.rand(input_vectorized.shape) / 10)

        # (batch_size * max_seq_length, embedding_dim)
        decoded_input = self.decoder(input_vectorized.reshape(-1, self.embedding_function.embedding_dim)[non_pads])

        loss = self.loss(input_vectorized.reshape(-1, self.embedding_function.embedding_dim)[non_pads], decoded_input)
    
        return loss

