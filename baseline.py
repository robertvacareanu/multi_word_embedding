from typing import List
import torch
import torch.nn as nn

class RandomLSTM(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.lstm = nn.LSTM(input_size=params['embedding_dim'], hidden_size=params['embedding_dim'], num_layers=params['num_layers'], bias=True,
                            bidirectional=False)

    def forward(self, batch_mwe: torch.Tensor, mwe_lengths: List[int]):
        x = nn.utils.rnn.pack_padded_sequence(batch_mwe, mwe_lengths, batch_first=True)
        _, (last_hidden, _) = self.lstm(x)

        # return torch.cat([last_hidden[0, :, :], last_hidden[1, :, :]], dim=1) #for bidirectional
        return last_hidden.squeeze(dim=0)


class Average(nn.Module):

    def __init__(self, params):
        super().__init__()


    def forward(self, batch_mwe: torch.Tensor, mwe_lengths: List[int]):
        """

        :param batch_mwe: (batch, max_entity_size, embedding_dim) - tensor containing the batch and the embedding of
        each word for a multi-word entity
        :return: A tensor (batch, embedding_dim) representing the embedding of each multi-word entity (mwe),
        corresponding to the last_hidden state for each mwe (if bidirectional = True return (batch, 2*embedding_dim)
        """

        return batch_mwe.mean(dim=1)


class Max(nn.Module):

    def __init__(self, params):
        super().__init__()


    def forward(self, batch_mwe: torch.Tensor, mwe_lengths: List[int]):
        """

        :param batch_mwe: (batch, max_entity_size, embedding_dim) - tensor containing the batch and the embedding of
        each word for a multi-word entity
        :return: A tensor (batch, embedding_dim) representing the embedding of each multi-word entity (mwe),
        corresponding to the last_hidden state for each mwe (if bidirectional = True return (batch, 2*embedding_dim)
        """

        return batch_mwe.max(dim=1)[0]

