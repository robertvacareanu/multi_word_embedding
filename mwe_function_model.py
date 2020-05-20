from typing import List, Any

import torch
import torch.nn as nn


"""
File holding classes that will transform the multi-word entity representation (and, additionally, any other information 
-- context words, etc) into a single embedding
"""


class LSTMMweModel(nn.Module):
    """
    Generating an embedding for a multi-word entity by applying an LSTM over the embeddings of each constituent word.
    """

    def __init__(self, params):
        super().__init__()
        self.lstm = nn.LSTM(input_size=params['embedding_dim'], hidden_size=params['embedding_dim'], num_layers=params['num_layers'], bias=True,
                            bidirectional=False)

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, batch_mwe: torch.Tensor, mwe_lengths: torch.Tensor):
        """

        :param batch_mwe: (batch_size, max_entity_size, embedding_dim) - tensor containing the batch and the embedding of
        each word for a multi-word entity
        :param mwe_lengths: A list containing the length (number of words) of each multi-word entity in the batch
        :return: A tensor (batch_size, embedding_dim) representing the embedding of each multi-word entity (mwe),
        corresponding to the last_hidden state for each mwe (if bidirectional = True return (batch, 2*embedding_dim)
        """
        x = nn.utils.rnn.pack_padded_sequence(batch_mwe, mwe_lengths, batch_first=True)
        _, (last_hidden, _) = self.lstm(x)

        # (num_layers, 2 if bidirectional else 1, batch_size, hidden_size)
        last_hidden = last_hidden.view(self.lstm.num_layers, 2 if self.lstm.bidirectional else 1, batch_mwe.shape[0], self.lstm.hidden_size)
        if self.lstm.bidirectional:
            last_hidden = torch.cat([last_hidden[-1, 0, :, :], last_hidden[-1, 1, :, :]], dim=1)
        else:
            last_hidden = last_hidden[-1,0,:,:]
        return last_hidden


class LSTMMweModelManualBidirectional(nn.Module):
    """
    Generating an embedding for a multi-word entity by applying an LSTM over the embeddings of each constituent word.
    """

    def __init__(self, params):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=params['embedding_dim'], hidden_size=params['embedding_dim'], num_layers=params['num_layers'], bias=True,
                            bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=params['embedding_dim'], hidden_size=params['embedding_dim'], num_layers=params['num_layers'], bias=True,
                            bidirectional=False)

        # same hidden size, same number of layers
        self.hidden_size = self.lstm1.hidden_size
        self.num_layers = self.lstm1.num_layers

        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        for name, param in self.lstm2.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, batch_mwe: torch.Tensor, mwe_lengths: torch.Tensor, which_lstm=0):
        """

        :param batch_mwe: (batch_size, max_entity_size, embedding_dim) - tensor containing the batch and the embedding of
        each word for a multi-word entity
        :param mwe_lengths: A list containing the length (number of words) of each multi-word entity in the batch
        :return: A tensor (batch_size, embedding_dim) representing the embedding of each multi-word entity (mwe),
        corresponding to the last_hidden state for each mwe (if bidirectional = True return (batch, 2*embedding_dim)
        """
        x = nn.utils.rnn.pack_padded_sequence(batch_mwe, mwe_lengths, batch_first=True)
        if which_lstm == 1:
            _, (last_hidden, _) = self.lstm1(x)
        elif which_lstm == 2:
            _, (last_hidden, _) = self.lstm2(x)
        elif which_lstm == 0:
            _, (last_hidden1, _) = self.lstm1(x)
            _, (last_hidden2, _) = self.lstm2(x)
            last_hidden1 = last_hidden1.view(self.num_layers, 1, batch_mwe.shape[0], self.hidden_size)
            last_hidden2 = last_hidden2.view(self.num_layers, 1, batch_mwe.shape[0], self.hidden_size)
            last_hidden = (last_hidden1 + last_hidden2)/2

        # (num_layers, 2 if bidirectional else 1, batch_size, hidden_size)
        last_hidden = last_hidden.view(self.num_layers, 1, batch_mwe.shape[0], self.hidden_size)[-1,0,:,:]

        return last_hidden        


class LSTMMultiply(nn.Module):
    """
    Generating an embedding for a multi-word entity by applying an LSTM over the embeddings of each constituent word.
    """

    def __init__(self, params):
        super().__init__()
        self.lstm = nn.LSTM(input_size=params['embedding_dim'], hidden_size=params['hidden_size'], num_layers=params['num_layers'], bias=True,
                            bidirectional=params['bidirectional'])

        self.ll = nn.Linear(in_features=2 * params['hidden_size'] if params['bidirectional'] else params['hidden_size'], out_features=params['embedding_dim'], bias=True)
        
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.ll.weight)
        nn.init.zeros_(self.ll.bias)

    def forward(self, batch_mwe: torch.Tensor, mwe_lengths: torch.Tensor):
        """

        :param batch_mwe: (batch, max_entity_size, embedding_dim) - tensor containing the batch and the embedding of
        each word for a multi-word entity
        :param mwe_lengths: A list containing the length (number of words) of each multi-word entity in the batch
        :return: A tensor (batch, embedding_dim) representing the embedding of each multi-word entity (mwe),
        corresponding to the last_hidden state for each mwe (if bidirectional = True return (batch, 2*embedding_dim)
        """
        x = nn.utils.rnn.pack_padded_sequence(batch_mwe, mwe_lengths, batch_first=True)
        _, (last_hidden, _) = self.lstm(x)

        if self.lstm.bidirectional:
            return self.ll(torch.cat([last_hidden[0, :, :], last_hidden[1, :, :]], dim=1)) #for bidirectional
        else:
            return self.ll(last_hidden.squeeze(dim=0))


class LSTMMweModelPool(nn.Module):
    """
    Generating an embedding for a multi-word entity by applying an LSTM over the sequence
    formed by the embeddings of each constituent word.
    """

    def __init__(self, params):
        super().__init__()
        self.lstms = nn.ModuleList([LSTMMweModel(params) for i in range(params['num_models'])])

        self.pool = nn.MaxPool1d(kernel_size=params['num_models'])


    def forward(self, batch_mwe: torch.Tensor, mwe_lengths: torch.Tensor):
        """

        :param batch_mwe: (batch, max_entity_size, embedding_dim) - tensor containing the batch and the embedding of
        each word for a multi-word entity
        :param mwe_lengths: A list containing the length (number of words) of each multi-word entity in the batch
        :return: A tensor (batch, embedding_dim) representing the embedding of each multi-word entity (mwe),
        corresponding to the last_hidden state for each mwe (if bidirectional = True return (batch, 2*embedding_dim)
        """
        x = batch_mwe
        hidden_states = []
        for lstm in self.lstms:
            hidden_states.append(lstm(x, mwe_lengths).unsqueeze(dim=2))
        hidden_states=torch.cat(hidden_states, dim=2)

        return self.pool(hidden_states).squeeze(dim=2)


class FullAdd(nn.Module):

    def __init__(self, params) -> None:
        super().__init__()
        self.ll1 = nn.Linear(in_features=params['embedding_dim'], out_features=params['embedding_dim'], bias=True)
        self.ll2 = nn.Linear(in_features=params['embedding_dim'], out_features=params['embedding_dim'], bias=True)
        nn.init.xavier_normal_(self.ll1.weight)
        nn.init.xavier_normal_(self.ll2.weight)

    def forward(self, batch_mwe: torch.Tensor, mwe_lengths: torch.Tensor):
        return self.ll1(batch_mwe[:,0,:].squeeze(dim=1)) + self.ll2(batch_mwe[:,1,:].squeeze(dim=1))


class MultiplyMean(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        self.ll = nn.Linear(in_features=params['embedding_dim'], out_features=params['embedding_dim'], bias=True)
        nn.init.eye(self.ll.weight)
        nn.init.zeros_(self.ll.bias)

    def forward(self, batch_mwe: torch.Tensor, mwe_lengths: torch.Tensor):
        return self.ll(torch.mean(batch_mwe, dim=1))


class MultiplyAdd(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        self.ll = nn.Linear(in_features=params['embedding_dim'], out_features=params['embedding_dim'], bias=True)
        nn.init.xavier_normal_(self.ll.weight)
        # self.ll.weight=torch.eye(300).float()

    def forward(self, batch_mwe: torch.Tensor, mwe_lengths: torch.Tensor):
        return self.ll(torch.sum(batch_mwe, dim=1))


class MatrixMweModel(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        self.ll = nn.Linear(in_features=2*params['embedding_dim'], out_features=params['embedding_dim'], bias=True)
        nn.init.xavier_normal_(self.ll.weight)
        # self.ll.weight=torch.eye(300).float()

    def forward(self, batch_mwe: torch.Tensor, mwe_lengths: torch.Tensor):
        return self.ll(torch.cat(batch_mwe[:,0,:].squeeze(dim=1), batch_mwe[:,1,:].squeeze(dim=1)), dim=1)


class AddMweModel(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        self.alpha = nn.Variable(torch.rand(1), requires_grad=True)
        self.beta = nn.Variable(torch.rand(1), requires_grad=True)


    def forward(self, batch_mwe: torch.Tensor, mwe_lengths: torch.Tensor):
        return self.alpha * batch_mwe[:,0,:].squeeze(dim=1) + self.beta * batch_mwe[:,1,:].squeeze(dim=1)


class CNNMweModel(nn.Module):
    """
    Generating an embedding for a multi-word entity by applying a CNN over the embeddings of each constituent word.
    """

    def __init__(self, params):
        super().__init__()
        self.cnn = nn.Conv1d(in_channels=params['embedding_dim'], out_channels=params['embedding_dim'], kernel_size=3, stride=1)

    def forward(self, batch_mwe: torch.Tensor, mwe_lengths: List[int]):
        """

        :param batch_mwe: (batch, max_entity_size, embedding_dim) - tensor containing the batch and the embedding of
        each word for a multi-word entity
        :param mwe_lengths: A list containing the length (number of words) of each multi-word entity in the batch
        :return: A tensor (batch, embedding_dim) representing the embedding of each multi-word entity (mwe),
        corresponding to the tensor obtained by applying max-pooling over the cnn result
        """

        return torch.zeros(batch_mwe.shape)


class GRUMweModel(nn.Module):
    """
    Generating an embedding for a multi-word entity by applying a GRU over the embeddings of each constituent word.
    """

    def __init__(self, params):
        super().__init__()
        self.gru = nn.GRU(input_size=params['embedding_dim'], hidden_size=params['embedding_dim'], num_layers=params['num_layers'], bias=True,
                            bidirectional=params['bidirectional'])

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, batch_mwe: torch.Tensor, mwe_lengths: torch.Tensor):
        """

        :param batch_mwe: (batch, max_entity_size, embedding_dim) - tensor containing the batch and the embedding of
        each word for a multi-word entity
        :param mwe_lengths: A list containing the length (number of words) of each multi-word entity in the batch
        :return: A tensor (batch, embedding_dim) representing the embedding of each multi-word entity (mwe),
        corresponding to the last_hidden state for each mwe (if bidirectional = True return (batch, 2*embedding_dim)
        """
        x = nn.utils.rnn.pack_padded_sequence(batch_mwe, mwe_lengths, batch_first=True)
        _, last_hidden = self.gru(x)
        
        if self.gru.bidirectional:
            return torch.cat([last_hidden[0, :, :], last_hidden[1, :, :]], dim=1) #for bidirectional
        else:
            return last_hidden.squeeze(dim=0)


class AttentionWeightedModel(nn.Module):
    """
    Generating an embedding for a multi-word entity by applying a GRU over the embeddings of each constituent word.
    """

    def __init__(self, params):
        super().__init__()
        

    def forward(self, batch_mwe: torch.Tensor, mwe_lengths: torch.Tensor):
        pass


