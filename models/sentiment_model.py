import torch
import torch.nn as nn
from torch.autograd import Variable
from models.embed_layer import Embedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SentimentModel(nn.Module):
    def __init__(self,  model_params, int2word):
        super(SentimentModel, self).__init__()

        self.hidden_dim = model_params.get('lstm_dim', 256)
        self.drop_prob = model_params.get('drop_prob', 0.5)
        self.output_dim = model_params.get('output_dim', 1)

        self.embed_layer = Embedding(int2word)
        self.embed_dim = self.embed_layer.embed_dim
        self.vocab_dim = self.embed_layer.vocab_size

        self.bilstm = nn.LSTM(input_size=self.embed_dim,
                              hidden_size=self.hidden_dim,
                              bidirectional=True,
                              batch_first=True)

        self.fc = nn.Linear(self.hidden_dim*2, self.output_dim)
        self.dropout = nn.Dropout(0.3)

        self.sig = nn.Sigmoid()

    def forward(self, X, hidden):
        embed = self.embed_layer(X).float()
        # concat embed vector and feature vec then feed to bi-lstm input
        lstm_out, hidden = self.bilstm(embed, hidden)

        # take the last sequence output
        out = self.fc(self.dropout(lstm_out[:, -1]))
        # sigmoid function
        sig_out = self.sig(out).squeeze()

        return sig_out, hidden

    def init_hidden(self, batch_size):
        hidden = (Variable(torch.zeros(2, batch_size, self.hidden_dim).to(device)),
                  Variable(torch.zeros(2, batch_size, self.hidden_dim).to(device)))
        return hidden

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

