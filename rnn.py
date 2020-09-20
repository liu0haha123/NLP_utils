import torch
import torch.nn as nn
import torch.nn.functional as F


class UnfoldingLSTM(nn.Module):
    def __init__(self):
        super(UnfoldingLSTM, self).__init__()

        # Number of samples per time step
        self.batch_size = 2

        # Dimension of weight vectors
        self.hidden_dim = 16

        # Dimension of embedded tensor
        self.embedding_dim = 2

        # The vocabulary size
        self.input_size = 4

        # Number of time steps
        self.sequence_len = 2

        # Initialize embedding layer
        self.embedding = nn.Embedding(self.input_size, self.embedding_dim, padding_idx=0)

        # Initialize LSTM Cell
        self.lstm_cell = nn.LSTMCell(self.embedding_dim, self.hidden_dim)

    def forward(self, x):
        # Creation of cell state and hidden state
        hidden_state = torch.zeros(x.size(0), self.hidden_dim)
        cell_state = torch.zeros(x.size(0), self.hidden_dim)

        # Weights initialization
        torch.nn.init.xavier_normal_(hidden_state)
        torch.nn.init.xavier_normal_(cell_state)

        # From idx-token to embedded tensors
        out = self.embedding(x)

        # Prepare the shape for LSTMCell
        out = out.view(self.sequence_len, x.size(0), -1)
        final_out = torch.zeros((self.sequence_len,self.batch_size,self.hidden_dim))
        # Unfolding LSTM
        for i in range(self.sequence_len):
            hidden_state, cell_state = self.lstm_cell(out[i], (hidden_state, cell_state))
            final_out[i] = hidden_state
        return final_out


class UnfoldingLSTM_m(nn.Module):
    def __init__(self):
        super(UnfoldingLSTM_m, self).__init__()

        # Number of samples per time step
        self.batch_size = 2

        # Dimension of weight vectors
        self.hidden_dim = 16

        # Dimension of embedded tensor
        self.embedding_dim = 2

        # The vocabulary size
        self.input_size = 4

        # Number of time steps
        self.sequence_len = 2

        # Initialize embedding layer
        self.embedding = nn.Embedding(self.input_size, self.embedding_dim, padding_idx=0)

        # Initialize LSTM Cell
        self.lstm_cell1 = nn.LSTMCell(self.embedding_dim, self.hidden_dim)
        self.lstm_cell2 = nn.LSTMCell(self.hidden_dim,self.hidden_dim)
        self.num_layer = 2

        self.num_direction = 2

    def forward(self, x):

        # batch_size x hidden_size
        hidden_state = torch.zeros(x.size(0), self.hidden_dim)
        cell_state = torch.zeros(x.size(0), self.hidden_dim)
        hidden_state_2 = torch.zeros(x.size(0), self.hidden_dim)
        cell_state_2 = torch.zeros(x.size(0), self.hidden_dim)

        # weights initialization
        torch.nn.init.xavier_normal_(hidden_state)
        torch.nn.init.xavier_normal_(cell_state)
        torch.nn.init.xavier_normal_(hidden_state_2)
        torch.nn.init.xavier_normal_(cell_state_2)

        # From idx to embedding
        out = self.embedding(x)

        # Prepare the shape for LSTMCell
        out = out.view(self.sequence_len, x.size(0), -1)

        # Unfolding LSTM
        # Last hidden_state will be used to feed the fully connected neural net
        final_out = torch.zeros((self.sequence_len,self.batch_size,self.hidden_dim))
        h = torch.zeros(self.num_layer*self.num_direction,self.batch_size,self.hidden_dim)
        for i in range(self.sequence_len):
            hidden_state, cell_state = self.lstm_cell1(out[:,i,:], (hidden_state, cell_state))
            hidden_state_2, cell_state_2 = self.lstm_cell2(hidden_state, (hidden_state_2, cell_state_2))
            # 也可以使用append+concat式的方法
            final_out[i] = hidden_state_2
        # Last hidden state is passed through a fully connected neural net

        final_out_reverse = final_out.flip(dims=[-1])
        out = torch.cat([final_out,final_out_reverse],dim=2)

        return out

test = torch.rand(size=(2,2)).type(dtype=torch.long)

LSTM = UnfoldingLSTM_m()

out = LSTM(test)
print(out.shape)