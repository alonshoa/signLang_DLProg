
import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self,hidden_dim,RNN_layers=3):
        super(Attention, self).__init__()
        self.rnn_layers = RNN_layers
        self.register_parameter('v', torch.nn.Parameter(torch.empty((1, hidden_dim), dtype=torch.float)))
        self.att_w = torch.nn.Linear(in_features=2 * hidden_dim, out_features=hidden_dim)
        self.initialize_param()

    def initialize_param(self):
        torch.nn.init.kaiming_uniform_(self.v)

    def forward(self,x,h):
        # print("out.shape",x.shape)
        # print("h.shape",h.shape)
        h = h[-1].repeat(x.shape[0] , 1, 1)
        # print("out.shape",x.shape)
        # print("h.shape",h.shape)

        t = torch.cat([h, x], 2)

        score = torch.tanh(self.att_w(t))
        out = score @ self.v.T
        alphas = F.softmax(out, dim=1)
        # return alphas

        out_att = alphas.transpose(1,2)@x
        return out_att


class HebLetterToSentence(nn.Module):
    def __init__(self, vocab_size,embedding_dim, lstm_size,hidden_dim, output_dim):
        super(HebLetterToSentence, self).__init__()
        self.lstm_size = lstm_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = 3


        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embedding_dim,
        )
        self.gru = nn.GRU(
            input_size=self.lstm_size,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(self.lstm_size, output_dim)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        # print("embed.shape", embed.shape)
        output, state = self.gru(embed, prev_state)
        # print("out.shape", output.shape)
        outputs = self.attention(output, state)
        # print("outs.shape", outputs.shape)

        logits = self.fc(outputs)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size))


class ImgToWordModel(nn.Module):
    def __init__(self, resnet_model, seq_model):
        self.renet = resnet_model
        self.seq = seq_model


    def forward(self,x):
        x_last,y = self.resnet(x)

        res = self.seq(x_last)
        return res

if __name__ == '__main__':
    # from torchsummary import summary
    # dl = create_
    model = HebLetterToSentence(30,128,128,30)
    h,c = model.init_state(3)
    words = torch.tensor([1,2,3]).reshape(1,-1)
    res = model(words,(h,c))
    print(res[0].shape)
    print(res[1][0].shape)
    print(res[1][1].shape)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
#
# class HebLetterToSentence(nn.Module):
#     def __init__(self,vocab_size,embedding_dim,input_size,output_dim):
#         super(HebLetterToSentence, self).__init__()
#         self.embedding = nn.Embedding(
#             vocab_size,
#             embedding_dim
#         )
#         self.lin = nn.Linear(
#             input_size * embedding_dim,
#             output_dim
#         )
#
#     def forward(self,x):
#         return x