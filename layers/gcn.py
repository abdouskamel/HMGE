# Code taken from : https://github.com/PetarV-/DGI/blob/master/layers/gcn.py
import torch
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act=nn.ReLU(), drop_prob=0.0, bias=True):
        super().__init__()

        if drop_prob > 0.0:
            self.dropout = nn.Dropout(drop_prob)
        else:
            self.dropout = None

        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.init_weights(m)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        if self.dropout is not None:
            seq = self.dropout(seq)

        seq_fts = self.fc(seq)

        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, dim=0)), dim=0)
        else:
            out = torch.bmm(adj, seq_fts)

        if self.bias is not None:
            out += self.bias
        
        return self.act(out), seq_fts
