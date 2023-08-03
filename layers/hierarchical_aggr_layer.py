import torch
import torch.nn as nn
from layers import GCN

def normalize_adj_dense(adj):
    adj = adj + torch.eye(adj.shape[1], device=adj.device)
    d_inv_sqrt = torch.pow(torch.sum(adj, dim=2), -0.5).flatten()

    adj_norm = torch.mul(d_inv_sqrt, adj)
    adj_norm = torch.mul(adj_norm, d_inv_sqrt.reshape(-1, 1))

    return adj_norm

class HierarchicalAggrLayer(nn.Module):
    def __init__(self, in_dim, out_dim, in_rels, out_rels, drop_prob=0.0, is_attn=False, single_gcn=False):
        super().__init__()
        self.in_rels = in_rels
        self.out_rels = out_rels
        self.is_attn = is_attn

        if not single_gcn:
            self.gcns = nn.ModuleList([GCN(in_dim, out_dim, drop_prob=drop_prob) for _ in range(in_rels)])

        else:
            self.one_gcn = GCN(in_dim, out_dim, drop_prob=drop_prob)
            self.gcns = [self.one_gcn for _ in range(in_rels)]
            
        self.alphas = nn.Parameter(torch.FloatTensor(out_rels, in_rels))

        # For attention
        if is_attn:
            self.V = nn.ModuleList([nn.Linear(out_dim, out_dim, bias=False) for _ in range(in_rels)])
            self.Y = nn.ModuleList([nn.Linear(out_dim, 1) for _ in range(in_rels)])

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.alphas)

    def forward(self, z_in, adjs_in, sparse=False):
        z_out_all = []
        adjs_out = []

        # Compute Z_out
        for i in range(self.in_rels):
            z, _ = self.gcns[i](z_in, adjs_in[i], sparse)
            z_out_all.append(z)

        if not self.is_attn:
            z_out = torch.mean(torch.cat(z_out_all), dim=0).unsqueeze(0)
        
        else:
            z_out, _ = self.aggregate_with_attention(z_out_all)

        # Compute adjs_out
        alphas_softmax = torch.nn.functional.softmax(self.alphas, dim=1)

        for i in range(self.out_rels):
            adj = torch.zeros(adjs_in[0].shape)
            adj = adj.to(adjs_in[0].device)

            for j in range(self.in_rels):
                adj += alphas_softmax[i, j] * adjs_in[j]

            adjs_out.append(normalize_adj_dense(adj))
        
        return z_out, adjs_out

    def aggregate_with_attention(self, z_list):
        attn_scores = []

        for i in range(len(z_list)):
            alpha = self.V[i](z_list[i])
            alpha = self.Y[i](z_list[i])

            attn_scores.append(alpha)

        attn_scores = torch.cat(attn_scores, dim=-1)
        attn_scores = torch.tanh(attn_scores)
        attn_scores = torch.softmax(attn_scores, dim=-1)
        attn_scores = torch.unsqueeze(attn_scores, dim=-1)

        z = torch.stack(z_list, dim=2)
        z = attn_scores * z
        z = torch.sum(z, dim=2)

        return z, attn_scores
