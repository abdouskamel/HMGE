import torch
import torch.nn as nn

from layers import HierarchicalAggrLayer, GCN, DiscriminatorDGI

def l2_normalize_tensor(x):
    norm = x.norm(p=2, dim=2, keepdim=True)
    norm = torch.where(norm==0, torch.tensor(1.0).to(norm.device), norm)

    return x.div(norm.expand_as(x))

class HMGE(nn.Module):
    def __init__(self, fts_dim, hid_dim, nb_relations, drop_prob=0.0, is_attn=False, common_gcn=False, single_gcn=False, normalize_z=False):
        super().__init__()
        self.hid_dim = hid_dim
        self.common_gcn = common_gcn
        self.normalize_z = normalize_z
        
        if self.common_gcn:
            self.common_gcn = GCN(fts_dim, hid_dim, drop_prob=drop_prob)
            self.hier_layer = HierarchicalAggrLayer(hid_dim, hid_dim, nb_relations, 1, drop_prob, is_attn, single_gcn)

        else:
            self.hier_layer = HierarchicalAggrLayer(fts_dim, hid_dim, nb_relations, 1, drop_prob, is_attn, single_gcn)
        
        self.gcn = GCN(hid_dim, hid_dim, drop_prob=drop_prob)
        self.disc = DiscriminatorDGI(hid_dim)

    def forward(self, fts, adjs_norm, fts_shuf, sparse=False):
        z_pos = self.embed(fts, adjs_norm, sparse)
        z_neg = self.embed(fts_shuf, adjs_norm, sparse)

        s = torch.mean(z_pos, dim=1)
        s = torch.sigmoid(s)
        s = torch.unsqueeze(s, dim=1)

        logits = self.disc(s, z_pos, z_neg)
        return logits

    def embed(self, fts, adjs_norm, sparse=False):
        if self.common_gcn:
            hid_fts = 0
            for adj in adjs_norm:
                hid_fts += self.common_gcn(fts, adj, sparse)[0]

            hid_fts /= len(adjs_norm)
            z, new_adjs = self.hier_layer(hid_fts, adjs_norm, sparse)

        else:
            z, new_adjs = self.hier_layer(fts, adjs_norm, sparse)
            self.z_mid = z

        z, _ = self.gcn(z, new_adjs[0], sparse)
        if self.normalize_z:
            z = l2_normalize_tensor(z)

        return z
