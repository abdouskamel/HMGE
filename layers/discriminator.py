import torch
import torch.nn as nn

class DiscriminatorDGI(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()

        self.bilinear = nn.Bilinear(hid_dim, hid_dim, 1)

        for m in self.modules():
            self.init_weights(m)

    def init_weights(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, s, h_pos, h_neg):
        s = s.expand_as(h_pos)

        logits_pos = torch.squeeze(self.bilinear(h_pos, s), dim=2)
        logits_neg = torch.squeeze(self.bilinear(h_neg, s), dim=2)

        return torch.cat((logits_pos, logits_neg), dim=1)
