import torch
import torch.nn as nn

class LogRegModel(nn.Module):
    def __init__(self, fts_dim, nb_classes):
        super().__init__()

        self.fc = nn.Linear(fts_dim, nb_classes)

    def forward(self, fts):
        return self.fc(fts)

class MultilabelLogRegModel(nn.Module):
    def __init__(self, fts_dim, nb_classes):
        super().__init__()

        self.fc = nn.Linear(fts_dim, nb_classes)
        self.sigm = nn.Sigmoid()

    def forward(self, fts):
        return self.sigm(self.fc(fts))