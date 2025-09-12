import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicNN(nn.Module):
    def __init__(self,n_feats, n_out , p: float = 0.2):
        super().__init__()
        print("NFEATS", n_feats)
        self.fc1 = nn.Linear(n_feats, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_out)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        return logits
# Igual tendríamos que añadir la función de train aquí mismo
"""
         self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),          # dropout para MC Dropout si lo quieres
            nn.Linear(64, num_classes)
        ).to(self.device)
"""
