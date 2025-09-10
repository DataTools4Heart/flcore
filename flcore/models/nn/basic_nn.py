import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicNN(nn.Module):
    def __init__(self,n_feats, n_out , p: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
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
