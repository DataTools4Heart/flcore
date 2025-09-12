# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# Uncertainty-Aware Neural Network
# Author: Jorge Fabila Fabian
# Fecha: September 2025
# Project: DT4H
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *

import torch
from typing import Dict, List, Tuple

@torch.no_grad()
def predict_proba_mc(self, x, T: int = 20):
    """Monte Carlo Dropout: devuelve prob. media y varianza por clase"""
    self.train() # Pone el modelo en modo train() para activar dropout durante inferencia.
    probs = []
    for _ in range(T):
        logits = self(x)
        probs.append(F.softmax(logits, dim=-1))
        probs = torch.stack(probs, dim=0) # [T, B, C]
        mean = probs.mean(dim=0)
        var = probs.var(dim=0) # var. epistemológica aprox.
        return mean, var

@torch.no_grad()
def predictive_entropy(self, x, T: int = 20):
    mean, _ = self.predict_proba_mc(x, T)
    eps = 1e-12
    ent = -(mean * (mean + eps).log()).sum(dim=-1) # [B]
    return ent


def uncertainty_metrics(model, val_loader, device="cpu", T: int = 20) -> Dict[str, float]:
    model.to(device)
    model.eval()
    ents = []
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            ent = model.predictive_entropy(x, T=T)
            ents.append(ent.cpu())
            # también accuracy con media predictiva
            mean, _ = model.predict_proba_mc(x, T=T)
            pred = mean.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
            entropy_mean = torch.cat(ents).mean().item()
            acc = correct / max(1, total)
    return {"entropy": float(entropy_mean), "val_accuracy": float(acc)}


# =================== LAS OTRAS

from typing import Dict, List
import numpy as np
import torch

def get_weights(model) -> List[np.ndarray]:
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]

def set_weights(model, weights: List[np.ndarray]):
    state_dict = model.state_dict()
    params = {k: torch.tensor(w) for k, w in zip(state_dict.keys(), weights)}
    model.load_state_dict(params)
