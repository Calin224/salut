"""Modele PyTorch (scenariul E): MLP clasificator si autoencoder. Intrarile trebuie sa fie deja scalate."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin


def _train(net, tensors, loss_fn, epochs, lr, batch):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(tensors[0])).split(batch):
            opt.zero_grad()
            loss_fn(net, *[t[idx] for t in tensors]).backward()
            opt.step()
    net.eval()


def _t(X):
    return torch.tensor(np.asarray(X, dtype=np.float32))


class TorchMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden=256, epochs=20, lr=1e-3, batch=512):
        self.hidden, self.epochs, self.lr, self.batch = hidden, epochs, lr, batch

    def fit(self, X, y):
        torch.manual_seed(0)
        self.classes_, yi = np.unique(np.asarray(y), return_inverse=True)
        cnt = np.bincount(yi)
        w = torch.tensor(len(yi) / (len(cnt) * cnt), dtype=torch.float32)  # echilibrare pentru Macro F1
        X, h = _t(X), self.hidden
        self.net_ = nn.Sequential(nn.Linear(X.shape[1], h), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, len(cnt)))
        _train(self.net_, (X, torch.tensor(yi)), lambda n, xb, yb: F.cross_entropy(n(xb), yb, weight=w),
               self.epochs, self.lr, self.batch)
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            return F.softmax(self.net_(_t(X)), 1).numpy()


class TorchAE:
    def __init__(self, epochs=15, lr=1e-3, batch=512):
        self.epochs, self.lr, self.batch = epochs, lr, batch

    def fit(self, X):
        torch.manual_seed(0)
        X = _t(X)
        d = X.shape[1]
        self.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 8), nn.ReLU(),
                                 nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, d))
        _train(self.net, (X,), lambda n, xb: F.mse_loss(n(xb), xb), self.epochs, self.lr, self.batch)
        return self

    def score(self, X):
        """Eroarea de reconstructie pe rand: mare = neobisnuit."""
        X = _t(X)
        with torch.no_grad():
            return ((self.net(X) - X) ** 2).mean(1).numpy()
