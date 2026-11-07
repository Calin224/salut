"""Scenariul F: ierarhic. Pas 1 Normal vs Atac, pas 2 clasa de atac. python src/f_hierarchical.py {2|3}"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier as HGB

from lib import fit_proba, finish, load, parse


class Hier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        self.gate_ = HGB(random_state=0).fit(X, y == "Normal")
        atk = y != "Normal"
        self.attack_ = HGB(class_weight="balanced", random_state=0).fit(X[atk], y[atk])
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        p_norm = self.gate_.predict_proba(X)[:, 1]
        p_atk = self.attack_.predict_proba(X) * (1 - p_norm)[:, None]
        cols = {c: p_atk[:, i] for i, c in enumerate(self.attack_.classes_)}
        cols["Normal"] = p_norm
        return np.column_stack([cols[c] for c in self.classes_])


a = parse(__doc__, levels=(2, 3))
X, y, Xt, ids = load(a.level, a.data, a.fe)
classes, proba = fit_proba(Hier(), X, y, Xt, a.level)
finish(a.level, ids, classes, proba, a.out, "F" + ("G" if a.fe else ""), a.thr, a.unk_frac)
