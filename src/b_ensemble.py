"""Scenariul B: ansamblu soft voting (HistGB + RandomForest + ExtraTrees). python src/b_ensemble.py {1|2|3}"""
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)

from lib import fit_proba, finish, load, parse

a = parse(__doc__)
X, y, Xt, ids = load(a.level, a.data, a.fe)
bal = a.level > 1
model = VotingClassifier([
    ("hgb", HistGradientBoostingClassifier(class_weight="balanced" if bal else None, random_state=0)),
    ("rf", RandomForestClassifier(300, class_weight="balanced_subsample" if bal else None, n_jobs=-1, random_state=0)),
    ("et", ExtraTreesClassifier(300, class_weight="balanced_subsample" if bal else None, n_jobs=-1, random_state=0)),
], voting="soft")
classes, proba = fit_proba(model, X, y, Xt, a.level)
finish(a.level, ids, classes, proba, a.out, "B" + ("G" if a.fe else ""), a.thr, a.unk_frac)
