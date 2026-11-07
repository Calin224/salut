"""Scenariul A: gradient boosting simplu. Rulare: python src/a_gbm.py {1|2|3} [--fe] [--thr X]"""
from sklearn.ensemble import HistGradientBoostingClassifier

from lib import fit_proba, finish, load, parse

a = parse(__doc__)
X, y, Xt, ids = load(a.level, a.data, a.fe)
# Macro F1 (nivelurile 2, 3) cere clase echilibrate
model = HistGradientBoostingClassifier(class_weight=None if a.level == 1 else "balanced", random_state=0)
classes, proba = fit_proba(model, X, y, Xt, a.level)
finish(a.level, ids, classes, proba, a.out, "A" + ("G" if a.fe else ""), a.thr, a.unk_frac)
