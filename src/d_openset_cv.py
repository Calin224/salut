"""Scenariul D (nivel 3): simuleaza open-set scotand pe rand cate o clasa din train si tratand-o Unknown.
Alege pragul absolut pe 1-max(prob). python src/d_openset_cv.py 3 [--fe]  ->  apoi: a_gbm.py 3 --thr <valoarea>
Limita: in simulare lipseste o clasa, in test lipsesc mai multe; pragul e o estimare, nu adevarul."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from lib import load, parse

a = parse(__doc__, levels=(3,))
X, y, _, _ = load(3, a.data, a.fe)
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
held = [c for c in sorted(y.unique()) if c != "Normal"]
grid = np.linspace(0.05, 0.95, 19)
res = np.zeros((len(held), len(grid)))

for i, k in enumerate(held):
    keep = (ya != k).to_numpy()
    m = HistGradientBoostingClassifier(class_weight="balanced", random_state=0).fit(Xa[keep], ya[keep])
    p = m.predict_proba(Xb)
    s, pred = 1 - p.max(1), m.classes_[p.argmax(1)]
    unk = (yb == k).to_numpy()
    auc = roc_auc_score(unk, s)
    truth = np.where(unk, "Unknown", yb)
    for j, t in enumerate(grid):
        res[i, j] = 0.6 * f1_score(truth, np.where(s >= t, "Unknown", pred), average="macro") + 0.4 * auc
    print(f"clasa scoasa {k:15s} AUROC {auc:.3f}  scor nivel3 la cel mai bun prag {res[i].max():.3f}")

mean = res.mean(0)
print(f"\nmedie pe clase scoase: cel mai bun prag = {grid[mean.argmax()]:.2f} (scor {mean.max():.3f})")
print(f"-> python src/a_gbm.py 3 --thr {grid[mean.argmax()]:.2f}")
