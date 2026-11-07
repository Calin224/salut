"""Scenariul C (nivel 3): clasificator + detector de anomalii pentru unknown_score.
python src/c_anomaly.py 3 [--detector iso|ae] [--w 0.5] [--unk-frac 0.05]
--w = ponderea detectorului in scor (restul = 1-max(prob)); --thr nu se aplica aici (scorul e pe ranguri)."""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.preprocessing import QuantileTransformer

from lib import fit_proba, finish, load, parse


def extra(p):
    p.add_argument("--detector", choices=["iso", "ae"], default="iso")
    p.add_argument("--w", type=float, default=0.5)


a = parse(__doc__, levels=(3,), extra=extra)
X, y, Xt, ids = load(3, a.data, a.fe)
classes, proba = fit_proba(HistGradientBoostingClassifier(class_weight="balanced", random_state=0), X, y, Xt, 3)

if a.detector == "iso":
    det = IsolationForest(n_estimators=200, random_state=0, n_jobs=-1).fit(X)
    s = -det.score_samples(Xt)
else:
    from torch_models import TorchAE  # necesita torch
    qt = QuantileTransformer(output_distribution="normal").fit(X)
    s = TorchAE().fit(qt.transform(X)).score(qt.transform(Xt))

# AUROC depinde doar de ordine, deci amestecam rangurile (0..1) ale celor doua scoruri
rank = lambda v: pd.Series(v).rank(pct=True).to_numpy()
score = a.w * rank(s) + (1 - a.w) * rank(1 - proba.max(1))
finish(3, ids, classes, proba, a.out, f"C_{a.detector}", unk_frac=a.unk_frac, score=score)
