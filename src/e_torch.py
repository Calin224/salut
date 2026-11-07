"""Scenariul E: MLP in PyTorch (necesita: pip install torch). python src/e_torch.py {1|2|3}"""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

from lib import fit_proba, finish, load, parse
from torch_models import TorchMLP

a = parse(__doc__)
X, y, Xt, ids = load(a.level, a.data, a.fe)
# QuantileTransformer: traficul de retea are cozi foarte lungi, retelele neuronale au nevoie de scala uniforma
model = make_pipeline(QuantileTransformer(output_distribution="normal"), TorchMLP())
classes, proba = fit_proba(model, X, y, Xt, a.level)
finish(a.level, ids, classes, proba, a.out, "E" + ("G" if a.fe else ""), a.thr, a.unk_frac)
