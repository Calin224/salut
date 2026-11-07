"""Utilitare comune pentru scenariile A-G: date, feature engineering (G), evaluare, submisie."""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

TARGET = {1: "label", 2: "attack_cat", 3: "attack_cat"}


def parse(desc, levels=(1, 2, 3), extra=None):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("level", type=int, choices=levels)
    p.add_argument("--data", default="data", help="folder cu level{N}_train.csv / level{N}_test.csv")
    p.add_argument("--out", default="out", help="folder pentru submisii")
    p.add_argument("--fe", action="store_true", help="scenariul G: feature engineering")
    p.add_argument("--thr", type=float, help="nivel 3: prag absolut Unknown pe 1-max(prob), din scenariul D")
    p.add_argument("--unk-frac", type=float, default=0.05,
                   help="nivel 3, fara --thr: fractiunea de randuri marcate Unknown (ghicita!)")
    if extra:
        extra(p)
    return p.parse_args()


def features(d):
    # G: rapoarte intre sensuri, diferenta de TTL, log pe coloane cu coada lunga
    d = d.copy()
    d["bytes_ratio"] = (d.src_bytes + 1) / (d.dst_bytes + 1)
    d["pkts_ratio"] = (d.src_packets + 1) / (d.dst_packets + 1)
    d["src_bpp"] = d.src_bytes / (d.src_packets + 1)
    d["dst_bpp"] = d.dst_bytes / (d.dst_packets + 1)
    d["ttl_diff"] = d.src_ttl - d.dst_ttl
    for c in ["flow_duration", "src_bytes", "dst_bytes", "flow_rate", "src_load_bps", "dst_load_bps"]:
        d["log_" + c] = np.log1p(d[c].clip(lower=0))
    return d


def load(level, data, fe=False):
    tr = pd.read_csv(f"{data}/level{level}_train.csv")
    te = pd.read_csv(f"{data}/level{level}_test.csv")
    cols = [c for c in te.columns if c != "id"]
    y = tr[TARGET[level]]
    X, Xt = tr[cols], te[cols]
    if fe:
        X, Xt = features(X), features(Xt)
    clean = lambda d: d.replace([np.inf, -np.inf], np.nan).fillna(0)
    return clean(X), y, clean(Xt), te["id"]


def fit_proba(model, X, y, Xt, level):
    """Holdout 80/20 pentru un scor aproximativ, apoi refit pe tot train. Returneaza (classes, proba pe test)."""
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    model.fit(Xa, ya)
    pred = model.classes_[model.predict_proba(Xb).argmax(1)]
    f1 = f1_score(yb, pred) if level == 1 else f1_score(yb, pred, average="macro")
    # nivel 3: AUROC si clasa Unknown nu se pot masura aici, vezi scenariul D
    print(f"holdout {'F1' if level == 1 else 'Macro F1'}: {f1:.4f}")
    model.fit(X, y)
    return model.classes_, model.predict_proba(Xt)


def finish(level, ids, classes, proba, out, tag, thr=None, unk_frac=0.05, score=None):
    pred = classes[proba.argmax(1)]
    cols = {"id": ids, TARGET[level]: pred}
    if level == 3:
        score = 1 - proba.max(1) if score is None else score
        t = thr if thr is not None else np.quantile(score, 1 - unk_frac)
        cols[TARGET[level]] = np.where(score >= t, "Unknown", pred)
        cols["unknown_score"] = np.clip(score, 0, 1).round(6)
        print(f"Unknown: {np.mean(cols[TARGET[level]] == 'Unknown'):.1%} (prag {t:.4f})")
    df = pd.DataFrame(cols)
    assert df["id"].is_unique and not df.isna().any().any(), "submisie invalida"
    os.makedirs(out, exist_ok=True)
    path = f"{out}/level{level}_{tag}.csv"
    df.to_csv(path, index=False)
    print(f"scris {path} ({len(df)} randuri)")
