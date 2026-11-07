"""
AI Cyber Defense Challenge — Nivelul 3 Baseline (.py)
Detectare Open-Set (Known Attacks + Unknown Novel Attacks)

Acest script este doar un punct de plecare minimal (intentionat neoptimizat)
care ilustreaza formatul cu 3 coloane: id, attack_cat, unknown_score.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "level3_train.csv")
TEST_PATH = os.path.join(BASE_DIR, "level3_test.csv")
OUT_PATH = os.path.join(BASE_DIR, "level3_submission.csv")

if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
    print("[EROARE] Nu am gasit fisierele de date pentru Nivelul 3!")
    print("Descarca 'Train CSV' si 'Test CSV' pentru Nivelul 3 de pe pagina /submit")
    print(f"si salveaza-le in acelasi folder cu acest script:\n  - {TRAIN_PATH}\n  - {TEST_PATH}")
    sys.exit(1)

# 1. Incarcarea datelor
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Train: {train_df.shape[0]} randuri, {train_df.shape[1]} coloane")
print(f"Test:  {test_df.shape[0]} randuri, {test_df.shape[1]} coloane")

# 2. Codificarea etichetelor cunoscute
le = LabelEncoder()
y = le.fit_transform(train_df["attack_cat"])

X = train_df.drop(columns=["attack_cat"])
feature_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
X = X[feature_cols].fillna(0)

test_ids = test_df["id"]
X_test = test_df[feature_cols].fillna(0)

print("Clase cunoscute in antrenare:", list(le.classes_))

# 3. Antrenare model minimal + estimare probabilitati pe test
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X, y)

probs = model.predict_proba(X_test)
max_probs = np.max(probs, axis=1)
pred_classes = np.argmax(probs, axis=1)

print(f"Probabilitatea medie de incredere: {max_probs.mean():.4f}")

# 4. Aplicarea unui prag minimal pentru Open-Set (Thresholding)
unknown_scores = np.clip(1.0 - max_probs, 0.0, 1.0)
THRESHOLD_SCORE = np.percentile(unknown_scores, 98.0)

known_labels = le.inverse_transform(pred_classes)
final_predictions = np.where(unknown_scores >= THRESHOLD_SCORE, "Unknown", known_labels)

unknown_cnt = int(np.sum(final_predictions == "Unknown"))
print(f"Clasificate ca Unknown: {unknown_cnt} din {len(final_predictions)} ({unknown_cnt / len(final_predictions) * 100:.1f}%)")

# 5. Generare submisie level3_submission.csv
submission = pd.DataFrame({
    "id": test_ids,
    "attack_cat": final_predictions,
    "unknown_score": np.round(unknown_scores, 4),
})

submission.to_csv(OUT_PATH, index=False)
print(f"Fisier generat cu succes: {OUT_PATH} ({len(submission)} randuri)")
print(submission.head(10))
