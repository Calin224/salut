"""
AI Cyber Defense Challenge — Nivelul 1 Baseline (.py)
Clasificare Binara: Trafic Normal (0) vs Atac Cibernetic (1)

Acest script este doar un punct de plecare minimal (intentionat neoptimizat)
pentru a demonstra incarcarea datelor si generarea unui fisier CSV valid.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "level1_train.csv")
TEST_PATH = os.path.join(BASE_DIR, "level1_test.csv")
OUT_PATH = os.path.join(BASE_DIR, "level1_submission.csv")

if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
    print("[EROARE] Nu am gasit fisierele de date pentru Nivelul 1!")
    print("Descarca 'Train CSV' si 'Test CSV' pentru Nivelul 1 de pe pagina /submit")
    print(f"si salveaza-le in acelasi folder cu acest script:\n  - {TRAIN_PATH}\n  - {TEST_PATH}")
    sys.exit(1)

# 1. Incarcarea datelor
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Train: {train_df.shape[0]} randuri, {train_df.shape[1]} coloane")
print(f"Test:  {test_df.shape[0]} randuri, {test_df.shape[1]} coloane")

# 2. Preprocesare de baza
X = train_df.drop(columns=["label"])
y = train_df["label"]

feature_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
X = X[feature_cols].fillna(0)

test_ids = test_df["id"]
X_test = test_df[feature_cols].fillna(0)

print(f"Features numerice selectate: {len(feature_cols)}")

# 3. Validare locala (Train / Validation Split)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)

val_preds = model.predict(X_val)
score = f1_score(y_val, val_preds)

print(f"Scor F1 validare locala: {score:.4f}\n")
print(classification_report(y_val, val_preds, zero_division=0))

# 4. Antrenare pe tot setul si generare submisie CSV
model.fit(X, y)
test_preds = model.predict(X_test).astype(int)

submission = pd.DataFrame({
    "id": test_ids,
    "label": test_preds,
})

submission.to_csv(OUT_PATH, index=False)
print(f"Fisier generat cu succes: {OUT_PATH} ({len(submission)} randuri)")
print(submission.head())
