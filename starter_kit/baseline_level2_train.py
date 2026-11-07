"""
AI Cyber Defense Challenge — Nivelul 2 Baseline (.py)
Clasificare Multi-clasa (Trafic Normal si 9 categorii de atac)

Acest script este doar un punct de plecare minimal (intentionat neoptimizat)
pentru a demonstra codificarea etichetelor si generarea unui CSV valid.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "level2_train.csv")
TEST_PATH = os.path.join(BASE_DIR, "level2_test.csv")
OUT_PATH = os.path.join(BASE_DIR, "level2_submission.csv")

if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
    print("[EROARE] Nu am gasit fisierele de date pentru Nivelul 2!")
    print("Descarca 'Train CSV' si 'Test CSV' pentru Nivelul 2 de pe pagina /submit")
    print(f"si salveaza-le in acelasi folder cu acest script:\n  - {TRAIN_PATH}\n  - {TEST_PATH}")
    sys.exit(1)

# 1. Incarcarea datelor
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Train: {train_df.shape[0]} randuri, {train_df.shape[1]} coloane")
print(f"Test:  {test_df.shape[0]} randuri, {test_df.shape[1]} coloane")

# 2. Codificarea etichetelor text (attack_cat)
le = LabelEncoder()
y = le.fit_transform(train_df["attack_cat"])
class_indices = np.arange(len(le.classes_))

X = train_df.drop(columns=["attack_cat", "id", "label"], errors="ignore")
feature_cols = X.select_dtypes(include="number").columns.intersection(test_df.columns).tolist()
X = X[feature_cols].replace([np.inf, -np.inf], np.nan)

test_ids = test_df["id"]
X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan)

print("Clase identificate:", list(le.classes_))

# 3. Validare locala cu Macro F1-Score
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = HistGradientBoostingClassifier(max_iter=160, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

val_preds = model.predict(X_val)
macro_f1 = f1_score(y_val, val_preds, labels=class_indices, average="macro", zero_division=0)

print(f"Macro F1-Score validare locala: {macro_f1:.4f}\n")
print(classification_report(y_val, val_preds, labels=class_indices, target_names=le.classes_, zero_division=0))

# 4. Antrenare pe tot setul si generare submisie CSV
model.fit(X, y)
test_preds_encoded = model.predict(X_test)
test_preds = le.inverse_transform(test_preds_encoded)

submission = pd.DataFrame({
    "id": test_ids,
    "attack_cat": test_preds,
})

submission.to_csv(OUT_PATH, index=False)
print(f"Fisier generat cu succes: {OUT_PATH} ({len(submission)} randuri)")
print(submission.head())
