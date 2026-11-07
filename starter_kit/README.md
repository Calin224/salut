# Starter Kit — AI Cyber Defense Challenge
## Zilele Tehnicii Studențești (ZTS) — Ediția a XIII-a

Acest pachet conține resursele de bază și exemplele de cod (atât ca scripturi clasice `.py`, cât și ca notebook-uri Jupyter `.ipynb`) pentru a genera primele submisii valide. Modelele din acest starter kit sunt **intenționat minimale** (un `LogisticRegression` de bază pe date brute) — rolul lor este doar să îți arate cum se încarcă datele și cum se salvează CSV-ul în formatul așteptat de platformă.

> **Documentația tehnică a competiției:** [https://hackathon.osace.ro/guide](https://hackathon.osace.ro/guide)  
> Consultă pagina `/guide` pentru descrierea celor 39 de caracteristici din dataset, metricile de evaluare și regulile probei.

---

## Structura arhivei

```
starter_kit/
├── README.md                   # Instrucțiuni de instalare, rulare și depanare
├── requirements.txt            # Librării Python de bază (pandas, numpy, scikit-learn)
├── baseline_level1.py          # Script Python direct executabil — Nivelul 1
├── baseline_level2.py          # Script Python direct executabil — Nivelul 2
├── baseline_level3.py          # Script Python direct executabil — Nivelul 3
├── baseline_level1.ipynb       # Notebook Jupyter echivalent — Nivelul 1
├── baseline_level2.ipynb       # Notebook Jupyter echivalent — Nivelul 2
├── baseline_level3.ipynb       # Notebook Jupyter echivalent — Nivelul 3
└── submission_example.csv      # Mostră de structură (doar primele 4 rânduri, Nivel 1)
```

---

## Instalare și rulare rapidă

### Varianta A — Cu mediu virtual (`venv`, recomandat)

```powershell
# 1. Creează și activează un mediu virtual în folderul starter_kit
python -m venv venv

# Activare pe Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Activare pe Windows (CMD):
.\venv\Scripts\activate.bat
# Activare pe Linux / macOS:
source venv/bin/activate

# 2. Instalează pachetele necesare
python -m pip install -r requirements.txt

# 3. Rulează scriptul dorit (după ce ai pus level1_train.csv și level1_test.csv în acest folder)
python baseline_level1.py
```

### Varianta B — Direct din terminal (fără `venv`)

> **Important pe Windows:** Folosește întotdeauna `python -m pip` (sau `py -m pip` dacă rulezi scripturile cu comanda `py`), astfel încât librăriile să se instaleze în **exact aceeași versiune de Python** cu care execuți scriptul!

```powershell
# Dacă rulezi cu `python`:
python -m pip install -r requirements.txt
python baseline_level1.py

# Dacă rulezi cu `py` (Windows Python Launcher):
py -m pip install -r requirements.txt
py baseline_level1.py
```

---

## Depanare rapidă (Troubleshooting)

1. **`ModuleNotFoundError: No module named 'sklearn'`** (chiar și după ce ai dat `pip install`):
   - Ai mai multe versiuni de Python instalate pe calculator (de ex. Python 3.11 și Python 3.14), iar comanda `pip` instalează într-o versiune în timp ce `py` sau `python` pornește cealaltă versiune.
   - **Soluție:** Rulează instalarea cu prefixul interpretului pe care îl folosești:
     - `py -m pip install -r requirements.txt` (dacă rulezi cu `py baseline_level1.py`), sau
     - `python -m pip install -r requirements.txt` (dacă rulezi cu `python baseline_level1.py`).
2. **Eroare la `pip install sklearn` sau `from scikit-learn import ...`:**
   - Pachetul pe PyPI se numește **`scikit-learn`** (`python -m pip install scikit-learn`), însă în codul Python modulul se importă întotdeauna ca **`sklearn`** (`from sklearn.linear_model import ...`).
3. **`FileNotFoundError` pentru `level1_train.csv` / `level1_test.csv`:**
   - Descarcă seturile de date de pe pagina `/submit` (după autentificarea cu token-ul echipei) și salvează-le în **același folder** cu scripturile `baseline_level*.py` (numele implicite la descărcare sunt `level1_train.csv`, `level1_test.csv` etc.).

---

## Formula scorului final

$$\text{Scor Total (\%)} = 0.20 \times \text{Nivel 1} + 0.30 \times \text{Nivel 2} + 0.30 \times \text{Nivel 3} + 0.20 \times \text{Raport Tehnic}$$

- **Nivelul 1 (20%):** Clasificare Binară (`F1-Score Binar`)
- **Nivelul 2 (30%):** Clasificare Multi-clasă (`Macro F1-Score` pe 10 clase)
- **Nivelul 3 (30%):** Detectare Open-Set (`Scor Compozit`: 60% Macro F1 pe 8 etichete + 40% AUROC pe `unknown_score`)
- **Raport Tehnic & Cod Sursă (20%):** Evaluare Juriu (Metodologie, documentare & validare cod sursă)

---

## Formatul submisiilor per nivel

Fiecare echipă are dreptul la **maximum 10 submisii per nivel** (frecvență: 1 submisie la 2 minute). Se ia în considerare **cel mai mare scor obținut**. Submisiile invalide structural **nu consumă** din cele 10 încercări.
*(Notă: fișierul `submission_example.csv` din arhivă conține doar 4 rânduri ca exemplu vizual de header — o submisie reală trebuie să conțină toate rândurile din `test.csv`).*

### Nivelul 1 — Clasificare Binară
- **Coloane cerute:** `id,label`
- **Valori permise:** `0` (Trafic Normal) sau `1` (Atac Cibernetic)

### Nivelul 2 — Clasificare Multi-clasă (10 Clase)
- **Coloane cerute:** `id,attack_cat`
- **Valori permise (case-sensitive):** `Normal`, `Generic`, `Exploits`, `Fuzzers`, `DoS`, `Reconnaissance`, `Analysis`, `Backdoor`, `Shellcode`, `Worms`

### Nivelul 3 — Detectare Open-Set (Known + Unknown)
- **Coloane cerute:** `id,attack_cat,unknown_score`
- **Valori permise:**
  - `attack_cat`: Cele 7 clase cunoscute din `level3_train.csv` (`Normal`, `Generic`, `Exploits`, `Fuzzers`, `DoS`, `Reconnaissance`, `Backdoor`) sau `Unknown` (pentru atacurile noi din test).
  - `unknown_score`: Număr zecimal float între `0.0` și `1.0` (probabilitatea/scorul de anomalie că mostra este un atac necunoscut).

---

*Documentație realizată de echipa de organizare OSACE × ZTS.*
