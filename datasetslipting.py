import pandas as pd
import numpy as np

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data.csv", encoding="latin-1", on_bad_lines="skip",
                 sep=None, engine="python")

# ── Triage label from KTAS_expert (1=most urgent, 5=least urgent) ────────────
def ktas_to_triage(k):
    try:
        k = int(float(k))
        if k <= 2: return 2      # Red   (immediate/emergent)
        elif k == 3: return 1    # Yellow (urgent)
        else: return 0           # Green  (semi/non-urgent)
    except:
        return 1                 # default Yellow if missing

df["triage_label"] = df["KTAS_expert"].apply(ktas_to_triage)
df["triage_name"]  = df["triage_label"].map({0:"Green", 1:"Yellow", 2:"Red"})

# ── Standardise columns ───────────────────────────────────────────────────────
df["age"]                  = pd.to_numeric(df["Age"], errors="coerce").fillna(df["Age"].median())
df["gender"]               = df["Sex"].map({1:"M", 2:"F"}).fillna("M")
df["temperature_celsius"]  = pd.to_numeric(df["BT"], errors="coerce").fillna(37.0)
df["heart_rate"]           = pd.to_numeric(df["HR"], errors="coerce").fillna(80)
df["oxygen_saturation"]    = pd.to_numeric(df["Saturation"], errors="coerce").fillna(98)
df["systolic_bp"]          = pd.to_numeric(df["SBP"], errors="coerce").fillna(120)
df["symptom_duration_days"]= 2   # not in dataset, use neutral default

# ── Comorbidities (proxies) ───────────────────────────────────────────────────
df["diabetes"]          = 0   # not in dataset
df["hypertension"]      = (df["systolic_bp"] > 140).astype(int)
df["immunocompromised"] = 0

# ── Symptoms from Chief_complain text ─────────────────────────────────────────
cc = df["Chief_complain"].fillna("").str.lower()

df["fever"]               = cc.str.contains("fever|febrile|temp", na=False).astype(int)
df["cough"]               = cc.str.contains("cough", na=False).astype(int)
df["shortness_of_breath"] = cc.str.contains("breath|dyspnea|sob|respiratory", na=False).astype(int)
df["chest_pain"]          = cc.str.contains("chest", na=False).astype(int)
df["headache"]            = cc.str.contains("head|headache|migraine", na=False).astype(int)
df["fatigue"]             = cc.str.contains("fatigue|weakness|dizz", na=False).astype(int)
df["nausea"]              = cc.str.contains("nausea|nausea", na=False).astype(int)
df["vomiting"]            = cc.str.contains("vomit|vomiting", na=False).astype(int)
df["diarrhea"]            = cc.str.contains("diarrhea|diarrhoea", na=False).astype(int)
df["rash"]                = cc.str.contains("rash|skin|urticaria", na=False).astype(int)
df["joint_pain"]          = cc.str.contains("joint|arthral", na=False).astype(int)
df["loss_of_smell"]       = cc.str.contains("smell|anosmia|taste", na=False).astype(int)

# ── Also flag pain from Pain column ──────────────────────────────────────────
df["fatigue"] = ((df["fatigue"] == 1) | (df["Pain"] == 1)).astype(int)

# ── Travel history (synthetic — not in dataset) ───────────────────────────────
np.random.seed(42)
df["travel_history"] = np.random.choice(
    ["none","domestic","southeast_asia","africa","europe","middle_east"],
    size=len(df), p=[0.65, 0.15, 0.08, 0.05, 0.05, 0.02]
)

# ── Keep only needed columns ──────────────────────────────────────────────────
final_cols = [
    "age","gender","diabetes","hypertension","immunocompromised",
    "fever","cough","shortness_of_breath","chest_pain","headache",
    "fatigue","nausea","vomiting","diarrhea","rash","joint_pain","loss_of_smell",
    "travel_history","temperature_celsius","heart_rate",
    "oxygen_saturation","systolic_bp","symptom_duration_days",
    "triage_label","triage_name"
]
df_clean = df[final_cols].dropna(subset=["age","heart_rate","systolic_bp"])

print(f"Clean dataset: {df_clean.shape}")
print(df_clean["triage_name"].value_counts())

# ── Split into 3 clinics by age (non-IID for federated learning) ──────────────
import os
os.makedirs("data/raw", exist_ok=True)

clinic_1 = df_clean[df_clean["age"] >= 50].copy()           # Urban/Elderly
clinic_2 = df_clean[(df_clean["age"] >= 18) & (df_clean["age"] < 50)].copy()  # Adults
clinic_3 = df_clean[df_clean["age"] < 18].copy()            # Pediatric

# If any clinic is too small, fallback to random split
if len(clinic_3) < 50:
    print("⚠️  Pediatric group too small — using random split instead")
    df_shuffled = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df_shuffled)
    clinic_1 = df_shuffled.iloc[:n//3]
    clinic_2 = df_shuffled.iloc[n//3:2*n//3]
    clinic_3 = df_shuffled.iloc[2*n//3:]

clinic_1.to_csv("data/raw/clinic_1.csv", index=False)
clinic_2.to_csv("data/raw/clinic_2.csv", index=False)
clinic_3.to_csv("data/raw/clinic_3.csv", index=False)

print(f"\n✅ clinic_1 (elderly 50+):  {len(clinic_1)} patients")
print(f"✅ clinic_2 (adult 18-49):  {len(clinic_2)} patients")
print(f"✅ clinic_3 (pediatric <18): {len(clinic_3)} patients")
print("\nDone! Run: python train.py")