import os, joblib, pandas as pd, numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing  import StandardScaler, LabelEncoder
from sklearn.pipeline       import Pipeline
from sklearn.ensemble       import RandomForestClassifier
from sklearn.metrics        import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline      import Pipeline as ImbPipeline

DATA_PATH  = Path("csv/file.csv")
MODEL_PATH = Path("model.pkl")

# ── Segment kuralları ─────────────────────────────────────────────────────────
SEG_MAP = {
    "High-Value Loyalists":  lambda r: r.R_Score==5 and r.F_Score==5,
    "Loyal":                 lambda r: r.F_Score==5 and r.R_Score>=4,
    "Potential Loyalists":   lambda r: r.R_Score==5 and r.F_Score==4,
    "Recent Customers":      lambda r: r.R_Score==5 and r.F_Score<=3,
    "Promising":             lambda r: r.R_Score==4 and r.F_Score>=3,
    "Need Attention":        lambda r: r.R_Score==3 and r.F_Score==3,
    "At Risk":               lambda r: r.R_Score<=2 and r.F_Score>=3,
    "Hibernating":           lambda r: r.R_Score<=2 and r.F_Score<=2,
}
def label_segment(row):
    for seg, rule in SEG_MAP.items():
        if rule(row):
            return seg
    return "Others"

# ── Veri yükle - özellik oluştur ------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)

df = pd.read_csv(DATA_PATH)
date_col = "Transaction_Date" if "Transaction_Date" in df.columns else "Date"
df["Transaction_Date"] = pd.to_datetime(df[date_col])

if "Total_Spend" not in df.columns:
    df["Total_Spend"] = df.get("Online_Spend", 0) + df.get("Offline_Spend", 0)

snapshot = df["Transaction_Date"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg(
        Recency   = ("Transaction_Date", lambda x: (snapshot - x.max()).days),
        Frequency = ("Transaction_ID",   "nunique"),
        Monetary  = ("Total_Spend",      "sum"),
        FirstDate = ("Transaction_Date", "min"),
        OnlineSum = ("Online_Spend",     "sum")
     )

# Quintile skorları & segment etiketi
rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5,4,3,2,1]).astype(int)
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5,
                         labels=[1,2,3,4,5]).astype(int)
rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1,2,3,4,5]).astype(int)
rfm["Segment"] = rfm.apply(label_segment, axis=1)

# Ek davranışsal özellikler
rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]
rfm["TenureDays"]    = (snapshot - rfm["FirstDate"]).dt.days
rfm["OnlineRatio"]   = np.where(rfm["Monetary"]>0,
                                rfm["OnlineSum"] / rfm["Monetary"], 0)

FEATURES = ["Recency","Frequency","Monetary",
            "AvgOrderValue","TenureDays","OnlineRatio"]

X = rfm[FEATURES].values
y = rfm["Segment"].values

# ── Etiket kodlama & veri bölme ------------------------------------------------
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc)

# ── SMOTE için dinamik k_neighbors (min(class_size)-1) -------------------------
minority_count = min(Counter(y_train).values())
smote_k = max(1, min(5, minority_count-1))
print(f"SMOTE k_neighbors set to {smote_k}  (smallest class size = {minority_count})")

pipe = ImbPipeline([
    ("smote",  SMOTE(k_neighbors=smote_k, random_state=42)),
    ("scaler", StandardScaler()),
    ("clf",    RandomForestClassifier(random_state=42))
])

param_grid = {
    "clf__n_estimators":      [400, 700],
    "clf__max_depth":         [None, 20, 40],
    "clf__min_samples_leaf":  [1, 3, 5],
    "clf__class_weight":      [None, "balanced"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs = GridSearchCV(pipe, param_grid,
                  scoring="f1_weighted",
                  cv=cv, n_jobs=-1, verbose=1)

gs.fit(X_train, y_train)
print("\nBest CV weighted-F1:", round(gs.best_score_, 4))
print("Best params:", gs.best_params_)

best_model = gs.best_estimator_

# ── Hold-out test metrikleri ---------------------------------------------------
print("\n— Hold-out test metrics —")
y_pred = best_model.predict(X_test)
print(classification_report(
        y_test, y_pred, target_names=le.classes_, zero_division=0))

# ── Özellik önemleri -----------------------------------------------------------
rf = best_model.named_steps["clf"]
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nTop features:")
print(imp.head(10))

# ── Modeli kaydet --------------------------------------------------------------
joblib.dump({"pipe": best_model, "label_enc": le}, MODEL_PATH)
print("\n✅  Model saved →", MODEL_PATH.resolve())
