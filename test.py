# multi_clustering_v3.py
"""
3‑Yöntemli Müşteri Segmentasyonu
* K‑Means, Hierarchical (Ward), DBSCAN
* Otomatik k seçimi (≥ DESIRED_MIN_CLUSTERS, Silhouette ≥ 0.20)
* Küme AÇIKLAMALARI → heuristik kurallarla adlandırılır
* 2‑D PCA scatter grafikleri (Plotly) her algoritma için
* Çıktılar: customers_with_clusters.csv,
           profiles_{algo}.csv,
           cluster_desc_{algo}.json,
           scatter_{algo}.html
"""
import pandas as pd, numpy as np, itertools, warnings, json, plotly.express as px
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- PARAMETRELER ----------
FILE_PATH = Path("csv/file.csv")
DATE_COL  = "Transaction_Date"               # yoksa ""
ID_COL    = "CustomerID"
DESIRED_MIN_CLUSTERS = 4
# -----------------------------------

# 1) VERİ OKU -----------------------------------------------------------
df = pd.read_csv(FILE_PATH,
                 parse_dates=[DATE_COL] if DATE_COL and DATE_COL in pd.read_csv(FILE_PATH, nrows=0).columns else [])
num_cols = ["Tenure_Months","Offline_Spend","Online_Spend","Quantity"]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df = df.dropna(subset=[ID_COL])

# 2) ÖZELLİK MÜHENDİSLİĞİ ----------------------------------------------
has_date = DATE_COL and DATE_COL in df.columns
max_date = df[DATE_COL].max() if has_date else None

grp = (df.groupby(ID_COL)
         .agg(Tenure_Months=("Tenure_Months","mean"),
              Offline_Spend=("Offline_Spend","sum"),
              Online_Spend=("Online_Spend","sum"),
              Quantity=("Quantity","sum"),
              Frequency=(DATE_COL if has_date else ID_COL,"count"),
              Last_Trans=(DATE_COL,"max") if has_date else (ID_COL,"count"))
         .reset_index())

grp["Total_Spend"] = grp["Offline_Spend"] + grp["Online_Spend"]
grp["Recency_Days"] = (max_date - grp["Last_Trans"]).dt.days if has_date else np.nan
grp.drop(columns=["Last_Trans"], inplace=True, errors="ignore")

feat_cols = ["Tenure_Months","Total_Spend","Quantity","Frequency"]
if has_date: feat_cols.append("Recency_Days")
X = grp[feat_cols].copy()
X["Recency_Days"] = X["Recency_Days"].fillna(X["Recency_Days"].median())
X = SimpleImputer(strategy="median").fit_transform(X)
X = StandardScaler().fit_transform(X)

# 3) K‑MEANS – en iyi k --------------------------------------------------
scores = []
for k in range(2, 13):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    sil = silhouette_score(X, km.labels_)
    scores.append((k, sil, davies_bouldin_score(X, km.labels_), calinski_harabasz_score(X, km.labels_)))
cands = [t for t in scores if t[0] >= DESIRED_MIN_CLUSTERS and t[1] >= 0.20]
best_k = max(cands, key=lambda t:(t[1],t[3]))[0] if cands else max(scores,key=lambda t:t[1])[0]
kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
grp["KM_Cluster"] = kmeans.labels_

# 4) HIERARCHICAL --------------------------------------------------------
hier = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit(X)
grp["H_Cluster"] = hier.labels_

# 5) DBSCAN – gürültü sınırlı tarama ------------------------------------
def tuned_dbscan(data, eps_grid, ms_grid, min_clusters):
    best = {"score":-1}
    for eps, ms in itertools.product(eps_grid, ms_grid):
        lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(data)
        k = len(set(lab)) - (1 if -1 in lab else 0)
        if k < min_clusters or (lab == -1).mean() > 0.35:   # koşullar
            continue
        s = silhouette_score(data, lab)
        if s > best["score"]:
            best.update(dict(score=s, eps=eps, ms=ms, labels=lab))
    return best

best_db = tuned_dbscan(X, np.arange(0.4, 2.1, 0.1), range(3,10), DESIRED_MIN_CLUSTERS)
grp["DB_Cluster"] = best_db["labels"] if best_db["score"] > 0 else DBSCAN(eps=2.0,min_samples=3).fit_predict(X)

# 6) KÜME AÇIKLAMALARI --------------------------------------------------
def segment_name(row, qspend, qqty):
    spend, qty, ten, rec = row["Total_Spend"], row["Quantity"], row["Tenure_Months"], row["Recency_Days"]
    if spend >= qspend[0.99]:           return "VIP Elite"
    if spend >= qspend[0.90]:           return "Sadık Büyük Harcayıcı"
    if ten < 12   and spend >= qspend[0.70]: return "Yeni-Yüksek Potansiyel"
    if qty >= qqty[0.90]:               return "Fiyat Duyarlı Toplayıcı"
    if rec and rec > 270:               return "Donuklaşan"
    if ten < 6:                         return "Tek-Siparişlik"
    return "İstikrarlı Orta"

q_spend = grp["Total_Spend"].quantile([0.70,0.90,0.99]).to_dict()
q_qty   = grp["Quantity"].quantile([0.90]).to_dict()

for algo, col in [("KM","KM_Cluster"),("H","H_Cluster"),("DB","DB_Cluster")]:
    grp[f"{algo}_Name"] = grp.apply(segment_name, axis=1, args=(q_spend,q_qty))

    # Küme profili + açıklamaları kaydet
    prof = (grp.groupby(col)
              .agg(Count=("CustomerID","count"),
                   Avg_Tenure=("Tenure_Months","mean"),
                   Avg_Spend=("Total_Spend","mean"),
                   Avg_Qty=("Quantity","mean"),
                   Avg_Freq=("Frequency","mean"),
                   Avg_Recency=("Recency_Days","mean"),
                   Segment_Name=(f"{algo}_Name","first"))
              .round(2))
    prof.to_csv(f"profiles_{algo.lower()}.csv")

    # Küme‑>Segment mapping JSON
    desc_map = {int(k):v for k,v in prof["Segment_Name"].to_dict().items()}
    Path(f"cluster_desc_{algo.lower()}.json").write_text(json.dumps(desc_map, indent=2, ensure_ascii=False))

# 7) 2‑D PCA GRAFİKLER --------------------------------------------------
pca = PCA(n_components=2, random_state=42).fit_transform(X)
grp["PC1"], grp["PC2"] = pca[:,0], pca[:,1]

def scatter(algo, col):
    fig = px.scatter(grp, x="PC1", y="PC2",
                     color=col, title=f"{algo} – 2‑D PCA Scatter",
                     hover_data=["KM_Name","H_Name","DB_Name","Total_Spend","Quantity"])
    fig.write_html(f"scatter_{algo.lower()}.html")
for algo,col in [("KM","KM_Cluster"),("H","H_Cluster"),("DB","DB_Cluster")]:
    scatter(algo,col)

# 8) SON KAYIT ----------------------------------------------------------
grp.to_csv("customers_with_clusters.csv", index=False)
print(f"Tamam | K‑Means k={best_k} | DBSCAN eps={best_db.get('eps')} ms={best_db.get('ms')} | "
      "Çıktılar: customers_with_clusters.csv, profiles_*, cluster_desc_*.json, scatter_*.html")
