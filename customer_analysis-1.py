#!/usr/bin/env python3
"""
Customer Type Analysis Pipeline – RFM‑score + Unique Cluster Names
------------------------------------------------------------------
Üretir:
 1. Bölge‑Ürün ısı haritası
 2. Online‑Offline yığılmış sütun
 3. Pareto grafiği
 4. RFM + KMeans (k=8)
    • R‑F‑M 1‑5 skor & segment
    • Küme açıklamaları → benzersiz (Champions‑A, Champions‑B …)
    • Silhouette skoru
    • 3B scatter
 5. Aylık satış & kupon zaman serisi
 6. Ürün ailesi treemap
Tüm çıktılar ./output/ klasöründe.
"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
from collections import defaultdict

DATA_PATH, OUTPUT_DIR = "csv/file.csv", "output"

# ───────────────────────────── I/O ─────────────────────────────
def ensure_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    date_col = "Transaction_Date" if "Transaction_Date" in df.columns else "Date"
    df["Transaction_Date"] = pd.to_datetime(df[date_col])
    return df

# ───────────────────────── RFM helpers ─────────────────────────
def rfm_table(df):
    snap = df["Transaction_Date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("Transaction_Date", lambda x: (snap - x.max()).days),
        Frequency=("Transaction_ID", "nunique"),
        Monetary=("Total_Spend", "sum"),
    )
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5,4,3,2,1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5,
                             labels=[1,2,3,4,5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1,2,3,4,5]).astype(int)
    rfm["RFM_Score"] = rfm["R_Score"].astype(str)+rfm["F_Score"].astype(str)+rfm["M_Score"].astype(str)
    return rfm

SEG_MAP = {
    "Champions":      lambda r: r.R_Score==5 and r.F_Score==5,
    "Loyal":          lambda r: r.F_Score==5 and r.R_Score>=4,
    "Potential":      lambda r: r.R_Score==5 and r.F_Score==4,
    "Recent":         lambda r: r.R_Score==5 and r.F_Score<=3,
    "Promising":      lambda r: r.R_Score==4 and r.F_Score>=3,
    "Need Attention": lambda r: r.R_Score==3 and r.F_Score==3,
    "At Risk":        lambda r: r.R_Score<=2 and r.F_Score>=3,
    "Hibernating":    lambda r: r.R_Score<=2 and r.F_Score<=2,
}
def rfm_segment(row):
    for name, rule in SEG_MAP.items():
        if rule(row):
            return name
    return "Others"

def describe_cluster(grp):
    cnt = len(grp)
    R, F, M = grp[["Recency","Frequency","Monetary"]].mean()
    return f"{cnt} müşteri (Ø R={R:.0f}g, F={F:.1f}, M={M:,.0f})"

def unique_cluster_names(base_map):
    inv = defaultdict(list)
    for cid, desc in base_map.items():
        seg = desc.split(" – ")[0]
        inv[seg].append(cid)
    new_map = base_map.copy()
    for seg, cids in inv.items():
        if len(cids) > 1:
            for i, cid in enumerate(sorted(cids), 1):
                new_label = f"{seg}-{chr(64+i)}"   # A, B, C…
                new_map[cid] = base_map[cid].replace(seg, new_label, 1)
    return new_map

def cluster_and_annotate(rfm):
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[["Recency","Frequency","Monetary"]])
    km = KMeans(n_clusters=8, random_state=42).fit(X)
    rfm["Cluster"] = km.labels_

    sil = silhouette_score(X, km.labels_)
    print(f"→ KMeans Silhouette score: {sil:.3f}")

    base_desc = {}
    for cid, grp in rfm.groupby("Cluster"):
        seg = grp["Segment"].value_counts().idxmax()
        base_desc[cid] = f"{seg} – {describe_cluster(grp)}"

    final_desc = unique_cluster_names(base_desc)

    for cid in sorted(final_desc):
        print(f"[Cluster {cid}] {final_desc[cid]}")
    with open(f"{OUTPUT_DIR}/cluster_descriptions.txt","w",encoding="utf-8") as f:
        for cid in sorted(final_desc):
            f.write(f"Cluster {cid}: {final_desc[cid]}\n")

    rfm["Cluster_Desc"] = rfm["Cluster"].map(final_desc)
    rfm.to_csv(f"{OUTPUT_DIR}/rfm_clusters.csv")
    return rfm

# ───────────────────────── 3D Scatter ─────────────────────────
def scatter_3d(rfm):
    cmap = matplotlib.colormaps.get_cmap("tab10")
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection="3d")
    for cid, grp in rfm.groupby("Cluster"):
        ax.scatter(grp.Recency, grp.Frequency, grp.Monetary,
                   s=30, alpha=0.7, color=cmap(cid%10), label=f"C{cid}")
    ax.set_xlabel("Recency"); ax.set_ylabel("Frequency"); ax.set_zlabel("Monetary")
    ax.set_title("RFM – KMeans Clusters")
    ax.legend()
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/rfm_3d_scatter.png"); plt.close()

# ───────────────────────────── MAIN ────────────────────────────
def main():
    ensure_output()
    df = load_data()
    if "Total_Spend" not in df.columns:
        df["Total_Spend"] = df["Online_Spend"] + df["Offline_Spend"]

    rfm = rfm_table(df)
    rfm["Segment"] = rfm.apply(rfm_segment, axis=1)
    rfm = cluster_and_annotate(rfm)
    scatter_3d(rfm)
    print("\nAnaliz tamamlandı → output/ klasörünü kontrol et.")

if __name__ == "__main__":
    main()
