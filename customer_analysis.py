#!/usr/bin/env python3

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from collections import defaultdict

DATA_PATH, OUTPUT_DIR = "csv/file.csv", "output"
N_CLUSTERS = 8

# ────────────────────────────────────────────────────────────
def ensure_output(): os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    date_col = "Transaction_Date" if "Transaction_Date" in df.columns else "Date"
    df["Transaction_Date"] = pd.to_datetime(df[date_col])
    return df

# ────────────────── R F M ───────────────────────────────────
def rfm_table(df):
    snap = df["Transaction_Date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency   = ("Transaction_Date", lambda x: (snap - x.max()).days),
        Frequency = ("Transaction_ID",  "nunique"),
        Monetary  = ("Total_Spend",     "sum"),
    )
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5,4,3,2,1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5,
                             labels=[1,2,3,4,5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1,2,3,4,5]).astype(int)
    return rfm

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
def rfm_segment(row):
    for name, rule in SEG_MAP.items():
        if rule(row): return name
    return "Others"

def describe_cluster(grp):
    c = len(grp)
    R,F,M = grp[["Recency","Frequency","Monetary"]].mean()
    return f"{c} müşteri (Ø R={R:.0f}g, F={F:.1f}, M={M:,.0f})"

# yinelenen segment isimlerini zenginleştir
def refine_duplicate_labels(rfm, col, base_map):
    seg_map = defaultdict(list)
    for cid, desc in base_map.items():
        seg = desc.split(" – ")[0]
        seg_map[seg].append(cid)
    new = base_map.copy()
    for seg, cids in seg_map.items():
        if len(cids) <= 1: continue
        ranked = sorted(cids, key=lambda c: rfm[rfm[col]==c]["Monetary"].mean(), reverse=True)
        suffix = ["High Value","Mid Value","Low Value","Very Low"][:len(ranked)]
        for cid, suf in zip(ranked, suffix):
            new[cid] = base_map[cid].replace(seg, f"{seg} – {suf}", 1)
    return new

# genel çıktı
def run_clustering(rfm, algo, labels):
    col = f"{algo}_Cluster"
    rfm[col] = labels
    base = {cid: f"{grp['Segment'].value_counts().idxmax()} – {describe_cluster(grp)}"
            for cid, grp in rfm.groupby(col)}
    desc = refine_duplicate_labels(rfm, col, base)
    rfm[f"{algo}_Desc"] = rfm[col].map(desc)
    rfm.to_csv(f"{OUTPUT_DIR}/rfm_{algo.lower()}.csv", index=False)
    with open(f"{OUTPUT_DIR}/cluster_descriptions_{algo.lower()}.txt","w",encoding="utf-8") as f:
        for cid in sorted(desc): f.write(f"Cluster {cid}: {desc[cid]}\n")
    return desc

def scatter_3d(rfm, algo, col):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    cmap = matplotlib.colormaps.get_cmap("tab10")
    for cid, grp in rfm.groupby(col):
        ax.scatter(grp.Recency, grp.Frequency, grp.Monetary,
                   s=25, alpha=.7, color=cmap(int(cid)%10), label=f"C{cid}")
    ax.set_xlabel("Recency"); ax.set_ylabel("Frequency"); ax.set_zlabel("Monetary")
    ax.set_title(f"RFM – {algo} Clusters"); ax.legend()
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/rfm_3d_{algo.lower()}.png"); plt.close()

# ────────────────── TOP-N PRODUCTS PER SEGMENT ──────────────────
def top_products_by_segment(df, rfm, top_n=10):
    seg_map = rfm["Segment"]
    merged  = df.merge(seg_map, left_on="CustomerID", right_index=True)

    for seg, grp in merged.groupby("Segment"):
        # ► adet bazlı TOP-N
        top = (grp["Product_Description"]
                 .value_counts()
                 .head(top_n)
                 .sort_values())          # h-bar ters sıralama için

        plt.figure(figsize=(9, 1.2*top_n + 2))
        top.plot(kind="barh", color="steelblue", zorder=3)
        plt.grid(axis="x", linestyle="--", alpha=.4, zorder=0)
        plt.title(f"Top {top_n} Products – {seg}")
        plt.xlabel("Sales Unit"); plt.ylabel("")
        plt.tight_layout()

        fname = f"{OUTPUT_DIR}/top{top_n}_{seg.replace(' ','_').lower()}.png"
        plt.savefig(fname, dpi=120)
        plt.close()
        print(" »", fname)


# ─────────────────── MAIN ────────────────────
def main():
    ensure_output()
    df = load_data()
    if "Total_Spend" not in df.columns:
        df["Total_Spend"] = df["Online_Spend"] + df["Offline_Spend"]

    rfm = rfm_table(df)
    rfm["Segment"] = rfm.apply(rfm_segment, axis=1)
    X = StandardScaler().fit_transform(rfm[["Recency","Frequency","Monetary"]])

    # K-Means
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42).fit(X)
    print(f"KMeans silhouette: {silhouette_score(X, km.labels_):.3f}")
    run_clustering(rfm, "KMeans", km.labels_); scatter_3d(rfm,"KMeans","KMeans_Cluster")

    # Hierarchical
    hier = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward").fit(X)
    print(f"Hierarchical silhouette: {silhouette_score(X, hier.labels_):.3f}")
    run_clustering(rfm,"Hierarchical",hier.labels_); scatter_3d(rfm,"Hierarchical","Hierarchical_Cluster")

    # DBSCAN – grid-search
    best={"score":-1}
    for eps in np.arange(0.3,3.05,0.1):
        for ms in range(3,9):
            lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
            k=len(set(lab))-(-1 in lab); noise=(lab==-1).mean()
            if k<3 or noise>0.4: continue
            try: s=silhouette_score(X,lab)
            except: continue
            if s>best["score"]: best=dict(score=s,eps=eps,ms=ms,labels=lab)
    if best["score"]>0:
        print(f"DBSCAN → eps={best['eps']:.1f}, ms={best['ms']} | Silhouette={best['score']:.3f}")
        run_clustering(rfm,"DBSCAN",best["labels"]); scatter_3d(rfm,"DBSCAN","DBSCAN_Cluster")
    else:
        print("DBSCAN anlamlı küme üretemedi; atlandı.")

    # Top-N ürünler
    top_products_by_segment(df, rfm, top_n=10)

    print("✓ Analiz tamamlandı  »  output/")

if __name__ == "__main__":
    main()
