#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customer Type Analysis – Full Pipeline
======================================

Çıktılar  : output/
Gereksinim: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, squarify, folium
Kurulum   : pip install -U pandas numpy matplotlib seaborn scikit-learn scipy squarify folium
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import folium

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3‑D)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# -------------  KONFİGÜRASYON ------------------
CSV_PATH      = "csv/file.csv"
DATE_COL      = "InvoiceDate"       # tarih/saat bilgisi
CUSTOMER_COL  = "CustomerID"
REGION_COL    = "Region"            # bölge
CHANNEL_COL   = "Channel"           # Online / Offline
PRODUCT_COL   = "ProductCategory"   # ürün kategorisi
FAMILY_COL    = "ProductFamily"     # ürün ailesi
AMOUNT_COL    = "Amount"            # satış tutarı
COUPON_COL    = "CouponUsed"        # 1/0 veya True/False
LAT_COL       = "Latitude"          # opsiyonel
LON_COL       = "Longitude"         # opsiyonel
# -----------------------------------------------


def prepare_output_dir():
    os.makedirs("output", exist_ok=True)


def load_data():
    df = pd.read_csv(CSV_PATH, parse_dates=[DATE_COL])
    # eksik değer temizlik örneği
    df = df.dropna(subset=[CUSTOMER_COL, AMOUNT_COL, DATE_COL])
    return df


# 1. Bölge‑Ürün Isı Haritası ----------------------
def plot_region_product_heatmap(df):
    pivot = (df
             .groupby([REGION_COL, PRODUCT_COL])[AMOUNT_COL]
             .sum()
             .unstack(fill_value=0))
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")
    plt.title("Bölge ‑ Ürün Kategorisi Isı Haritası")
    plt.ylabel("Bölge"), plt.xlabel("Ürün Kategorisi")
    plt.tight_layout()
    plt.savefig("output/heatmap_region_product.png")
    plt.close()


# 2. Satış Kanalları Yığılmış Sütun ---------------
def plot_channel_stacked_bar(df):
    data = (df
            .groupby([REGION_COL, CHANNEL_COL])[AMOUNT_COL]
            .sum()
            .unstack(fill_value=0))
    data.plot(kind="bar", stacked=True, figsize=(10, 6))
    plt.title("Online vs Offline Satış Tutarı (Bölgelere Göre)")
    plt.ylabel("Satış Tutarı")
    plt.tight_layout()
    plt.savefig("output/stacked_channel.png")
    plt.close()


# 3. Pareto (80/20) -------------------------------
def plot_pareto(df):
    sales = (df.groupby(CUSTOMER_COL)[AMOUNT_COL]
             .sum()
             .sort_values(ascending=False))
    cum_pct = sales.cumsum() / sales.sum() * 100
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(range(len(sales)), sales.values)
    ax1.set_ylabel("Satış Tutarı")
    ax2 = ax1.twinx()
    ax2.plot(range(len(sales)), cum_pct, color="red")
    ax2.set_ylabel("Kümülatif %")
    ax2.axhline(80, color="grey", ls="--")
    ax1.set_title("Pareto Analizi – Satışların Yoğunlaştığı Müşteriler")
    plt.tight_layout()
    plt.savefig("output/pareto_customers.png")
    plt.close()


# 4. RFM Kümeleme + 3B Dağılım --------------------
def rfm_clustering(df, n_clusters=4):
    snapshot_date = df[DATE_COL].max() + pd.Timedelta(days=1)
    rfm = df.groupby(CUSTOMER_COL).agg({
        DATE_COL: lambda x: (snapshot_date - x.max()).days,
        AMOUNT_COL: ["count", "sum"]
    })
    rfm.columns = ["Recency", "Frequency", "Monetary"]
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(rfm_scaled)
    rfm["Cluster"] = kmeans.labels_

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(rfm["Recency"], rfm["Frequency"], rfm["Monetary"],
                    c=rfm["Cluster"], cmap="rainbow", alpha=0.6)
    ax.set_xlabel("Recency"), ax.set_ylabel("Frequency"), ax.set_zlabel("Monetary")
    plt.title("RFM Kümeleme – 3B Dağılım")
    plt.tight_layout()
    plt.savefig("output/rfm_3dscatter.png")
    plt.close()
    return rfm


# 5. Zaman Serisi Analizi -------------------------
def plot_time_series(df):
    df_month = (df
                .set_index(DATE_COL)
                .resample("M")
                .agg({"Amount": "sum", COUPON_COL: "sum"}))
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df_month.index, df_month["Amount"], marker="o")
    ax1.set_ylabel("Aylık Satış")
    ax2 = ax1.twinx()
    ax2.bar(df_month.index, df_month[COUPON_COL], alpha=0.3)
    ax2.set_ylabel("Kupon Kullanımı")
    plt.title("Aylık Satış & Kupon Kullanımı")
    plt.tight_layout()
    plt.savefig("output/time_series_sales_coupon.png")
    plt.close()


# 6. Ürün Ailesi Treemap --------------------------
def plot_treemap(df):
    family_sales = (df
                    .groupby(FAMILY_COL)[AMOUNT_COL]
                    .sum()
                    .sort_values(ascending=False))
    plt.figure(figsize=(12, 8))
    squarify.plot(sizes=family_sales.values,
                  label=[f"{k}\n{v:,.0f}" for k, v in family_sales.items()],
                  alpha=.8)
    plt.axis("off")
    plt.title("Ürün Ailesi – Ciro Payı Treemap")
    plt.savefig("output/treemap_family.png")
    plt.close()


# 7. Opsiyonel Harita -----------------------------
def plot_map(df):
    if LAT_COL not in df.columns or LON_COL not in df.columns:
        print("Harita için koordinat bulunamadı, atlanıyor.")
        return
    map_df = (df
              .groupby([REGION_COL, LAT_COL, LON_COL])[AMOUNT_COL]
              .sum()
              .reset_index())

    m = folium.Map(location=[map_df[LAT_COL].mean(), map_df[LON_COL].mean()],
                   zoom_start=5)

    for _, row in map_df.iterrows():
        folium.CircleMarker(
            location=[row[LAT_COL], row[LON_COL]],
            radius=row[AMOUNT_COL] ** 0.5 / 50,  # ölçekleme
            popup=f"{row[REGION_COL]}: {row[AMOUNT_COL]:,.0f}",
            color="blue", fill=True, fill_opacity=0.6
        ).add_to(m)

    m.save("output/region_sales_map.html")
    print("Harita kaydedildi → output/region_sales_map.html")


# -------------  CLUSTERING EKSTRA ----------------
def additional_clustering(df):
    feats = ["Recency", "Frequency", "Monetary"]
    rfm = rfm_clustering(df, n_clusters=4)  # kmeans sonucu zaten geliyor
    X = StandardScaler().fit_transform(rfm[feats])

    # Hiyerarşik
    Z = linkage(X, method="ward")
    plt.figure(figsize=(10, 6))
    dendrogram(Z, truncate_mode="level", p=5)
    plt.title("Hiyerarşik Kümeleme – Dendrogram")
    plt.savefig("output/hierarchical_dendrogram.png")
    plt.close()
    rfm["HCluster"] = fcluster(Z, t=4, criterion="maxclust") - 1

    # DBSCAN
    db = DBSCAN(eps=0.8, min_samples=5).fit(X)
    rfm["DBSCAN"] = db.labels_
    rfm.to_csv("output/rfm_with_clusters.csv")
    return rfm


# --------------------  MAIN  ---------------------
def main():
    prepare_output_dir()
    df = load_data()
    plot_region_product_heatmap(df)
    plot_channel_stacked_bar(df)
    plot_pareto(df)
    rfm_clustering(df)
    plot_time_series(df)
    plot_treemap(df)
    plot_map(df)
    additional_clustering(df)
    print("Tüm analizler tamamlandı → output/ klasörünü kontrol edin.")


if __name__ == "__main__":
    main()
