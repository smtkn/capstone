#!/usr/bin/env python3
"""
Customer Insights – Presentation Pack (v5)
──────────────────────────────────────────
Creates PNG files in output/plots/ and, if coordinates exist,
an interactive HTML map in output/map/.

Charts (filenames) ─────────────────────────────────────
01_region_heatmap.png     Region × Product heat-map
02_channel_stack.png      Online vs Offline stacked bar
03_pareto.png             80/20 revenue concentration
04_rfm_3d.png             RFM 3-D K-Means scatter
05_month_ts.png           Monthly revenue & coupon line/area
06_treemap.png            Product-family treemap
   ↳ fallback → 06_treemap_fallback.png (bar)
07_sales_map.html         Interactive bubble map  (HTML)
   ↳ fallback → 07_region_bar.png        (bar)
08_gender_coupon.png      Coupon usage by gender
09_coupon_status.png      Coupon Status distribution (pie/bar)
"""

# ── imports ───────────────────────────────────────────
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib
from mpl_toolkits.mplot3d import Axes3D    # noqa
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import squarify
try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# ── paths & tweaks ────────────────────────────────────
DATA_PATH  = "csv/file.csv"        # update if needed
OUT_DIR    = "output"
PLOT_DIR   = f"{OUT_DIR}/plots";  MAP_DIR = f"{OUT_DIR}/map"
N_CLUSTERS = 8

os.makedirs(PLOT_DIR, exist_ok=True); os.makedirs(MAP_DIR, exist_ok=True)

def save_fig(fname):
    plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/{fname}", dpi=300,
                                    bbox_inches="tight"); plt.close()

# ── load data ─────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["Transaction_Date", "Date"])

# Ensure Total_Spend
if "Total_Spend" not in df.columns:
    spend_cols = [c for c in df.columns if c.lower().endswith("_spend")]
    if not spend_cols:
        raise ValueError("No Total_Spend or *_Spend columns found.")
    df["Total_Spend"] = df[spend_cols].sum(axis=1)

df["Month_Year"] = df["Transaction_Date"].dt.to_period("M").astype(str)

# ── 1 Region × Product heat-map ──────────────────────
try:
    heat = (df.groupby(["Location", "Product_Category"])["Quantity"]
              .sum().unstack(fill_value=0))
    plt.figure(figsize=(10, 6))
    plt.imshow(heat, aspect="auto", cmap="YlOrRd")
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=90)
    plt.yticks(range(len(heat.index)),   heat.index)
    plt.colorbar(label="Units Sold")
    plt.title("Region vs Product Category – Heatmap")
    save_fig("01_region_heatmap.png")
except Exception as e:
    print(f"[skip] Heat-map → {e}")

# ── 2 Online vs Offline stacked bar ──────────────────
try:
    channels = ["Online_Spend", "Offline_Spend"]
    if not set(channels).issubset(df.columns):
        raise KeyError("Online_Spend / Offline_Spend missing.")
    ch = (df.groupby("Location")[channels]
            .sum().sort_values("Online_Spend", ascending=False))
    ch.plot(kind="bar", stacked=True, figsize=(10, 5))
    plt.ylabel("Revenue ($)")
    plt.title("Online vs Offline Revenue by Region")
    plt.legend(["Online", "Offline"])
    save_fig("02_channel_stack.png")
except Exception as e:
    print(f"[skip] Channel stacked bar → {e}")

# ── 3 Pareto 80/20 ───────────────────────────────────
try:
    cust = (df.groupby("CustomerID")["Total_Spend"]
              .sum().sort_values(ascending=False))
    cum = cust.cumsum() / cust.sum() * 100
    fig, ax1 = plt.subplots(figsize=(10, 5))
    cust.plot(kind="bar", ax=ax1, color="steelblue")
    ax1.set_ylabel("Revenue ($)"); ax1.set_xticks([])
    ax2 = ax1.twinx()
    ax2.plot(cum.values, color="crimson", marker="D")
    ax2.axhline(80, color="gray", ls="--")
    ax2.set_ylabel("Cumulative %")
    ax1.set_title("Pareto Analysis – Cumulative Share of Revenue")
    ax2.text(len(cum)*0.98, 82, "80 %", ha="right", va="bottom")
    save_fig("03_pareto.png")
except Exception as e:
    print(f"[skip] Pareto → {e}")

# ── 4 RFM 3-D scatter ────────────────────────────────
try:
    snap = df["Transaction_Date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency   = ("Transaction_Date", lambda x: (snap - x.max()).days),
        Frequency = ("Transaction_ID",  "nunique"),
        Monetary  = ("Total_Spend",     "sum"))
    X = StandardScaler().fit_transform(rfm)
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42).fit(X)
    rfm["Cluster"] = km.labels_
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    cmap = matplotlib.colormaps.get_cmap("tab10")
    for cid, grp in rfm.groupby("Cluster"):
        ax.scatter(grp.Recency, grp.Frequency, grp.Monetary,
                   s=20, color=cmap(cid % 10), label=f"C{cid}")
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Frequency (orders)")
    ax.set_zlabel("Monetary ($)")
    ax.set_title("RFM Segmentation – 3-D K-Means")
    ax.legend()
    save_fig("04_rfm_3d.png")
    print(f"[info] Silhouette: {silhouette_score(X, km.labels_):.3f}")
except Exception as e:
    print(f"[skip] RFM 3-D → {e}")

# ── 5 Monthly revenue & coupon line ──────────────────
try:
    rev_month  = df.groupby("Month_Year")["Total_Spend"].sum()
    coup_month = (df[df["Coupon_Code"].notna()]
                    .groupby("Month_Year")["Total_Spend"].sum())
    fig, ax1 = plt.subplots(figsize=(10, 5))
    rev_month.plot(marker="o", ax=ax1, label="Total Revenue")
    ax1.set_ylabel("Revenue ($)")
    ax2 = ax1.twinx()
    coup_month.plot(marker="s", linestyle="--",
                    color="green", ax=ax2, label="Coupon Revenue")
    ax2.set_ylabel("Coupon Revenue ($)")
    ax1.set_title("Monthly Revenue & Coupon Revenue")
    ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
    plt.xticks(rotation=45)
    save_fig("05_month_ts.png")
except Exception as e:
    print(f"[skip] Monthly TS → {e}")

# ── 6 Treemap (with fallback) ─────────────────────────
try:
    fam_col = next(c for c in ["Product_Family", "Product_Category", "Category"]
                   if c in df.columns)
    fam = (df.groupby(fam_col)["Total_Spend"]
             .sum().sort_values(ascending=False))
    if len(fam) >= 2 and fam.sum() > 0:
        labels = [f"{k}\n{v/1_000:.1f} k$" for k, v in fam.items()]
        plt.figure(figsize=(12, 7))
        squarify.plot(sizes=fam.values, label=labels, alpha=0.85)
        plt.axis("off"); plt.title("Product-Family Revenue Share (Treemap)")
        save_fig("06_treemap.png")
    else:
        raise ValueError("One category or zero values – bar fallback.")
except Exception as e:
    print(f"[fallback] Treemap → {e}")
    try:
        fam.plot(kind="bar", figsize=(7, 4), color="steelblue")
        plt.ylabel("Revenue ($)")
        plt.title("Product Revenue by Category (fallback)")
        save_fig("06_treemap_fallback.png")
    except Exception as e2:
        print(f"[skip] Treemap fallback → {e2}")

# ── 7 Bubble map (with fallback bar) ──────────────────
map_done = False
if {"Latitude", "Longitude"}.issubset(df.columns) and HAS_FOLIUM:
    try:
        df["Latitude"]  = pd.to_numeric(df["Latitude"],  errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
        loc_rev = (df.dropna(subset=["Latitude", "Longitude"])
                     .groupby(["Location", "Latitude", "Longitude"])["Total_Spend"]
                     .sum().reset_index())
        if not loc_rev.empty:
            m = folium.Map(location=[loc_rev["Latitude"].mean(),
                                     loc_rev["Longitude"].mean()],
                           zoom_start=5, tiles="cartodbpositron")
            for _, r in loc_rev.iterrows():
                folium.CircleMarker(
                    location=[r.Latitude, r.Longitude],
                    radius=max(3, np.log10(r.Total_Spend + 1)*3),
                    popup=f"{r.Location}: {r.Total_Spend:,.0f} $",
                    color="crimson", fill=True, fill_opacity=0.65
                ).add_to(m)
            m.save(f"{MAP_DIR}/07_sales_map.html")
            print("[info] Interactive map saved → output/map/07_sales_map.html")
            map_done = True
    except Exception as e:
        print(f"[map error] → {e}")

if not map_done:
    print("[fallback] Using region bar instead of map.")
    try:
        reg = (df.groupby("Location")["Total_Spend"]
                 .sum().sort_values())
        reg.plot(kind="barh", figsize=(8,5), color="darkorange")
        plt.xlabel("Revenue ($)")
        plt.title("Revenue by Region (no coordinates)")
        save_fig("07_region_bar.png")
    except Exception as e:
        print(f"[skip] Map fallback → {e}")

# ── 8 Coupon usage by gender ──────────────────────────
try:
    if "Gender" not in df.columns:
        raise KeyError("Gender column missing.")
    flag = df["Coupon_Code"].notna().rename({True:"With Coupon",
                                             False:"No Coupon"})
    gtab = pd.crosstab(df["Gender"], flag)
    gtab.plot(kind="bar", stacked=True, figsize=(8, 4))
    plt.ylabel("Order Count")
    plt.title("Coupon Usage by Gender")
    plt.legend(["No Coupon", "With Coupon"])
    save_fig("08_gender_coupon.png")
except Exception as e:
    print(f"[skip] Gender-Coupon → {e}")

# ── 9 Coupon Status distribution ──────────────────────
try:
    # ➊ Pick the first matching status column
    status_col = next(c for c in ["Coupon_Status", "CouponState", "Coupon_Flag"]
                      if c in df.columns)

    # ➋  Harmonise spelling / spaces / case  ────────────
    clean = (df[status_col]
             .astype(str)
             .str.strip()          # trim spaces
             .str.lower())         # ignore case

    # ➌  Map synonyms to a single label
    synonym_map = {
        "notused": "Not used",
        "not used": "Not used",
        "unused": "Not used",
        "": "Not used",
        "nan": "Not used",

        "clicked": "Clicked",
        "click": "Clicked",

        "used": "Used",
        "applied": "Used"
    }
    unified = clean.map(lambda x: synonym_map.get(x, x.title()))

    counts = unified.value_counts()
    if counts.empty or counts.sum() == 0:
        raise ValueError("Coupon status column is empty.")

    # ➍ Plot – pie if ≤5 categories, else bar
    if len(counts) > 5:
        counts.plot(kind="bar", figsize=(8, 4), color="mediumpurple")
        plt.ylabel("Order Count"); plt.title("Coupon Status Distribution")
    else:
        plt.figure(figsize=(6, 6))
        plt.pie(counts, labels=counts.index,
                autopct="%1.1f%%", startangle=140)
        plt.title("Coupon Status Distribution")
    save_fig("09_coupon_status.png")

except StopIteration:
    print("[skip] Coupon Status – no suitable column found.")
except Exception as e:
    print(f"[skip] Coupon Status → {e}")

# ── 10  GST descriptive statistics (bar) ─────────────
try:
    if "GST" in df.columns:
        gst_stats = {
            "Min":    df["GST"].min(),
            "Median": df["GST"].median(),
            "Mean":   df["GST"].mean(),
            "Max":    df["GST"].max()
        }
        plt.figure(figsize=(7, 4))
        plt.barh(list(gst_stats.keys()), list(gst_stats.values()),
                 color="salmon")
        plt.xlabel("Amount ($)")
        plt.title("GST – Descriptive Statistics")
        for i, v in enumerate(gst_stats.values()):
            plt.text(v, i, f"{v:,.2f}", va="center", ha="left")
        save_fig("10_gst_stats.png")
        print("[info] 10_gst_stats.png created.")
    else:
        raise KeyError("GST column missing.")
except Exception as e:
    print(f"[skip] GST stats → {e}")

# ── 11  Delivery_Charges descriptive statistics ─────
try:
    if "Delivery_Charges" in df.columns:
        ship_stats = {
            "Min":    df["Delivery_Charges"].min(),
            "Median": df["Delivery_Charges"].median(),
            "Mean":   df["Delivery_Charges"].mean(),
            "Max":    df["Delivery_Charges"].max()
        }
        plt.figure(figsize=(7, 4))
        plt.barh(list(ship_stats.keys()), list(ship_stats.values()),
                 color="teal")
        plt.xlabel("Amount ($)")
        plt.title("Delivery Charges – Descriptive Statistics")
        for i, v in enumerate(ship_stats.values()):
            plt.text(v, i, f"{v:,.2f}", va="center", ha="left")
        save_fig("11_delivery_stats.png")
        print("[info] 11_delivery_stats.png created.")
    else:
        raise KeyError("Delivery_Charges column missing.")
except Exception as e:
    print(f"[skip] Delivery stats → {e}")

# ── 12  GST_Amount distribution (histogram) ──────────
try:
    # Gerekli sütunlar
    on_col, off_col, gst_rate_col = "Online_Spend", "Offline_Spend", "GST"

    # Kontrol
    for col in [on_col, off_col, gst_rate_col]:
        if col not in df.columns:
            raise KeyError(f"{col} column missing.")

    # GST oranını yüzde ise 0.18 formatına çevir
    if df[gst_rate_col].median() > 1:
        df[gst_rate_col] = df[gst_rate_col] / 100.0

    # Sütun yoksa tekrar hesaplar, varsa üzerine yazmaz
    if "GST_Amount" not in df.columns:
        df["GST_Amount"] = (df[on_col] + df[off_col]) * df[gst_rate_col]

    gst_series = df["GST_Amount"].dropna()
    if gst_series.empty or gst_series.sum() == 0:
        raise ValueError("GST_Amount column empty or zero.")

    # Histogram + KDE
    plt.figure(figsize=(8, 4))
    gst_series.plot(kind="hist", bins=30, density=True, alpha=0.6,
                    color="salmon", label="Histogram")
    gst_series.plot(kind="kde", color="darkred", label="KDE")
    plt.xlabel("GST Amount per Transaction (₺)")
    plt.title("Distribution of GST Amounts")
    plt.legend()
    save_fig("12_gst_amount_dist.png")
    print("[info] 12_gst_amount_dist.png created.")

except Exception as e:
    print(f"[skip] GST_Amount distribution → {e}")

# ── 13  Weekday vs Weekend – Online vs Offline ───────
try:
    required = {"Online_Spend", "Offline_Spend", "Transaction_Date"}
    if required.issubset(df.columns):

        # Hafta tipi (0‒4 = Weekday, 5‒6 = Weekend)
        df["Weektype"] = df["Transaction_Date"].dt.dayofweek \
                            .apply(lambda d: "Weekend" if d >= 5 else "Weekday")

        # Toplam harcamalar
        channel = (df.groupby("Weektype")[["Online_Spend", "Offline_Spend"]]
                     .sum().loc[["Weekday", "Weekend"]])

        # ── Grafik ──
        channel.plot(kind="bar", stacked=False, figsize=(8, 5),
                     color=["steelblue", "darkorange"])
        plt.ylabel("Revenue (₺)")
        plt.title("Online vs Offline Spending: Weekday vs Weekend")
        plt.xticks(rotation=0)
        plt.legend(["Online", "Offline"])
        save_fig("13_weektype_channel.png")
        print("[info] 13_weektype_channel.png created.")

        # ── Özet tablo (CSV) ──
        summary = channel.assign(
            Total=lambda x: x["Online_Spend"] + x["Offline_Spend"],
            Online_Ratio=lambda x: (x["Online_Spend"] / x["Total"]).round(3),
            Offline_Ratio=lambda x: (x["Offline_Spend"] / x["Total"]).round(3)
        )
        summary.to_csv(f"{OUT_DIR}/weekday_weekend_online_offline.csv")
        print("[info] weekday_weekend_online_offline.csv saved.")

    else:
        raise KeyError(f"Missing columns: {required - set(df.columns)}")

except Exception as e:
    print(f"[skip] Weektype channel → {e}")

# ── 14  Online vs Offline by each weekday ─────────────
try:
    req = {"Online_Spend", "Offline_Spend", "Transaction_Date"}
    if req.issubset(df.columns):

        # Gün adı (İngilizce) ve sıralama listesi
        df["Weekday"] = df["Transaction_Date"].dt.day_name()
        order = ["Monday","Tuesday","Wednesday","Thursday",
                 "Friday","Saturday","Sunday"]

        # Toplam harcamalar
        week_df = (df.groupby("Weekday")[["Online_Spend","Offline_Spend"]]
                     .sum().reindex(order))

        # ── Grafik ──
        ax = week_df.plot(kind="bar", stacked=False, figsize=(10,5),
                          color=["steelblue","darkorange"])
        plt.ylabel("Revenue (₺)")
        plt.title("Online vs Offline Spending by Weekday")
        plt.xticks(rotation=15)
        plt.legend(["Online","Offline"])
        save_fig("14_weekday_channel.png")
        print("[info] 14_weekday_channel.png created.")

        # ── Özet tablo ──
        weekday_summary = week_df.assign(
            Total=lambda x: x["Online_Spend"] + x["Offline_Spend"],
            Online_Ratio=lambda x: (x["Online_Spend"]/x["Total"]).round(3),
            Offline_Ratio=lambda x: (x["Offline_Spend"]/x["Total"]).round(3)
        )
        weekday_summary.to_csv(f"{OUT_DIR}/weekday_online_offline.csv")
        print("[info] weekday_online_offline.csv saved.")

    else:
        raise KeyError(f"Missing columns: {req - set(df.columns)}")

except Exception as e:
    print(f"[skip] Weekday channel → {e}")

print("✔ All available charts saved to output/plots/")
