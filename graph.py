#!/usr/bin/env python3

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

# ── 5 Monthly revenue (line) & coupon revenue (bar) ───────────────
try:
    # Toplam ciro (tüm satışlar)
    rev_month = df.groupby("Month_Year")["Total_Spend"].sum()

    # Kupon ciro (Coupon_Status == "Used")
    coup_month = (df[df["Coupon_Status"].eq("Used")]
                    .groupby("Month_Year")["Total_Spend"]
                    .sum())

    # 1️⃣ Dizide "NaT" yazılı etiketi veya gerçek NaT'i temizle
    for s in (rev_month, coup_month):
        s.drop(labels=["NaT"], errors="ignore", inplace=True)     # metin olarak "NaT"
        s.dropna(inplace=True)                                    # gerçek NaT/NaN

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Toplam ciro: line
    rev_month.plot(marker="o",
                   color="steelblue",
                   ax=ax1,
                   label="Total Revenue")
    ax1.set_ylabel("Revenue ($)")

    # Kupon ciro: bar
    ax2 = ax1.twinx()
    ax2.bar(coup_month.index,
            coup_month.values,
            color="seagreen",
            alpha=0.6,
            label="Coupon Revenue (Used)")
    ax2.set_ylabel("Coupon Revenue ($)")

    ax1.set_title("Monthly Revenue & Coupon Revenue")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.xticks(rotation=45)

    save_fig("05_month_ts.png")
except Exception as e:
    print(f"[skip] Monthly TS → {e}")

# ── 6 Treemap (Plotly) + otomatik bar fallback ────────────────────
import plotly.express as px
import os

def treemap_product_family(df, output_dir=MAP_DIR):
    fam = (df.groupby('Product_Category', as_index=False)['Total_Spend']
             .sum()
             .sort_values('Total_Spend', ascending=False))

    # Fallback koşulu
    if len(fam) < 2 or fam['Total_Spend'].sum() == 0:
        fam.set_index('Product_Category')['Total_Spend'] \
           .plot(kind='bar', figsize=(7, 4), color='steelblue')
        plt.ylabel("Revenue ($)")
        plt.title("Product Revenue by Category (fallback)")
        save_fig("06_treemap_fallback.png")
        plt.close()
        print("[info] Treemap yerine bar grafiği kaydedildi → 06_treemap_fallback.png")
        return

    # Plotly treemap
    fig = px.treemap(
        fam,
        path=['Product_Category'],
        values='Total_Spend',
        title='Product Family Revenue Share'
    )

    html_path = os.path.join(output_dir, '06_treemap.html')
    fig.write_html(html_path)

    # (İsteğe bağlı) statik PNG – kaleido kuruluysa
    try:
        fig.write_image(os.path.join(output_dir, '06_treemap.png'))
    except Exception:
        pass

    print(f"[ok] Treemap kaydedildi → {html_path}")

# ➤ Ana akışta doğrudan çağırın:
try:
    treemap_product_family(df)
except Exception as e:
    print(f"[skip] Treemap → {e}")


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
    if "Coupon_Status" not in df.columns:
        raise KeyError("Coupon_Status column missing.")

    # 1️⃣ Kupon durumunu normalize et
    status = (df["Coupon_Status"]
                .fillna("Not Used")            # eksikler → Not Used
                .replace({"": "Not Used"}))    # boş string varsa

    # 'Used', 'Clicked', diğer → 'Not Used'
    status = status.where(status.isin(["Used", "Clicked"]),
                          other="Not Used")

    gtab = pd.crosstab(df["Gender"], status)

    gtab = gtab[["Not Used", "Clicked", "Used"]]

    gtab.plot(kind="bar",
              stacked=True,
              figsize=(8, 4))

    plt.ylabel("Order Count")
    plt.title("Coupon Usage by Gender")
    plt.legend(["Not Used", "Clicked", "Used"])
    plt.tight_layout()
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

# ── 15  Top-20 products – Online vs Offline revenue ──
try:
    need = {"Product_Description", "Online_Spend", "Offline_Spend"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {', '.join(missing)}")

    # ➊  Ürün bazında toplam gelirler
    prod_rev = (df.groupby("Product_Description")[["Online_Spend","Offline_Spend"]]
                  .sum()
                  .assign(Total=lambda x: x["Online_Spend"] + x["Offline_Spend"])
                  .sort_values("Total", ascending=False)
                  .head(20)
                  .iloc[::-1])      # barh için ters çevir → en yüksek üstte

    # ➋  Grafik
    ax = prod_rev[["Online_Spend","Offline_Spend"]].plot(
            kind="barh", figsize=(10, 6), color=["steelblue","darkorange"])
    plt.xlabel("Revenue (₺)")
    plt.title("Top-20 Products – Online vs Offline Revenue")
    plt.legend(["Online","Offline"])
    plt.tight_layout()
    save_fig("15_top_products_online_offline.png")
    print("[info] 15_top_products_online_offline.png created.")

except Exception as e:
    print(f"[skip] Top products chart → {e}")

# ── 16  Top-5 products per Location (facet grid) ─────
try:
    cols_needed = {"Location", "Product_Description", "Quantity"}
    missing = cols_needed - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {', '.join(missing)}")

    # ➊  Konum listesi
    locs = df["Location"].dropna().unique()
    n_loc = len(locs)
    if n_loc == 0:
        raise ValueError("No locations found.")

    # ➋  Izgara boyutu (en fazla 3 sütun)
    ncols = 3
    nrows = int(np.ceil(n_loc / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols*5, nrows*4),
                             squeeze=False)

    for idx, loc in enumerate(sorted(locs)):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        # En çok satılan 5 ürün
        top5 = (df[df["Location"] == loc]
                  .groupby("Product_Description")["Quantity"]
                  .sum()
                  .sort_values(ascending=False)
                  .head(5)
                  .iloc[::-1])          # barh için ters çevir

        top5.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title(f"{loc} – Top-5 Products")
        ax.set_xlabel("Units Sold")
        ax.set_ylabel("")

    # Boş alt grafik varsa temizle
    for j in range(idx+1, nrows*ncols):
        r, c = divmod(j, ncols)
        fig.delaxes(axes[r][c])

    fig.suptitle("Top-5 Products per Location", fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig("16_top5_products_per_location.png")
    print("[info] 16_top5_products_per_location.png created.")

except Exception as e:
    print(f"[skip] Top-5 products per location → {e}")

# ── 17  Top-20 products for each Gender ───────────────
# ── 17  Top-20 products per Gender – Treemap & HTML ──
try:
    need = {"Gender", "Product_Description", "Quantity"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"Missing columns: {', '.join(miss)}")

    import plotly.express as px
    import matplotlib.pyplot as plt
    import squarify

    # ---------------- Prepare data ----------------
    top20_df = (df.groupby(["Gender", "Product_Description"])["Quantity"]
                  .sum()
                  .sort_values(ascending=False)
                  .groupby(level=0)
                  .head(20)                         # top-20 within each gender
                  .reset_index())

    # 1) Matplotlib PNG treemap (all genders in one figure)
    genders = top20_df["Gender"].unique()
    ncols = 2
    nrows = int(np.ceil(len(genders) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols*6, nrows*5), squeeze=False)

    for idx, g in enumerate(sorted(genders)):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        sub = top20_df[top20_df["Gender"] == g]
        sizes = sub["Quantity"].values
        labels = [f"{p}\n{sub['Quantity'].iloc[i]}" for i, p in enumerate(sub["Product_Description"])]
        squarify.plot(sizes=sizes, label=labels, ax=ax, alpha=.8)
        ax.axis("off")
        ax.set_title(f"{g} – Top-20 Products")

    # boş eksenleri temizle
    for j in range(len(genders), nrows*ncols):
        r, c = divmod(j, ncols); fig.delaxes(axes[r][c])

    plt.tight_layout()
    save_fig("17_top20_products_gender_treemap.png")
    print("[info] 17_top20_products_gender_treemap.png created.")

    # 2) Plotly HTML treemap (interactive)
    fig_html = px.treemap(top20_df,
                          path=["Gender", "Product_Description"],
                          values="Quantity",
                          color="Gender",
                          title="Top-20 Products per Gender – Interactive Treemap")
    fig_html.write_html(f"{MAP_DIR}/17_top_20_treemap_gender.html")
    print("[info] Interactive treemap saved → output/treemap_gender.html")

except ImportError:
    print("[skip] Plotly not installed – install plotly to enable interactive treemap.")
except Exception as e:
    print(f"[skip] Gender treemap → {e}")

# ── 18  Top-10 products by total revenue ──────────────
try:
    need_cols = {"Product_Description", "Total_Spend"}
    absent = need_cols - set(df.columns)
    if absent:
        raise KeyError(f"Missing columns: {', '.join(absent)}")

    # En yüksek gelire göre ilk 10 ürün
    top10_rev = (df.groupby("Product_Description")["Total_Spend"]
                   .sum()
                   .sort_values(ascending=False)
                   .head(10)
                   .iloc[::-1])    # barh için ters → en yüksek yukarı

    # Grafik
    plt.figure(figsize=(8, 5))
    top10_rev.plot(kind="barh", color="forestgreen")
    plt.xlabel("Revenue ($)")
    plt.title("Top-10 Products by Revenue")
    for idx, val in enumerate(top10_rev.values):
        plt.text(val, idx, f"{val:,.0f}", va="center", ha="left")
    plt.tight_layout()
    save_fig("18_top10_products_revenue.png")
    print("[info] 18_top10_products_revenue.png created.")

except Exception as e:
    print(f"[skip] Top-10 revenue chart → {e}")


print("✔ All available charts saved to output/plots/")
