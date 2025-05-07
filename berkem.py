# -----------------------------------------------
# Customer & Sales Analytics Master Script (Python)
# Generates insightful visualisations + clustering analyses
# Data source: file.csv (columns auto‑mapped to script variables)
# Author: ChatGPT
# -----------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.io as pio
from matplotlib.ticker import PercentFormatter
import os

# 0 — PARAMETERS ---------------------------------------------------------------
DATA_PATH = 'csv/file.csv'          # adjust if file in different folder
OUT_DIR   = 'outputs'           # all PNG / HTML will be saved here
os.makedirs(OUT_DIR, exist_ok=True)

# 1 — LOAD & BASIC CLEAN -------------------------------------------------------
df = pd.read_csv(DATA_PATH)
# Basic renaming map based on dataset’s actual column names
col_map = {
    'CustomerID'       : 'customer_id',
    'Location'         : 'location',
    'Product_Category' : 'product_category',
    'Quantity'         : 'quantity',
    'Online_Spend'     : 'online_spend',
    'Offline_Spend'    : 'offline_spend',
    'Transaction_Date' : 'transaction_date',
    'Transaction_ID'   : 'transaction_id'
}
df = df.rename(columns = col_map)
# Parse dates
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
df['month'] = df['transaction_date'].dt.to_period('M')
df['year']  = df['transaction_date'].dt.year

# ------------------------------------------------------------------------------
# PART A – EXPLORATORY VISUALISATIONS
# ------------------------------------------------------------------------------

## 1. Region × Product Heatmap --------------------------------------------------
heat_df = (
    df.groupby(['location','product_category'])
      .agg(total_qty=('quantity','sum'))
      .reset_index()
      .pivot(index='product_category', columns='location', values='total_qty')
      .fillna(0)
)
plt.figure(figsize=(10,6))
sns.heatmap(heat_df, cmap='Reds', linewidths=.5)
plt.title('Quantity Sold by Product Category & Region')
plt.xlabel('Region')
plt.ylabel('Product Category')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/heatmap_region_product.png')
plt.close()

## 2. Online vs Offline Share (Stacked) ----------------------------------------
channel_df = (
    df.assign(total_spend = df['online_spend'] + df['offline_spend'])
      .melt(id_vars=['location'], 
            value_vars=['online_spend','offline_spend'],
            var_name='channel', value_name='spend')
      .groupby(['location','channel'])['spend']
      .sum()
      .reset_index()
)
pivot_chan = channel_df.pivot(index='location', columns='channel', values='spend')
pivot_chan_norm = pivot_chan.div(pivot_chan.sum(axis=1), axis=0)

pivot_chan_norm.plot(kind='bar', stacked=True, figsize=(10,6))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
plt.title('Channel Share by Region')
plt.ylabel('Share')
plt.xlabel('Region')
plt.legend(title='Channel')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/stacked_channel_region.png')
plt.close()

## 3. Pareto Analysis of Customer Spend ----------------------------------------
cust_df = (
    df.assign(total_spend = df['online_spend'] + df['offline_spend'])
      .groupby('customer_id')
      .agg(total_spend=('total_spend','sum'))
      .sort_values('total_spend', ascending=False)
      .reset_index()
)
cust_df['cum_pct'] = cust_df['total_spend'].cumsum() / cust_df['total_spend'].sum()

fig, ax1 = plt.subplots(figsize=(10,6))
ax1.bar(cust_df.index, cust_df['total_spend'], color='steelblue')
ax2 = ax1.twinx()
ax2.plot(cust_df.index, cust_df['cum_pct'], color='red', linewidth=2)
ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
ax1.set_title('Pareto Analysis – Customer Contribution')
ax1.set_xlabel('Customers (sorted)')
ax1.set_ylabel('Total Spend')
ax2.set_ylabel('Cumulative %')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/pareto_customer.png')
plt.close()

# ------------------------------------------------------------------------------
# PART B – K‑MEANS CLUSTERING WITH SILHOUETTE
# ------------------------------------------------------------------------------

rfm = (
    df.assign(total_spend = df['online_spend'] + df['offline_spend'])
      .groupby('customer_id')
      .agg(
          recency   = ('transaction_date', lambda x: (df['transaction_date'].max() - x.max()).days),
          frequency = ('transaction_id', 'nunique'),
          monetary  = ('total_spend','sum')
      )
      .reset_index()
)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['recency','frequency','monetary']])

sil_scores = []
for k in range(2,11):
    kmodel = KMeans(n_clusters=k, n_init=20, random_state=42)
    clusters = kmodel.fit_predict(rfm_scaled)
    sil_scores.append(silhouette_score(rfm_scaled, clusters))

opt_k = sil_scores.index(max(sil_scores)) + 2
kmodel_final = KMeans(n_clusters=opt_k, n_init=20, random_state=42)
rfm['cluster'] = kmodel_final.fit_predict(rfm_scaled)

print(f'Optimal k determined by silhouette: {opt_k}, score={max(sil_scores):.3f}')

# 3D scatter (interactive)
fig = px.scatter_3d(
    rfm, x='recency', y='frequency', z='monetary',
    color='cluster', symbol='cluster',
    title=f'RFM Clusters (k={opt_k})')
pio.write_html(fig, file=f'{OUT_DIR}/rfm_kmeans.html', auto_open=False)

# ------------------------------------------------------------------------------
# PART C – DBSCAN EXAMPLES
# ------------------------------------------------------------------------------

## 1. Customer Behaviour (Recency & Monetary) ----------------------------------
beh_df = rfm[['recency','monetary']].copy()
db = DBSCAN(eps=0.5, min_samples=5)
beh_df['cluster'] = db.fit_predict(StandardScaler().fit_transform(beh_df))

plt.figure(figsize=(8,6))
sns.scatterplot(data=beh_df, x='recency', y='monetary',
                hue='cluster', palette='Set1', legend='full')
plt.title('DBSCAN on Recency vs Monetary')
plt.savefig(f'{OUT_DIR}/dbscan_behaviour.png')
plt.close()

## 2. Daily Sales Anomaly Detection -------------------------------------------
daily = (
    df.assign(total_spend = df['online_spend'] + df['offline_spend'])
      .groupby(df['transaction_date'].dt.date)['total_spend']
      .sum()
      .reset_index()
      .rename(columns={'transaction_date':'date','total_spend':'sales'})
)
daily['date_idx'] = (pd.to_datetime(daily['date']) - pd.to_datetime(daily['date']).min()).dt.days
X = StandardScaler().fit_transform(daily[['date_idx','sales']])
db_ts = DBSCAN(eps=0.8, min_samples=5).fit(X)
daily['anomaly'] = db_ts.labels_ == -1

plt.figure(figsize=(10,5))
sns.scatterplot(data=daily, x='date', y='sales', hue='anomaly',
                palette={True:'red', False:'grey'}, legend=False)
plt.title('Sales Time‑Series Anomalies (DBSCAN)')
plt.xlabel('Date'); plt.ylabel('Sales')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/sales_anomaly_dbscan.png')
plt.close()

# ------------------------------------------------------------------------------
# PART D – BCG MATRIX (Product Growth vs Share)
# ------------------------------------------------------------------------------

# compute revenue current year vs previous year (yoy)
prod_year = (
    df.assign(revenue = df['online_spend'] + df['offline_spend'])
      .groupby(['product_category','year'])['revenue']
      .sum()
      .reset_index()
)
# pivot so we can compute yoy – requires at least 2 yrs
pivot_year = prod_year.pivot(index='product_category', columns='year', values='revenue').fillna(0)
if pivot_year.shape[1] >= 2:
    latest_year = pivot_year.columns.max()
    prev_year   = pivot_year.columns.sort_values()[-2]
    prod_df = pd.DataFrame({
        'product_category': pivot_year.index,
        'revenue_latest':   pivot_year[latest_year],
        'revenue_prev':     pivot_year[prev_year]
    })
    prod_df['yoy_growth'] = (prod_df['revenue_latest'] - prod_df['revenue_prev']) / prod_df['revenue_prev'].replace(0, np.nan)
    prod_df['market_share'] = prod_df['revenue_latest'] / prod_df['revenue_latest'].sum()

    plt.figure(figsize=(10,6))
    sns.scatterplot(data=prod_df, x='market_share', y='yoy_growth',
                    size='revenue_latest', sizes=(20, 400), hue='product_category', legend=False)
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.title('BCG Growth‑Share Matrix')
    plt.xlabel('Market Share'); plt.ylabel('YOY Growth')
    for _,row in prod_df.iterrows():
        plt.text(row['market_share'], row['yoy_growth'], row['product_category'], fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/bcg_matrix.png')
    plt.close()

# ------------------------------------------------------------------------------
# PART E – LORENZ CURVE & GINI
# ------------------------------------------------------------------------------

def gini(array):
    array = np.sort(array)
    n = array.size
    cum = np.cumsum(array, dtype=float)
    return (n + 1 - 2 * (cum / cum[-1]).sum()) / n

spend_vals = cust_df['total_spend'].values
spend_sorted = np.sort(spend_vals)
cum_spend = np.cumsum(spend_sorted) / spend_sorted.sum()
cum_pop = np.arange(1, len(spend_sorted)+1) / len(spend_sorted)

plt.figure(figsize=(6,6))
plt.plot(cum_pop, cum_spend, label='Lorenz Curve')
plt.plot([0,1],[0,1], color='black', linestyle='--')
plt.title(f'Lorenz Curve (Gini = {gini(spend_vals):.3f})')
plt.xlabel('Cumulative Share of Customers')
plt.ylabel('Cumulative Share of Spend')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/lorenz_curve.png')
plt.close()

print('All analyses complete. Outputs saved to', OUT_DIR)