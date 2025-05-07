
#!/usr/bin/env python3
"""
Customer Type Analysis Pipeline
--------------------------------
Reads data from csv/file.csv and generates:
    1. Region-Product heatmap
    2. Online vs Offline spend stacked bar per region
    3. Pareto chart of customer spend
    4. RFM calculation and KMeans / Hierarchical / DBSCAN clustering with 3D scatter
    5. Monthly sales & coupon usage time series
    6. Product family treemap
    7. (Optional) Region-based sales bubble map if coordinates available

All outputs are stored under ./output/
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px  # for treemap and optional map

DATA_PATH = 'csv/file.csv'
OUTPUT_DIR = 'output'

def ensure_output():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def load_data():
    df = pd.read_csv(DATA_PATH)
    # Standardize column names
    df.columns = df.columns.str.strip()
    # Parse dates
    if 'Transaction_Date' in df.columns:
        df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
    elif 'Date' in df.columns:
        df['Transaction_Date'] = pd.to_datetime(df['Date'])
    else:
        raise ValueError('Date column not found')
    return df

def region_product_heatmap(df):
    pivot = df.pivot_table(values='Quantity', index='Location',
                           columns='Product_Category', aggfunc='sum', fill_value=0)
    plt.figure(figsize=(12, 8))
    plt.imshow(pivot, aspect='auto')
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label='Quantity')
    plt.title('Region vs Product Category – Quantity Sold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'region_product_heatmap.png'))
    plt.close()

def sales_channel_stacked(df):
    channel = df.groupby('Location')[['Online_Spend', 'Offline_Spend']].sum()
    locations = channel.index
    online = channel['Online_Spend']
    offline = channel['Offline_Spend']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(locations, offline, label='Offline')
    ax.bar(locations, online, bottom=offline, label='Online')
    ax.set_ylabel('Total Spend')
    ax.set_title('Online vs Offline Spend by Region')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'online_offline_stacked.png'))
    plt.close()

def pareto_chart(df):
    df['Total_Spend'] = df['Online_Spend'] + df['Offline_Spend']
    customer_spend = df.groupby('CustomerID')['Total_Spend'].sum().sort_values(ascending=False)
    cumperc = customer_spend.cumsum() / customer_spend.sum() * 100
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(range(len(customer_spend)), customer_spend.values, label='Customer Spend')
    ax1.set_xlabel('Customers (ranked)')
    ax1.set_ylabel('Spend')
    ax2 = ax1.twinx()
    ax2.plot(range(len(cumperc)), cumperc.values, color='red', linestyle='--', marker='o', label='Cumulative %')
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.axhline(80, color='grey', linestyle='dotted')
    ax2.set_ylabel('Cumulative Percentage')
    ax1.set_title('Pareto Chart – Customer Spend Distribution')
    fig.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pareto_customer_spend.png'))
    plt.close()

def compute_rfm(df, snapshot_date=None):
    if snapshot_date is None:
        snapshot_date = df['Transaction_Date'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'Transaction_Date': lambda x: (snapshot_date - x.max()).days,
        'Transaction_ID': 'nunique',
        'Total_Spend': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm

def clustering_models(rfm):
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm)
    # KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['KMeans'] = kmeans.fit_predict(X)
    # Agglomerative
    hier = AgglomerativeClustering(n_clusters=4)
    rfm['Hierarchical'] = hier.fit_predict(X)
    # DBSCAN (auto eps estimation might be needed)
    db = DBSCAN(eps=0.8, min_samples=5)
    rfm['DBSCAN'] = db.fit_predict(X)
    # Save cluster assignments
    rfm.to_csv(os.path.join(OUTPUT_DIR, 'rfm_clusters.csv'))
    return rfm

def rfm_3d_scatter(rfm):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(rfm['Recency'], rfm['Frequency'], rfm['Monetary'],
                         c=rfm['KMeans'], s=50, alpha=0.6)
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Monetary')
    ax.set_title('RFM Segmentation – KMeans Clusters')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rfm_3d_scatter.png'))
    plt.close()

def monthly_sales_coupon(df):
    df['Month_Year'] = df['Transaction_Date'].dt.to_period('M')
    monthly = df.groupby('Month_Year').agg({
        'Total_Spend': 'sum',
        'Coupon_Status': lambda x: (x == 'Used').sum()
    }).reset_index()
    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(monthly['Month_Year'].astype(str), monthly['Total_Spend'], marker='o')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Total Sales')
    ax1.set_title('Monthly Sales & Coupon Usage')
    ax2 = ax1.twinx()
    ax2.bar(monthly['Month_Year'].astype(str), monthly['Coupon_Status'], alpha=0.3, label='Coupons Used')
    ax2.set_ylabel('Coupons Used')
    fig.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'monthly_sales_coupon.png'))
    plt.close()

def treemap_product_family(df):
    fam = df.groupby('Product_Category').agg({'Total_Spend': 'sum'}).reset_index()
    fig = px.treemap(fam, path=['Product_Category'], values='Total_Spend',
                     title='Product Family Revenue Share')
    fig.write_html(os.path.join(OUTPUT_DIR, 'product_family_treemap.html'))

def main():
    ensure_output()
    df = load_data()
    # Ensure Total_Spend exists
    if 'Total_Spend' not in df.columns:
        df['Total_Spend'] = df['Online_Spend'] + df['Offline_Spend']
    # Visualizations
    region_product_heatmap(df)
    sales_channel_stacked(df)
    pareto_chart(df)
    rfm = compute_rfm(df)
    rfm = clustering_models(rfm)
    rfm_3d_scatter(rfm)
    monthly_sales_coupon(df)
    treemap_product_family(df)
    print('Analysis complete. Check the output folder for results.')

if __name__ == '__main__':
    main()
