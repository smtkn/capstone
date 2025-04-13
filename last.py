import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# 1. Veri Yükleme
file_path = "csv/file.csv"
df = pd.read_csv(file_path)

# 2. Veri Temizleme
print("Eksik veriler:")
print(df.isnull().sum())

columns_to_use = ['Tenure_Months', 'Offline_Spend', 'Online_Spend', 'Quantity']
for col in columns_to_use:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].mean())

df = df.dropna(subset=columns_to_use)
df['Total_Spend'] = df['Offline_Spend'] + df['Online_Spend']

# 3. Özellikler ve Ölçekleme
features = df[['Tenure_Months', 'Total_Spend', 'Quantity']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# 4. Kümelendirme Algoritmaları
def evaluate_model(name, model, data):
    labels = model.fit_predict(data)
    df[f'{name}_Cluster'] = labels
    if len(set(labels)) > 1 and (name != 'DBSCAN' or -1 not in labels):
        score = silhouette_score(data, labels)
        print(f"{name} Silhouette Skoru: {score:.2f}")
    else:
        print(f"{name}: Yetersiz veya geçersiz küme yapısı")
    return labels

kmeans_labels = evaluate_model('KMeans', KMeans(n_clusters=4, random_state=42), X_scaled)
hier_labels = evaluate_model('Hierarchical', AgglomerativeClustering(n_clusters=4), X_scaled)
# DBSCAN için farklı eps ve min_samples değerleri deneyerek otomatik seçim
def auto_tune_dbscan(data, eps_values, min_samples_values):
    best_score = -1
    best_labels = None
    best_params = None

    for eps in eps_values:
        for min_samples in min_samples_values:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(data)
            if len(set(labels)) > 1 and -1 not in labels:
                try:
                    score = silhouette_score(data, labels)
                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        best_params = (eps, min_samples)
                except:
                    continue
    return best_labels, best_params, best_score

# Parametre aralığı tanımı
eps_range = [0.8, 1.0, 1.2, 1.5, 2.0]
min_samples_range = [3, 4, 5, 6]

dbscan_labels, dbscan_best_params, dbscan_score = auto_tune_dbscan(X_scaled, eps_range, min_samples_range)

if dbscan_labels is not None:
    df['DBSCAN_Cluster'] = dbscan_labels
    print(f"DBSCAN en iyi parametreler: eps={dbscan_best_params[0]}, min_samples={dbscan_best_params[1]}")
    print(f"DBSCAN Silhouette Skoru: {dbscan_score:.2f}")
else:
    print("DBSCAN: Uygun parametre bulunamadı. Kümeleme başarısız.")

# 5. Küme Açıklamaları ve Görselleştirme Hazırlığı
def describe_clusters(name, cluster_col):
    descriptions = {}
    clusters = df[cluster_col].unique()
    for c in clusters:
        subset = df[df[cluster_col] == c]
        tenure = subset['Tenure_Months'].mean()
        spend = subset['Total_Spend'].mean()
        quantity = subset['Quantity'].mean()

        if spend > 10000 and tenure > 12:
            desc = "Sadık ve yüksek harcama yapan müşteriler"
        elif spend > 10000:
            desc = "Yüksek harcayan ama yeni müşteriler"
        elif quantity > 20:
            desc = "Çok sık alışveriş yapan ama düşük harcama yapan müşteriler"
        elif tenure < 6 and spend < 3000:
            desc = "Yeni ve düşük harcama yapan müşteriler"
        else:
            desc = "Orta seviye ama potansiyeli olan müşteriler"

        descriptions[c] = desc
        print(f"{name} Cluster {c}: {desc}")

    df[f'{name}_Label'] = df[cluster_col].map(descriptions)
    return df[f'{name}_Label']

kmeans_labels_named = describe_clusters('KMeans', 'KMeans_Cluster')
hier_labels_named = describe_clusters('Hierarchical', 'Hierarchical_Cluster')
if 'DBSCAN_Cluster' in df.columns and len(set(df['DBSCAN_Cluster'])) > 1 and -1 not in df['DBSCAN_Cluster'].values:
    dbscan_labels_named = describe_clusters('DBSCAN', 'DBSCAN_Cluster')

def plot_clusters_3d(title, labels):
    fig = px.scatter_3d(df, x='Tenure_Months', y='Total_Spend', z='Quantity',
                        color=labels.astype(str), title=title)
    fig.show()

def plot_cluster_distribution(name, labels):
    label_series = pd.Series(labels).astype(str)
    label_counts = label_series.value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f"{name} Küme Dağılımı")
    plt.tight_layout()
    plt.axis('equal')
    plt.show()

# KMeans Görselleştirme
plot_clusters_3d("KMeans - 3D Küme Görselleştirmesi", kmeans_labels_named)
plot_cluster_distribution("KMeans", kmeans_labels_named)
plot_clusters_3d("KMeans - 3D Küme Görselleştirmesi", df['KMeans_Cluster'].values)
plot_cluster_distribution("KMeans", df['KMeans_Cluster'].values)

# Hierarchical Görselleştirme
plot_clusters_3d("Hierarchical - 3D Küme Görselleştirmesi", hier_labels_named)
plot_cluster_distribution("Hierarchical", hier_labels_named)
plot_clusters_3d("Hierarchical - 3D Küme Görselleştirmesi", df['Hierarchical_Cluster'].values)
plot_cluster_distribution("Hierarchical", df['Hierarchical_Cluster'].values)

# DBSCAN Görselleştirme (varsa)
if 'DBSCAN_Cluster' in df.columns and len(set(dbscan_labels)) > 1 and -1 not in dbscan_labels:
    plot_clusters_3d("DBSCAN - 3D Küme Görselleştirmesi", dbscan_labels_named)
    plot_cluster_distribution("DBSCAN", dbscan_labels_named)
else:
    print("DBSCAN sonuçları yeterli küme oluşturmuyor, görselleştirme atlandı.")
