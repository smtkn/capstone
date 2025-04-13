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
dbscan_labels = evaluate_model('DBSCAN', DBSCAN(eps=1.2, min_samples=5), X_scaled)

# 5. Görselleştirme
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
plot_clusters_3d("KMeans - 3D Küme Görselleştirmesi", df['KMeans_Cluster'].values)
plot_cluster_distribution("KMeans", df['KMeans_Cluster'].values)

# Hierarchical Görselleştirme
plot_clusters_3d("Hierarchical - 3D Küme Görselleştirmesi", df['Hierarchical_Cluster'].values)
plot_cluster_distribution("Hierarchical", df['Hierarchical_Cluster'].values)

# DBSCAN Görselleştirme (varsa)
if len(set(dbscan_labels)) > 1:
    plot_clusters_3d("DBSCAN - 3D Küme Görselleştirmesi", df['DBSCAN_Cluster'].values)
    plot_cluster_distribution("DBSCAN", df['DBSCAN_Cluster'].values)
else:
    print("DBSCAN sonuçları yeterli küme oluşturmuyor, görselleştirme atlandı.")
