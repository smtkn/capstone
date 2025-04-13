import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

# 1. Data Ingestion - Veriyi yerelden oku
file_path = "file.csv"
df = pd.read_csv(file_path)

# 2. Data Preprocessing - Eksik verileri kontrol et ve düzelt
print("Eksik veriler:")
print(df.isnull().sum())
cols_to_fill = ['Tenure_Months', 'Offline_Spend', 'Online_Spend', 'Quantity']
for col in cols_to_fill:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].mean())

# Boş olanları görmezden gelmek için sadece gerekli sütunları seçiyoruz
# (önceki işlemler zaten ortalama ile doldurdu, ama kontrol için ekledik)
df = df.dropna(subset=['Tenure_Months', 'Offline_Spend', 'Online_Spend', 'Quantity'])

# 3. Feature Extraction
df['Total_Spend'] = df['Offline_Spend'] + df['Online_Spend']
features = df[['Tenure_Months', 'Total_Spend', 'Quantity']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# 4. Clustering Algorithms
models = {
    'KMeans': KMeans(n_clusters=4, random_state=42),
    'Hierarchical': AgglomerativeClustering(n_clusters=4),
    'DBSCAN': DBSCAN(eps=1.0, min_samples=5)
}

results = {}
for name, model in models.items():
    labels = model.fit_predict(X_scaled)
    df[f'{name}_Cluster'] = labels
    if len(set(labels)) > 1 and -1 not in labels:
        score = silhouette_score(X_scaled, labels)
        print(f"{name} Silhouette Skoru: {score:.2f}")
    else:
        print(f"{name} için uygun küme bulunamadı veya çok az küme oluştu.")
    results[name] = labels

# 5. Visualization (KMeans örneği üzerinden)
df['KMeans_Cluster'] = df['KMeans_Cluster'].astype(int)
kmeans_means = df.groupby('KMeans_Cluster')[['Tenure_Months', 'Total_Spend', 'Quantity']].mean()

print("\nKMeans Küme Açıklamaları:")
kmeans_labels_dict = {}
for c in sorted(df['KMeans_Cluster'].unique()):
    cluster_data = df[df['KMeans_Cluster'] == c]
    tenure = cluster_data['Tenure_Months'].mean()
    spend = cluster_data['Total_Spend'].mean()
    quantity = cluster_data['Quantity'].mean()

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

    kmeans_labels_dict[c] = desc
    print(f"Cluster {c}: {desc}")

# Etiket ata
df['KMeans_Label'] = df['KMeans_Cluster'].map(kmeans_labels_dict)

# Küme isimlerini kategoriye çevirerek sıralamayı koru
df['KMeans_Label'] = pd.Categorical(df['KMeans_Label'], categories=list(set(kmeans_labels_dict.values())), ordered=False)

# 6. Visualizations
fig_3d = px.scatter_3d(df, x='Tenure_Months', y='Total_Spend', z='Quantity',
                       color='KMeans_Label', title='3D Müşteri Segmentasyonu (KMeans)')
fig_3d.show()

# Ortalama değer barplot
means_plot = df.groupby('KMeans_Label')[['Tenure_Months', 'Total_Spend', 'Quantity']].mean().reset_index().melt(id_vars='KMeans_Label', var_name='Özellik', value_name='Ortalama')
plt.figure(figsize=(10,6))
sns.barplot(data=means_plot, x='Özellik', y='Ortalama', hue='KMeans_Label')
plt.title("KMeans - Küme Bazında Ortalama Özellikler")
plt.tight_layout()
plt.show()

# Küme dağılımı - Pasta
plt.figure(figsize=(6, 6))
k_counts = df['KMeans_Label'].value_counts()
colors = sns.color_palette('Set2', len(k_counts))
plt.pie(k_counts, labels=k_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title("KMeans: Kümelere Göre Müşteri Dağılımı")
plt.axis('equal')
plt.tight_layout()
plt.show()
