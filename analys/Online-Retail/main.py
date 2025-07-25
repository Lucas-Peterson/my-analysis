import pandas as pd
import kagglehub
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch

# Download dataset using KaggleHub API
path = kagglehub.dataset_download("hellbuoy/online-retail-customer-clustering")
dataset_path = f"{path}/OnlineRetail.csv"

# Read the data
df = pd.read_csv(dataset_path, encoding='ISO-8859-1')

# Data preprocessing: remove rows with missing values and invalid entries
df = df.dropna(subset=['CustomerID'])
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

# Aggregate data at the customer level
customer_df = df.groupby('CustomerID').agg({
    'Quantity': 'sum',
    'UnitPrice': 'mean',
    'InvoiceNo': 'nunique',
    'StockCode': 'nunique'
}).reset_index()

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_df.drop(columns=['CustomerID']))

# --------- K-Means Clustering ---------
# Elbow method to determine optimal clusters
sse = []
k_values = list(range(1, 11))
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, sse, marker='o')
plt.title('Elbow Method for K-Means')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.grid(True)
plt.show()

# Clustering with the optimal number of clusters (e.g., 4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
customer_df['KMeans_Cluster'] = kmeans.fit_predict(scaled_data)

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Quantity', y='UnitPrice', hue='KMeans_Cluster', data=customer_df, palette='viridis', s=100)
plt.title('Customer Clustering (K-Means)')
plt.xlabel('Total Purchases')
plt.ylabel('Average Unit Price')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# --------- Hierarchical Clustering ---------
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(scaled_data, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.grid(True)
plt.show()

hierarchical = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
customer_df['Hierarchical_Cluster'] = hierarchical.fit_predict(scaled_data)

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Quantity', y='UnitPrice', hue='Hierarchical_Cluster', data=customer_df, palette='viridis', s=100)
plt.title('Customer Clustering (Hierarchical)')
plt.xlabel('Total Purchases')
plt.ylabel('Average Unit Price')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()
