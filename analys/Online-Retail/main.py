import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch

# Reading the data
df = pd.read_csv('OnlineRetail.csv', encoding='ISO-8859-1')

# Data preprocessing: removing rows with missing values and invalid entries
df = df.dropna(subset=['CustomerID'])
df = df[df['Quantity'] > 0]  # Remove negative or zero quantities
df = df[df['UnitPrice'] > 0]  # Remove rows with zero price

# Aggregating data at the customer level
customer_df = df.groupby('CustomerID').agg({
    'Quantity': 'sum',           # Total number of purchases
    'UnitPrice': 'mean',         # Average unit price
    'InvoiceNo': 'nunique',      # Number of unique purchases (invoices)
    'StockCode': 'nunique'       # Number of unique products
}).reset_index()

# Scaling the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_df.drop(columns=['CustomerID']))

# ----------------- K-Means Clustering -----------------
# Determining the optimal number of clusters using the elbow method
sse = []
k_values = list(range(1, 11))
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    sse.append(kmeans.inertia_)

# Visualizing the elbow method
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

# Visualizing K-Means clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Quantity', y='UnitPrice', hue='KMeans_Cluster', data=customer_df, palette='viridis', s=100)
plt.title('Customer Clustering (K-Means)')
plt.xlabel('Total Purchases')
plt.ylabel('Average Unit Price')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# ----------------- Hierarchical Clustering -----------------
# Building a dendrogram for hierarchical clustering
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(scaled_data, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.grid(True)
plt.show()

# Hierarchical clustering with the optimal number of clusters (e.g., 4)
hierarchical = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
customer_df['Hierarchical_Cluster'] = hierarchical.fit_predict(scaled_data)

# Visualizing hierarchical clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Quantity', y='UnitPrice', hue='Hierarchical_Cluster', data=customer_df, palette='viridis', s=100)
plt.title('Customer Clustering (Hierarchical)')
plt.xlabel('Total Purchases')
plt.ylabel('Average Unit Price')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()
