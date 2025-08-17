# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load Dataset
data = pd.read_csv("Mall_Customers.csv")   # <- your dataset file
print("First 5 rows of dataset:")
print(data.head())

# For clustering, select useful features (e.g., Annual Income & Spending Score)
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Elbow Method to find optimal K
inertia = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow curve
plt.figure(figsize=(6,4))
plt.plot(K, inertia, 'bo-')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()

# 3. Fit KMeans with chosen K (e.g., 5 from elbow)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataframe
data["Cluster"] = labels

# 4. Evaluate Clustering with Silhouette Score
score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score for k={optimal_k}: {score:.3f}")

# 5. Visualize Clusters
plt.figure(figsize=(6,4))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels, cmap="viridis", s=50)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            c="red", marker="X", s=200, label="Centroids")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.title("Customer Segmentation with K-Means")
plt.legend()
plt.show()

# (Optional) PCA for high-dimensional datasets
# If you use >2 features, project data to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
labels_pca = KMeans(n_clusters=optimal_k, random_state=42).fit_predict(X_pca)

plt.figure(figsize=(6,4))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels_pca, cmap="viridis", s=50)
plt.title("Clusters Visualized with PCA")
plt.show()
