"""Assignment — K-Means Clustering Project"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
ds = pd.read_csv(r"C:\Users\Amirhamza\OneDrive\Desktop\Clustering\Project_Clustering_Dataset.csv")

# Drop unnecessary column
if "ID" in ds.columns:
    ds.drop(columns=["ID"], inplace=True)

# -----------------------------
# 2️⃣ Extract text features
# -----------------------------
# Extract Keyword (word after 'for')
ds['Keyword'] = ds['Description'].str.extract(r'for (\w+)')[0]
ds.drop(columns=["Description"], inplace=True)

# Extract Company (word before 'for')
ds['Company'] = ds['Title'].str.extract(r'^(.*?)\s+for')[0]
ds.drop(columns=["Title"], inplace=True)

"""
Regex explanation:
^           → Start of string
(.*?)       → Capture everything (non-greedy)
\s+for      → Space(s) followed by literal 'for'
"""

# -----------------------------
# 3️⃣ Encode categorical columns
# -----------------------------
le = LabelEncoder()
ds['Keyword_encoded'] = le.fit_transform(ds['Keyword'])
ds['Company_encoded'] = le.fit_transform(ds['Company'])

print("\n✅ Sample data after encoding:")
print(ds.head())

# -----------------------------
# 4️⃣ Prepare numeric features
# -----------------------------
X = ds[['Keyword_encoded', 'Company_encoded']]

# -----------------------------
# 5️⃣ Elbow Method to find best K
# -----------------------------
wcss = []
for i in range(2, 21):
    kn = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kn.fit(X)
    wcss.append(kn.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8,5))
plt.plot(range(2, 21), wcss, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS (Inertia)")
plt.grid(axis='x')
plt.tight_layout()
plt.savefig("images/elbow_method.png")
plt.show()

# -----------------------------
# 6️⃣ Apply KMeans with chosen K
# -----------------------------
k_optimal = 5  # (you can change this based on elbow result)
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
ds['Cluster'] = kmeans.fit_predict(X)

# -----------------------------
# 7️⃣ Evaluate with Silhouette Score
# -----------------------------
score = silhouette_score(X, ds['Cluster'])
print(f"\n📊 Silhouette Score: {score:.4f}")

# -----------------------------
# 8️⃣ Visualization of Clusters
# -----------------------------
plt.figure(figsize=(7,6))
plt.scatter(ds['Keyword_encoded'], ds['Company_encoded'], c=ds['Cluster'], cmap='viridis')
plt.title(f"K-Means Clustering (k={k_optimal})")
plt.xlabel("Keyword Encoded")
plt.ylabel("Company Encoded")
plt.tight_layout()
plt.savefig("images/kmeans_clusters.png")
plt.show()

# -----------------------------
# 9️⃣ View sample rows from each cluster
# -----------------------------
for c in sorted(ds['Cluster'].unique()):
    print(f"\n🔹 Cluster {c}:")
    print(ds[ds['Cluster'] == c][['Keyword','Company']].head(10))

# -----------------------------
# 🔟 Save clustered results
# -----------------------------
ds.to_csv("Clustered_Output.csv", index=False)
print("\n✅ Clustered data saved as 'Clustered_Output.csv'")
print("✅ Elbow and cluster plots saved in 'images/' folder")
