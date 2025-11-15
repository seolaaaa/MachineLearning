import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("\nLoading dataset...")

#Load dataset
df = pd.read_csv("StudentsPerformance.csv")

print("\nDataset Loaded Successfully!")
print(df.head())
print("\nColumns:", df.columns)


#ENCODE CATEGORICAL COLUMNS


label_cols = ["gender", "race/ethnicity", "parental level of education",
              "lunch", "test preparation course"]

encoder = LabelEncoder()

for col in label_cols:
    df[col] = encoder.fit_transform(df[col])


#SELECT FEATURES (no target label needed)

features = df[[
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
    "math score",
    "reading score",
    "writing score"
]]


scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


#K-MEANS CLUSTERING


kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Add cluster label to dataframe
df["Cluster"] = clusters

print("\nClustering complete!")
print(df.head())


#VISUALIZE CLUSTERS USING MATPLOTLIB


plt.figure(figsize=(8,5))

#Create a scatter plot for each cluster with labels
for cluster_id in df["Cluster"].unique():
    cluster_data = df[df["Cluster"] == cluster_id]
    plt.scatter(
        cluster_data["math score"],
        cluster_data["reading score"],
        label=f"Cluster {cluster_id}"
    )

plt.xlabel("Math Score")
plt.ylabel("Reading Score")
plt.title("Student Clusters (Unsupervised Learning)")
plt.legend(title="Cluster Groups")
plt.tight_layout()
plt.show()
