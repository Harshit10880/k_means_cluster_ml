from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

iris = load_iris()
X = iris.data[:, :2]

X[:5]

wcss = []
k_range = range(1, 11)

for k in k_range:
  kmean_algo = KMeans(n_clusters=k, random_state=42)
  kmean_algo.fit(X)
  wcss.append(kmean_algo.inertia_)


plt.plot(k_range, wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal k')
# plt.xticks(list(k_range))
plt.grid(True)
plt.show()

kmean_algo_final = KMeans(n_clusters=2)
cluster_label = kmean_algo_final.fit_predict(X)

X_df = pd.DataFrame(X, columns=['sepal_length', 'sepal_width'])
X_df['cluster_label'] = cluster_label
X = X_df

feature_x = X.columns[0]
feature_y = X.columns[1]
sns.scatterplot(x=feature_x, y=feature_y, hue=cluster_label, data=X)