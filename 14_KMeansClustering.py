from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('income.csv')
# Data Scale
scaler = MinMaxScaler()
scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])
scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])
# Cluster
km = KMeans(n_clusters=3)
yPredicted = km.fit_predict(df[['Age', 'Income($)']])
df['cluster'] = yPredicted
# Filter Data
df0 = df[df['cluster']==0]
df1 = df[df['cluster']==1]
df2 = df[df['cluster']==2]
# plt.scatter(df0['Age'], df0['Income($)'], color='green')
# plt.scatter(df1['Age'], df1['Income($)'], color='red')
# plt.scatter(df2['Age'], df2['Income($)'], color='black')
# plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='purple', marker='+', label='centroid')

k_rgn = range(1, 10)
sse = []
for k in k_rgn:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age', 'Income($)']])
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(k_rgn, sse)
plt.show()