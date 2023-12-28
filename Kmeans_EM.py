import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Extract features (assuming all columns except 'CustomerID' and 'Genre' are features)
X = data.iloc[:, 2:].values

# Apply K-means clustering
kmeans = KMeans(n_clusters=3)  # You can change the number of clusters as needed
kmeans_labels = kmeans.fit_predict(X)

# Apply EM clustering (Gaussian Mixture Model)
em = GaussianMixture(n_components=3)  # You can change the number of components as needed
em_labels = em.fit_predict(X)

# Add the cluster labels to the original dataset
data['KMeans_Labels'] = kmeans_labels
data['EM_Labels'] = em_labels

# Visualize the results
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='KMeans_Labels', data=data, palette='Set1', legend='full')
plt.title('K-Means Clustering')
plt.show()

sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='EM_Labels', data=data, palette='Set1', legend='full')
plt.title('Expectation-Maximization (EM) Clustering')
plt.show()
