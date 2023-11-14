import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics.cluster import v_measure_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from IPython.display import display

class ULVQ:
    """
    Unsupervised Learning Vector Quantization (ULVQ) for clustering.
    """
    def __init__(self, n_clusters=2, learning_rate=0.1):
        """
        Initialize ULVQ with the number of clusters and learning rate.

        Parameters:
        - n_clusters: Number of clusters.
        - learning_rate: Learning rate for updating cluster centroids.
        """
        self.n_clusters = n_clusters
        self.learning_rate = learning_rate

    def fit(self, X, n_epochs=100):
        """
        Fit ULVQ to the data.

        Parameters:
        - X: Input data.
        - n_epochs: Number of training epochs.
        """
        # Initialize cluster centroids randomly
        self.weights = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        
        for epoch in range(n_epochs):
            for i in range(X.shape[0]):
                x = X[i]
                
                # Calculate distances to cluster centroids
                distances = np.linalg.norm(x - self.weights, axis=1)
                
                # Find the closest cluster centroid
                winner = np.argmin(distances)
                
                # Update the centroid
                self.weights[winner] += self.learning_rate * (x - self.weights[winner])

    def predict(self, X):
        """
        Predict the cluster labels for input data.

        Parameters:
        - X: Input data.

        Returns:
        - Array of cluster labels.
        """
        # Calculate distances to cluster centroids for predictions
        distances = np.linalg.norm(X[:, np.newaxis] - self.weights, axis=2)

        # Predict the cluster based on the closest centroid
        return np.argmin(distances, axis=1)

    def label_clusters(self, X, method='kmeans'):
        """
        Label clusters using a specified clustering method.

        Parameters:
        - X: Input data.
        - method: Clustering method ('kmeans', 'hierarchical', 'dbscan', 'gmm', 'agglomerative').

        Returns:
        - Array of cluster labels.
        """
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_clusters)
            return kmeans.fit_predict(X)
        elif method == 'hierarchical':
            linkage_matrix = linkage(X, method='complete')
            dendrogram_result = dendrogram(linkage_matrix, truncate_mode='lastp')
            cluster_labels = fcluster(linkage_matrix, t=0.8, criterion='distance')
            return cluster_labels - 1  # Adjust labels to start from 0
        elif method == 'dbscan':
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            return dbscan.fit_predict(X)
        elif method == 'gmm':
            gmm = GaussianMixture(n_components=self.n_clusters)
            return gmm.fit_predict(X)
        elif method == 'agglomerative':
            agglomerative = AgglomerativeClustering(n_clusters=self.n_clusters)
            return agglomerative.fit_predict(X)
        else:
            raise ValueError("Invalid clustering method. Supported methods: 'kmeans', 'hierarchical', 'dbscan', 'gmm', 'agglomerative'")
    
    def evaluate_clustering(self, X, labels_true, method='v_measure', display_all=False):
        """
        Evaluate clustering using specified method.

        Parameters:
        - X: Input data.
        - labels_true: True cluster labels.
        - method: Evaluation method ('v_measure', 'silhouette', 'davies_bouldin', 'calinski_harabasz').
        - display_all: If True, display all evaluation metrics.

        Returns:
        - Evaluation score or None if display_all is True.
        """
        labels_pred = self.predict(X)

        if display_all:
            return display(pd.DataFrame(
                {
                    "v_measure": [v_measure_score(labels_true, labels_pred)],
                    "silhouette": [silhouette_score(X, labels_pred)],
                    "davies_bouldin": [davies_bouldin_score(X, labels_pred)],
                    "calinski_harabasz": [calinski_harabasz_score(X, labels_pred)]
                }
            ))
        elif method == 'v_measure':
            return v_measure_score(labels_true, labels_pred)
        elif method == 'silhouette':
            return silhouette_score(X, labels_pred)
        elif method == 'davies_bouldin':
            return davies_bouldin_score(X, labels_pred)
        elif method == 'calinski_harabasz':
            return calinski_harabasz_score(X, labels_pred)
        else:
            raise ValueError("Invalid evaluation method. Supported methods: 'v_measure', 'silhouette', 'davies_bouldin', 'calinski_harabasz'")
    
    def evaluate_labeling(self, X, true_labels, method='accuracy', display_all=False):
        """
        Evaluate labeling using specified method.

        Parameters:
        - method: Evaluation method ('accuracy', 'adjusted_rand', 'normalized_mutual_info').

        Returns:
        - Evaluation score.
        """
        predicted_labels = self.predict(X)
        if display_all:
            return display(pd.DataFrame(
                {
                    "accuracy": [accuracy_score(true_labels, predicted_labels)],
                    "adjusted_rand": [adjusted_rand_score(true_labels, predicted_labels)],
                    "normalized_mutual_info": [normalized_mutual_info_score(true_labels, predicted_labels)]
                }
            ))
        elif method == 'accuracy':
            return accuracy_score(true_labels, predicted_labels)
        elif method == 'adjusted_rand':
            return adjusted_rand_score(true_labels, predicted_labels)
        elif method == 'normalized_mutual_info':
            return normalized_mutual_info_score(true_labels, predicted_labels)
        else:
            raise ValueError("Invalid evaluation method. Supported methods: 'accuracy', 'adjusted_rand', 'normalized_mutual_info'")
        
    def visualize_clusters(self, X, labels_true, labels_pred, feature_names, title="Unsupervised LVQ Clustering"):
        """
        Visualize clustering results with true labels, clustering labels, and centroids markers.

        Parameters:
        - X: Input data.
        - labels_true: True cluster labels.
        - labels_pred: Predicted cluster labels.
        - feature_names: Names of features.
        - title: Title for the plot.
        """
        data_true = np.column_stack((X, labels_true))
        data_pred = np.column_stack((X, labels_pred))
        
        features = [f for f in feature_names]
        features.append('True_Label')
        df_true = pd.DataFrame(data_true, columns=features)
        
        features[-1] = 'Cluster_Label'
        df_pred = pd.DataFrame(data_pred, columns=features)

        plt.figure(figsize=(14, 6))

        # Plot True Labels
        plt.subplot(1, 2, 1)
        sns.scatterplot(data=df_true, x=df_true.columns[0], y=df_true.columns[1], hue=df_true.columns[2], palette="Set1", s=100, marker="o")
        
        # Add Centroids Markers
        centroids_true = self.weights[:, :2]  # Assuming the centroids have 2 features
        plt.scatter(centroids_true[:, 0], centroids_true[:, 1], marker='X', s=200, c='black', label='Centroids')
        
        plt.title("True Labels")
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.legend()

        # Plot Cluster Labels
        plt.subplot(1, 2, 2)
        sns.scatterplot(data=df_pred, x=df_pred.columns[0], y=df_pred.columns[1], hue=df_pred.columns[2], palette="Set1", s=100, marker="o")
        
        # Add Centroids Markers
        centroids_pred = self.weights[:, :2]  # Assuming the centroids have 2 features
        plt.scatter(centroids_pred[:, 0], centroids_pred[:, 1], marker='X', s=200, c='black', label='Centroids')
        
        plt.title("Cluster Labels")
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.legend()

        plt.suptitle(title)
        plt.show()
