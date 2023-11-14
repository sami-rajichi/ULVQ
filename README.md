# Unsupervised Learning Vector Quantization (ULVQ)

Welcome to the Unsupervised Learning Vector Quantization (ULVQ) project! This repository contains an implementation of ULVQ for clustering in Python.

## Table of Contents
- [Introduction](#introduction)
- [Experimental Setup](#experimental-setup)
- [ULVQ Implementation](#ulvq-implementation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Introduction

ULVQ is an unsupervised learning algorithm designed for clustering. It aims to partition data into distinct groups based on their feature representations.

## Experimental Setup

### Dataset
For our experiments, we used the well-known Iris dataset. The Iris dataset consists of 150 samples, each belonging to one of three species of iris flowers. It is commonly used for testing clustering algorithms.

### Preprocessing and Feature Selection
We randomly selected two features from the Iris dataset for clustering experiments.

### Splitting the Dataset
The dataset was not split into training and testing sets for clustering, as our focus was on unsupervised learning.

## ULVQ Implementation

The ULVQ algorithm is implemented in the `ulvq.py` file. It includes a class, `ULVQ`, which can be used for unsupervised clustering.

### Parameters
- `n_clusters`: Number of clusters.
- `learning_rate`: Learning rate for updating cluster centroids.

### Methods
- `fit(X, n_epochs)`: Train ULVQ on input data `X` for a specified number of epochs.
- `predict(X)`: Predict cluster labels for input data `X`.
- `label_clusters(X, method)`: Label clusters using a specified clustering method ('kmeans', 'hierarchical', 'dbscan', 'gmm', 'agglomerative').
- `evaluate_clustering(X, labels_true, method)`: Evaluate clustering performance using specified method ('v_measure', 'silhouette', 'davies_bouldin', 'calinski_harabasz').
- `visualize_clusters(X, labels_true, labels_pred, feature_names)`: Visualize clustering results.

## Usage

To use ULVQ for clustering, follow these steps:

1. Import the `ULVQ` class from `ulvq.py`.
2. Create an instance of `ULVQ`.
3. Fit the model on your data using the `fit` method.
4. Predict cluster labels using the `predict` method.
5. Evaluate clustering performance with the `evaluate_clustering` method.
6. Visualize the clustering results with the `visualize_clusters` method.

Example usage:

```python
from ulvq import ULVQ
ulvq = ULVQ(n_clusters=3, learning_rate=0.1)
ulvq.fit(X, n_epochs=1000)
labels_pred = ulvq.predict(X)
ulvq.visualize_clusters(X, labels_true, labels_pred, feature_names=['feature1', 'feature2'])
ulvq.evaluate_clustering(X_scaled, labels_true, display_all='True')
ulvq.evaluate_labeling(X_scaled, labels_true, display_all='True')
```

## Results

The experiments yielded the following evaluation scores with the IRIS Dataset:

### Clustering Evaluation Metrics:
- V-Measure: 0.643923
- Silhouette: 0.431089
- Davies-Bouldin Index: 0.723859
- Calinski-Harabasz Index: 171.915681

### Labeling Evaluation Metrics:
- Accuracy: 0.726667
- Adjusted Rand Index: 0.544069
- Normalized Mutual Information: 0.643923

## License

This project is licensed under the [Apache License 2.0](LICENSE)
