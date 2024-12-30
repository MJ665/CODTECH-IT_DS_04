
---

# Mall Customers Clustering Analysis

## Overview
This project performs various unsupervised learning techniques for clustering analysis on the Mall Customer dataset. The task involves identifying groups of customers based on their demographic and spending behavior. Multiple clustering algorithms including K-Means, DBSCAN, and Hierarchical Clustering are implemented. The analysis also includes dimensionality reduction techniques like PCA and t-SNE for better visualization of the clusters.

## Table of Contents
1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Clustering Algorithms](#clustering-algorithms)
   - [K-Means](#k-means)
   - [DBSCAN](#dbscan)
   - [Hierarchical Clustering](#hierarchical-clustering)
5. [Dimensionality Reduction](#dimensionality-reduction)
6. [Visualizations](#visualizations)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Usage](#usage)
9. [Contributing](#contributing)

## Installation

To run the code, you will need to install the following libraries:

```bash
pip install pandas matplotlib seaborn scikit-learn scipy
```

## Dataset

The dataset used in this analysis is the **Mall_Customers.csv**, which contains demographic and spending information about customers. The relevant columns for clustering include:

- **Age**: Customer age.
- **Annual Income (k$)**: Annual income of the customer (in thousands of dollars).
- **Spending Score (1-100)**: A score assigned to customers based on their spending behavior.

## Data Preprocessing

- **Missing Value Handling**: Missing values in the dataset are dropped.
- **Feature Scaling**: The features (Age, Annual Income, and Spending Score) are scaled using **StandardScaler** to standardize the data before applying clustering algorithms.

## Clustering Algorithms

### K-Means
The K-Means algorithm is applied to the scaled features, and the dataset is divided into 5 clusters. We evaluate the clustering performance using:

- **Silhouette Score**: Measures the quality of clusters.
- **Davies-Bouldin Index**: A lower score indicates better clustering.

### DBSCAN
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is used to identify clusters of varying shapes. It does not require specifying the number of clusters beforehand and is effective in finding noise points.

### Hierarchical Clustering
Agglomerative hierarchical clustering is performed and visualized using a **dendrogram** to determine the optimal number of clusters.

## Dimensionality Reduction

To better visualize the high-dimensional clustering results, we apply dimensionality reduction techniques:

- **PCA (Principal Component Analysis)**: Reduces the data to two components for visualization in 2D space.
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: A non-linear dimensionality reduction technique for visualizing clusters.

## Visualizations

- **Elbow Method**: Used to determine the optimal number of clusters for K-Means by plotting the inertia (sum of squared distances).
- **Pairplot**: Visualizes the pairwise relationships between features, colored by the K-Means cluster labels.
- **Scatter Plots**: Visualize the results of clustering and dimensionality reduction using PCA and t-SNE.

## Evaluation Metrics

- **Silhouette Score**: Measures how similar each sample is to its own cluster compared to other clusters.
- **Davies-Bouldin Index**: Measures the average similarity ratio of each cluster with the cluster that is most similar to it.

## Usage

1. Clone the repository or download the **Mall_Customers.csv** dataset.
2. Run the Python script containing the clustering code.
3. Review the visualizations and evaluation metrics to understand the clustering results.
4. Experiment with different values for DBSCAN's `eps` and `min_samples` parameters, and explore the hierarchical clustering dendrogram for alternative clustering results.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit pull requests. Contributions can include improvements in clustering algorithms, additional data visualizations, or enhancing the README with more detailed instructions.

---

