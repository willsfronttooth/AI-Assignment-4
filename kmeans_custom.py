# src/kmeans_custom.py
"""
K-means clustering implementation from scratch with kmeans++ initialization.
"""
import numpy as np

class KMeansCustom:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, init='kmeans++', random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = np.random.RandomState(random_state)
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _init_centroids(self, X):
        n_samples = X.shape[0]
        if self.init == 'random':
            idx = self.random_state.choice(n_samples, self.n_clusters, replace=False)
            return X[idx].copy()
        # kmeans++:
        centers = []
        first_idx = self.random_state.randint(0, n_samples)
        centers.append(X[first_idx])
        for _ in range(1, self.n_clusters):
            dists = np.min(np.square(np.linalg.norm(X[:, None] - np.array(centers)[None, :, :], axis=2)), axis=1)
            probs = dists / dists.sum()
            cumulative = np.cumsum(probs)
            r = self.random_state.rand()
            next_idx = np.searchsorted(cumulative, r)
            centers.append(X[next_idx])
        return np.array(centers)

    def _assign(self, X, centers):
        distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def _update_centers(self, X, labels):
        centers = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            members = X[labels == k]
            if len(members) == 0:
                centers[k] = X[self.random_state.randint(0, X.shape[0])]
            else:
                centers[k] = members.mean(axis=0)
        return centers

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        centers = self._init_centroids(X)
        for i in range(self.max_iter):
            labels = self._assign(X, centers)
            new_centers = self._update_centers(X, labels)
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift <= self.tol:
                break
        self.cluster_centers_ = centers
        self.labels_ = self._assign(X, centers)
        self.inertia_ = np.sum((np.linalg.norm(X - centers[self.labels_], axis=1))**2)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self._assign(X, self.cluster_centers_)