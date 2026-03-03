from __future__ import annotations
from sklearn.cluster import KMeans

def fit_kmeans(X, n_clusters: int = 6, random_state: int = 42):
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    km.fit(X)
    return km
