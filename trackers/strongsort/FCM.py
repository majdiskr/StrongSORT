pip install scikit-fuzzy
import numpy as np
import skfuzzy as fuzz

class FCM:
    def __init__(self, data, num_clusters, max_iters=100, error=1e-5):
        self.data = data
        self.num_clusters = num_clusters
        self.max_iters = max_iters
        self.error = error

    def cluster(self):
        # Initialize the cluster centers
        centers, _, _, _, _, _, _ = fuzz.cluster.cmeans(
            self.data.T,
            self.num_clusters,
            m=2,
            error=self.error,
            maxiter=self.max_iters
        )

        # Assign each point to the cluster with the highest membership value
        memberships, _ = fuzz.cluster.cmeans_predict(
            self.data.T,
            centers,
            m=2,
            error=self.error,
            maxiter=self.max_iters
        )

        # Return the cluster centers and the memberships
        return centers.T, memberships.T
