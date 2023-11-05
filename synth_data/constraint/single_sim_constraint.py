from .constraint import Constraint 

from abc import ABC, abstractmethod

import numpy as np

class SingleSimilarityConstraint(Constraint, ABC):

    def __init__(self, threshold, metric, cluster_idxs, list_of_clusters):
        
        enforced_clusters = [list_of_clusters[i] for i in cluster_idxs]
        super().__init__(enforced_clusters)

        self.threshold      = threshold
        self.metric         = metric


    def compute_sim_kern(self, cluster_1, cluster_2):

        if self.metric == "cosine":
            norm_cluster_1  = np.sqrt(np.sum(cluster_1 ** 2, axis=-1))
            norm_cluster_2  = np.sqrt(np.sum(cluster_2 ** 2, axis=-1))
            scale_cluster_1 = cluster_1 / norm_cluster_1
            scale_cluster_2 = cluster_2 / norm_cluster_2
            similarity_kern = (1 + np.tensordot(scale_cluster_1, scale_cluster_2.T, axes=1)) / 2
        elif self.metric.startswith("rbf"):
            sigma           = float(self.metric.split("_")[1])
            pairwise_diff   = cluster_1[:,None,:] - cluster_2[None,:,:]
            norm_sq         = np.sum(pairwise_diff ** 2, axis=-1)
            exponent        = -norm_sq / sigma
            similarity_kern = np.exp(exponent)

        return similarity_kern 


    @abstractmethod
    def kernel_check(self, sim_kern):
        pass


    def enforce(self):
        
        n_clusters          = len(self.clusters)
        cluster_removal_idx = {i: set() for i in range(n_clusters)}

        # Mark idx that should be removed
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                cluster_1_np    = self.clusters[i].numpy_cluster
                cluster_2_np    = self.clusters[j].numpy_cluster
                sim_kern        = self.compute_sim_kern(cluster_1_np, cluster_2_np)
                good_cells      = self.kernel_check(sim_kern)
                remove_1_idx    = np.arange(cluster_1_np.shape[0])[~np.all(good_cells, axis=1)]
                remove_2_idx    = np.arange(cluster_2_np.shape[0])[~np.all(good_cells, axis=0)]
                cluster_removal_idx[i].update(remove_1_idx.tolist())
                cluster_removal_idx[j].update(remove_2_idx.tolist())
            
        # For each cluster, keep only that which isn't slated for removal
        removed_any = False
        for i in range(n_clusters):
            removed_any                     = removed_any or len(cluster_removal_idx[i]) > 0
            keep_idx                        = np.arange(self.clusters[i].numpy_cluster.shape[0])
            keep_idx                        = keep_idx[~np.isin(keep_idx, list(cluster_removal_idx[i]))]
            self.clusters[i].numpy_cluster  = self.clusters[i].numpy_cluster[keep_idx]

        return not removed_any