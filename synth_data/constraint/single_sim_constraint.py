from .constraint import Constraint 

from abc import ABC, abstractmethod

import numpy as np

class SingleSimilarityConstraint(Constraint, ABC):

    def __init__(self, threshold, metric, cluster_idxs, list_of_clusters):
        
        enforced_clusters = [list_of_clusters[i] for i in cluster_idxs]
        super().__init__(enforced_clusters)

        self.cluster_idxs   = cluster_idxs
        self.threshold      = threshold
        self.metric         = metric

        if self.cluster_idxs[0] == self.cluster_idxs[1]:
            self.intra_cluster = True
        else:
            self.intra_cluster = False


    def compute_sim_kern(self, cluster_1, cluster_2):

        if self.metric == "cosine":
            norm_cluster_1  = np.sqrt(np.sum(cluster_1 ** 2, axis=-1))
            norm_cluster_2  = np.sqrt(np.sum(cluster_2 ** 2, axis=-1))
            scale_cluster_1 = cluster_1 / norm_cluster_1[:,None]
            scale_cluster_2 = cluster_2 / norm_cluster_2[:,None]
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
        print("kernel check not implemented")

    @abstractmethod
    def heuristic(self, sim_kern, axis=1):
        print("heuristic not implemented")

    @abstractmethod
    def calculate_constraint(self, sim_kernel):
        print("calculate_constraint not implemented")


    def enforce(self):
        
        n_clusters          = len(self.clusters)
        cluster_removal_idx = {i: set() for i in range(n_clusters)}

        # Mark idx that should be removed
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                cluster_1_np    = self.clusters[i].numpy_cluster
                cluster_2_np    = self.clusters[j].numpy_cluster
                sim_kern        = self.compute_sim_kern(cluster_1_np, cluster_2_np)
                remove_idxes    = self._selection(sim_kern)
                remove_1_idx    = remove_idxes[0]
                remove_2_idx    = remove_idxes[1]
                cluster_removal_idx[i].update(remove_1_idx)
                cluster_removal_idx[j].update(remove_2_idx)
        # For each cluster, keep only that which isn't slated for removal
        removed_any = False
        for i in range(n_clusters):
            removed_any                     = removed_any or len(cluster_removal_idx[i]) > 0
            keep_idx                        = np.arange(self.clusters[i].numpy_cluster.shape[0])
            keep_idx                        = keep_idx[~np.isin(keep_idx, list(cluster_removal_idx[i]))]
            self.clusters[i].numpy_cluster  = self.clusters[i].numpy_cluster[keep_idx]

        return not removed_any
    
    def _selection(self, sim_kern):
        """
            Description:
                selecting points that don't conform the constraint
                alternate removing from cluster 1 and 2
        """
        remove_from_cluster_1 = True
        cluster1_len = sim_kern.shape[0]
        cluster2_len = sim_kern.shape[1]

        cluster1_good_idx = list(range(cluster1_len))
        cluster2_good_idx = list(range(cluster2_len))

        while not np.all(self.kernel_check(sim_kern)):
            if remove_from_cluster_1:
                removal_idx = self.heuristic(sim_kern)
                cluster1_good_idx.pop(removal_idx)
                sim_kern = np.delete(sim_kern, removal_idx, 0)
                if self.intra_cluster:
                    sim_kern = np.delete(sim_kern, removal_idx, 1)
            else:
                removal_idx = self.heuristic(sim_kern.T)
                cluster2_good_idx.pop(removal_idx)
                sim_kern = np.delete(sim_kern, removal_idx, 1)
            
            if remove_from_cluster_1 and not self.intra_cluster:
                remove_from_cluster_1 = False
            elif not remove_from_cluster_1 and not self.intra_cluster:
                remove_from_cluster_1 = True
        
        cluster1_idx = np.arange(cluster1_len)
        cluster2_idx = np.arange(cluster2_len)

        cluster1_removal_idx = cluster1_idx[~np.isin(cluster1_idx, cluster1_good_idx)]
        cluster2_removal_idx = cluster2_idx[~np.isin(cluster2_idx, cluster2_good_idx)]
        return cluster1_removal_idx, cluster2_removal_idx
                