from .single_sim_constraint import SingleSimilarityConstraint

import numpy as np

class MaximumSimilarityUpperBoundConstraint(SingleSimilarityConstraint):

    def __init__(self, max_similarity, metric, cluster_idxs, list_of_clusters):
        
        super().__init__(max_similarity, metric, cluster_idxs, list_of_clusters)


    def kernel_check(self, sim_kern):
        return sim_kern <= self.threshold
    
    def heuristic(self, sim_kern, axis=1):
        row_sum = np.sum(sim_kern, axis=axis)
        remove_idx = np.argmax(row_sum)
        return remove_idx
    
    def calculate_constraint(self, sim_kernel):
        return sim_kernel.max()

    
    def __str__(self) -> str:
        return f"MaxSimUB({self.cluster_idxs})"