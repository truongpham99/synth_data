from .single_sim_constraint import SingleSimilarityConstraint

import numpy as np

class MinimumSimilarityLowerBoundConstraint(SingleSimilarityConstraint):

    def __init__(self, min_similarity, metric, cluster_idxs, list_of_clusters):
        
        super().__init__(min_similarity, metric, cluster_idxs, list_of_clusters)


    def kernel_check(self, sim_kern):
        return sim_kern >= self.threshold
    
    def heuristic(self, sim_kern):
        row_sum = np.sum(sim_kern, axis=1)
        remove_idx = np.argmin(row_sum)
        return remove_idx
    
    def calculate_constraint(self, sim_kernel):
        return sim_kernel.min()

    def __str__(self) -> str:
        return f"MinSimLB({self.cluster_idxs})"
    
class MaximumSimilarityLowerBoundConstraint(SingleSimilarityConstraint):

    def __init__(self, min_similarity, metric, cluster_idxs, list_of_clusters):
        
        super().__init__(min_similarity, metric, cluster_idxs, list_of_clusters)


    def kernel_check(self, sim_kern):
        return sim_kern >= self.threshold
        
    def calculate_constraint(self, sim_kernel):
        return sim_kernel.max(axis=0).min()
    
    def heuristic(self, sim_kern):
        print("{} heuristic not implemented".format(self.__str__))
    
    def __str__(self) -> str:
        return f"MinSimLB({self.cluster_idxs})"
    
