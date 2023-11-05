from .single_sim_constraint import SingleSimilarityConstraint

import numpy as np

class MinimumSimilarityLowerBoundConstraint(SingleSimilarityConstraint):

    def __init__(self, min_similarity, metric, cluster_idxs, list_of_clusters):
        
        super().__init__(min_similarity, metric, cluster_idxs, list_of_clusters)


    def kernel_check(self, sim_kern):
        return sim_kern >= self.threshold