from abc import ABC, abstractmethod

import numpy as np

class Cluster:

    def __init__(self, num_points, dimensionality):
        
        self.num_points     = num_points
        self.dimensionality = dimensionality
        self.numpy_cluster  = None

    
    def draw(self):
        
        current_cluster_size    = self.numpy_cluster.shape[0] if self.numpy_cluster is not None else 0
        to_sample               = self.num_points - current_cluster_size
        if to_sample > 0:
            sampled_numpy_array = self.sample(self.num_points - current_cluster_size)
            self.numpy_cluster  = np.concatenate([self.numpy_cluster, sampled_numpy_array], axis=0) if self.numpy_cluster is not None else sampled_numpy_array


    @abstractmethod
    def sample(self, num_to_sample):
        pass