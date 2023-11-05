from .cluster import Cluster

import numpy as np

class GaussianCluster(Cluster):

    def __init__(self, mean_vector, covariance_matrix, num_points, dimensionality):
        
        super().__init__(num_points, dimensionality)

        self.mean_vector        = mean_vector
        self.covariance_matrix  = covariance_matrix 
        

    def sample(self, num_to_sample):
        
        return np.random.multivariate_normal(self.mean_vector, self.covariance_matrix, size=num_to_sample)