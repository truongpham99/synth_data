from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

class Dataset(ABC):

    @abstractmethod
    def __init__(self, dimensionality=2, patience=100):
        self.dimensionality = dimensionality 
        self.patience       = patience


    def draw(self):
        
        validated           = False
        patience_counter    = 0

        while not validated and patience_counter < self.patience:
            for cluster in self.clusters:
                cluster.draw()
            validated           = self.validate_clusters()
            patience_counter    = patience_counter + 1

        if patience_counter == self.patience:
            raise ValueError(F"Failed to generate data after {self.patience} attempts!")

        # for cluster in self.clusters:
        #     cluster.draw()
        # _ = self.validate_clusters()


    def validate_clusters(self):

        validated = True
        for constraint in self.constraints:
            passed_enforcement  = constraint.enforce()
            validated           = validated and passed_enforcement
        return validated

    
    def get_numpy_dict(self):
        
        dataset_dictionary          = {}
        dataset_dictionary["data"]  = [cluster.numpy_cluster for cluster in self.clusters]
        return dataset_dictionary