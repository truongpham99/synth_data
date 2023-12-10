import numpy as np 
from abc import ABC, abstractmethod
from ..constraint import ConstraintFactory

class Bounds(ABC):
    def __init__(self, full_data, random_subset, selected_subset, constraint_configs):
        self.full_data = full_data
        self.random_subset = random_subset
        self.selected_subset = selected_subset

        constraint_factory = ConstraintFactory()
        self.constraints = []
        self.similarity_bounds = {}
        for constraint_config in constraint_configs:
            idx1, idx2 = constraint_config["cluster_idxs"]
            cluster1 = self.full_data[idx1]
            cluster2 = self.full_data[idx2]

            constraint_type = constraint_config["type"]

            formed_constraint = constraint_factory.get_constraint(constraint_config, self.full_data)
            self.constraints.append(formed_constraint)
            sim_kern = formed_constraint.compute_sim_kern(cluster1, cluster2)
            calculated_sim_constraint = formed_constraint.calculate_constraint(sim_kern)
            
            if idx1 not in self.similarity_bounds:
                self.similarity_bounds[idx1] = {}
            if idx2 not in self.similarity_bounds[idx1]:
                self.similarity_bounds[idx1][idx2] = {}

            self.similarity_bounds[idx1][idx2][constraint_type] = calculated_sim_constraint
        
        print(self.similarity_bounds)

        

        