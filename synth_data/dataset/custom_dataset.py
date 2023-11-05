from .dataset import Dataset
from ..cluster import ClusterFactory
from ..constraint import ConstraintFactory

class CustomDataset(Dataset):
    
    def __init__(self, cluster_configs, constraint_configs, dimensionality=2, patience=100):
        
        super().__init__(dimensionality, patience) 

        cluster_factory = ClusterFactory()
        self.clusters   = []
        for cluster_config in cluster_configs:
            cluster_config["dimensionality"]    = dimensionality
            formed_cluster                      = cluster_factory.get_cluster(cluster_config)
            self.clusters.append(formed_cluster)

        constraint_factory = ConstraintFactory()
        self.constraints = []
        for constraint_config in constraint_configs:
            formed_constraint = constraint_factory.get_constraint(constraint_config, self.clusters)
            self.constraints.append(formed_constraint)
        