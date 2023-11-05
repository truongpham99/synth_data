from .gaussian_cluster import GaussianCluster

class ClusterFactory:

    def __init__(self):
        
        self.cluster_type_mapping = {"gaussian": GaussianCluster}

    
    def get_cluster(self, cluster_config):
        
        cluster_name                        = cluster_config["type"]
        cluster_type                        = self.cluster_type_mapping[cluster_name]
        del cluster_config["type"]

        created_cluster = cluster_type(**cluster_config)
        return created_cluster