from abc import ABC, abstractmethod

class Constraint(ABC):

    def __init__(self, list_of_clusters):
        self.clusters = list_of_clusters

    
    @abstractmethod
    def enforce(self):
        pass 