from .mslb_constraint import MinimumSimilarityLowerBoundConstraint
from .msub_constraint import MaximumSimilarityUpperBoundConstraint

class ConstraintFactory:

    def __init__(self):

        self.constraint_type_mapping = {"min_similarity": MinimumSimilarityLowerBoundConstraint,
                                        "max_similarity": MaximumSimilarityUpperBoundConstraint}


    def get_constraint(self, constraint_config, list_of_clusters):
        
        constraint_name = constraint_config["type"]
        constraint_type = self.constraint_type_mapping[constraint_name]
        del constraint_config["type"]

        created_constraint = constraint_type(**constraint_config, list_of_clusters=list_of_clusters)
        return created_constraint