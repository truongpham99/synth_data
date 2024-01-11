import numpy as np 
from abc import ABC, abstractmethod
from ..constraint import ConstraintFactory
from ..constraint import MaximumSimilarityUpperBoundConstraint as UBC
from ..constraint import MinimumSimilarityLowerBoundConstraint as LBC
from ..constraint import MaximumSimilarityLowerBoundConstraint as MLBC
from ..selection.selection import *
import matplotlib
from ..utils import *         
from submodlib import FacilityLocationVariantMutualInformationFunction as FLQMI
from submodlib import FacilityLocationMutualInformationFunction as FLVMI

def flqmi_bounds(ground_set, query_clusters, rare_cluster_idx, common_cluster_idx, k, metric="rbf_1", epochs=10):
    """
    Inputs:
        ground_set and query_clusters are list of clusters. 

    Description:
        selecting greedy set once and random selection k times to calculate bounds
    """

    common_clusters = [ground_set[i] for i in common_cluster_idx]
    rare_clusters = [ground_set[i] for i in rare_cluster_idx]

    # union of the clusters
    union_query_cluster = np.concatenate(query_clusters)
    union_common_cluster = np.concatenate(common_clusters)
    union_rare_cluster = np.concatenate(rare_clusters)
    
    query_size = union_query_cluster.shape[0]
    
    ubc = UBC(None, metric, [0,1], [0,1])
    q_c_sim_kern = ubc.compute_sim_kern(union_query_cluster, union_common_cluster)
    epsilon_1 = ubc.calculate_constraint(q_c_sim_kern)

    lbc = LBC(None, metric, [0,1], [0,1])
    q_r_sim_kern = lbc.compute_sim_kern(union_query_cluster, union_rare_cluster)
    epsilon_2 = 1 - lbc.calculate_constraint(q_r_sim_kern)
    # print(q_r_sim_kern)
    print(epsilon_1, epsilon_2)

    # submod selection
    submod_select_clusters, submod_selected_idx_clusters, flqmi = flqmi_select(ground_set, query_clusters, k)

    union_submod_select_clusters = [submod_select_clusters[i] for i in range(len(submod_select_clusters)) if 
                                    len(submod_select_clusters[i]) > 0]
    union_submod_select_clusters = np.concatenate(union_submod_select_clusters)

    eval_submod_selection = flqmi.evaluate(set(submod_selected_idx_clusters))

    # eval_submod_selection = FacilityLocationQMILoss(union_submod_select_clusters,
    #                                                 union_query_cluster)
    # print("submod selection from submodlib: ",eval_submod_selection_1)
    # print("submod selection from pytorch",eval_submod_selection)

    upper_bounds, lower_bounds, deltas, delta_bounds = [], [], [], []

    for _ in range(epochs):
        random_select_clusters, random_selected_idx_clusters = random_selection(ground_set, k)
        random_selection_in_rare = [random_select_clusters[i] for i in rare_cluster_idx]
        chi = np.concatenate(random_selection_in_rare).shape[0]

        union_random_select_clusters = [random_select_clusters[i] for i in range(len(random_select_clusters)) if 
                                        len(random_select_clusters[i]) > 0]
        union_random_select_clusters = np.concatenate(union_random_select_clusters)

        eval_random_selection = flqmi.evaluate(set(random_selected_idx_clusters))
        # eval_random_selection = FacilityLocationQMILoss(union_random_select_clusters,
        #                                                 union_query_cluster)
        
        # print("random selection from submodlib: ",eval_random_selection_1)
        # print("random selection from pytorch",eval_random_selection)

        submod_selection_in_rare = np.concatenate([submod_select_clusters[i] for i in rare_cluster_idx])
        deltas.append(submod_selection_in_rare.shape[0] - chi)

        gamma = eval_submod_selection - eval_random_selection

        t = ((k - chi)*epsilon_1 + (query_size + chi)*epsilon_2)/(1-epsilon_2)
        delta_bounds.append(t)

        lb = (gamma - (k - chi)*epsilon_1 - (query_size + chi)*epsilon_2)/(1-epsilon_2)
        lower_bounds.append(lb)
        
        ub = (gamma + (k - chi)*epsilon_1 + (query_size + chi)*epsilon_2)/(1-epsilon_2)
        upper_bounds.append(ub)

    # print(eval_submod_selection - eval_random_selection)
    # print(eval_submod_selection_1 - eval_random_selection_1)
        
    # print(deltas, delta_bounds)

    # as_in_r = submod_selection_in_rare.shape[0] 

    # eval_lower_bound = as_in_r*(1-epsilon_2)+query_size*(1-epsilon_2)

    # print()
    # # print("value: ", eval_submod_selection)
    # print("A star lower: ", eval_lower_bound)

    # a_in_t = np.concatenate([random_select_clusters[i] for i in common_cluster_idx]).shape[0]
    # eval_upper_bound = chi + a_in_t*epsilon_1 + query_size

    # print("value: ", eval_random_selection_1)
    # print("A upper: ", eval_upper_bound)
    # print(eval_lower_bound  - eval_upper_bound)
        
    return upper_bounds, lower_bounds, deltas, delta_bounds


# import torch
# import torch.nn.functional as F
# def gaussian_sim(a, b, sigma=1):
#     # a_norm = F.normalize(a)
#     # b_norm = F.normalize(b)

#     dist_mat = torch.cdist(a, b)
#     sim_mat = torch.exp(-dist_mat/ (2*sigma**2))
#     return sim_mat

# # def mahalnobis_sim(a,b):
# #     cov = torch.cov(b)
# #     delta = a - b
# #     m = torch.matmul(delta, torch.matmul(torch.inverse(cov), delta))
# #     return -torch.sqrt(m)


# def FacilityLocationQMILoss(A, B, eta = 1.0, sim_matrix=gaussian_sim):
#     ubc = UBC(None, "rbf_1", [0,1], [0,1])
#     sim_mat = ubc.compute_sim_kern(A, B)
#     sim_mat = torch.tensor(sim_mat)
#     result = 0
#     sims = []
#     for i, a in enumerate(A):
#         val = torch.max(sim_mat[i,:])
#         result += val
#         sims.append(val.item())
#     # print(sims)

#     sims = []
#     for j, b in enumerate(B):
#         val = eta*torch.max(sim_mat[:,j])
#         result += val
#         sims.append(val.item())
#     # print(sims)
#     return result
