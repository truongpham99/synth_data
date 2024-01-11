import numpy as np
from ..utils import *
from submodlib import FacilityLocationVariantMutualInformationFunction as FLQMI
from submodlib import FacilityLocationMutualInformationFunction as FLVMI

def random_selection(clusters, k):
    all_points = np.concatenate(clusters, axis=0)
    selected_idx = np.random.choice(all_points.shape[0], k, replace=False)

    selected_point_clusters, _ = selected_points_by_original_format(clusters, selected_idx)

    return selected_point_clusters, list(selected_idx.astype(int))

def flqmi_select(full_data, query_cluster, k, metric="euclidean"):
    full_data_len = []
    [full_data_len.append(c.shape[0]) for c in full_data]
    full_data_len = sum(full_data_len)

    query_len = []
    [query_len.append(c.shape[0]) for c in query_cluster]
    query_len = sum(query_len)

    full_data_list = np.concatenate(full_data)
    query_cluster_list = np.concatenate(query_cluster)
    obj = FLQMI(n=full_data_len,
                        num_queries=query_len,
                        data=full_data_list,
                        queryData=query_cluster_list,
                        metric=metric)
    greedyList = obj.maximize(budget=k, optimizer='LazyGreedy', stopIfZeroGain=False,
                          stopIfNegativeGain=False, verbose=False)

    selected_idx = list(np.array(greedyList)[:, 0].astype(int))
    selected_point_clusters, _ = selected_points_by_original_format(full_data, selected_idx)

    return selected_point_clusters, selected_idx, obj


