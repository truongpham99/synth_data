import numpy as np

def selected_points_by_original_format(data, selected_idx):
    all_points = np.concatenate(data, axis=0)
    cluster_by_size = [len(cluster) for cluster in data]

    selected_point_clusters = [[] for _ in range(len(data))]
    selected_idx_clusters = [[] for _ in range(len(data))]

    for si in selected_idx:
        for i in range(len(cluster_by_size)):
            if si < sum(cluster_by_size[:i+1]):
                selected_point_clusters[i].append(all_points[si,:].tolist())
                selected_idx_clusters[i].append(si-sum(cluster_by_size[:i]))
                break
    
    selected_point_clusters = [np.array(c) for c in selected_point_clusters]
    selected_idx_clusters = [np.array(c) for c in selected_idx_clusters]

    return selected_point_clusters, selected_idx_clusters