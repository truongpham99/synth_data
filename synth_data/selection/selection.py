import numpy as np

def random_selection(clusters, k):
    all_points = np.concatenate(clusters, axis=0)
    cluster_by_size = [len(cluster) for cluster in clusters]

    selected_idx = np.random.choice(all_points.shape[0], k, replace=False)

    selected_point_clusters = [[] for _ in range(len(clusters))]
    for si in selected_idx:
        for i in range(len(cluster_by_size)):
            if si < sum(cluster_by_size[:i+1]):
                selected_point_clusters[i].append(all_points[si,:].tolist())
                break
    
    selected_point_clusters = [np.array(c) for c in selected_point_clusters]

    return selected_point_clusters


