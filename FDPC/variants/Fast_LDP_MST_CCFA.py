# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
from sklearn.datasets import load_iris
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import minimum_spanning_tree
import time

def fast_ldp_mst(data, num_clusters, min_cluster_size, num_neighbors, knn_method='kd_tree'):
    """
    Fast LDP-MST clustering algorithm.
    Input:
        data: numpy array, shape (n_samples, n_features), input dataset.
        num_clusters: int, number of clusters.
        min_cluster_size: int, minimum cluster size.
        num_neighbors: int, number of nearest neighbors.
        knn_method: str, method for nearest neighbors search ('kd_tree' or 'auto').
    Output:
        cluster_labels: numpy array, shape (n_samples,), cluster labels.
        total_time: float, total time taken by the algorithm.
    """
    num_samples, num_features = data.shape  # 获取数据集的样本数量和特征维度
    if min_cluster_size is None:
        min_cluster_size = int(0.01 * num_samples)  # 默认最小聚类大小为样本数的1%
    if num_neighbors is None:
        num_neighbors = int(np.ceil(np.log2(num_samples)))  # 默认k值为log2(N)的上限

    start_time = time.time()  # 记录开始时间

    # Step 1-4: Density and distance calculations to identify local density peaks
    neighbor_ids, local_density, peak_indices, cluster_labels = compute_local_density_peaks(data, knn_method, num_neighbors)

    root_samples = data[peak_indices, :]  # Root samples
    num_roots = len(peak_indices)

    if num_roots > num_clusters:
        cluster_distances_start_time = time.time()
        num_neighbors = neighbor_ids.shape[1]
        temp_data = np.column_stack((neighbor_ids.flatten(), cluster_labels[np.tile(np.arange(num_samples), num_neighbors)]))
        unique_pairs = np.unique(temp_data, axis=0)
        node_indices = unique_pairs[:, 0].astype(int)
        cluster_indices = unique_pairs[:, 1].astype(int)

        start_idx = np.zeros(num_samples, dtype=int)
        end_idx = np.zeros(num_samples, dtype=int)
        cluster_size = np.zeros(num_samples, dtype=int)

        cluster_id = 0
        start_idx[cluster_id] = 0
        for t in range(1, len(node_indices)):
            if node_indices[t] != node_indices[t - 1]:
                end_idx[cluster_id] = t - 1
                cluster_size[cluster_id] = end_idx[cluster_id] - start_idx[cluster_id] + 1
                cluster_id += 1
                start_idx[cluster_id] = t
        end_idx[cluster_id] = len(node_indices) - 1
        cluster_size[cluster_id] = end_idx[cluster_id] - start_idx[cluster_id] + 1

        total_num_pairs = np.sum(cluster_size * (cluster_size - 1)) // 2
        pairs = np.zeros((total_num_pairs, 2), dtype=int)
        pair_densities = np.zeros(total_num_pairs)

        pair_index = 0
        for j in range(num_samples):
            if cluster_size[j] != 1:
                cluster_j = cluster_indices[start_idx[j]:end_idx[j] + 1]
                for s in range(cluster_size[j] - 1):
                    for t in range(s + 1, cluster_size[j]):
                        pairs[pair_index, 0] = cluster_j[s]
                        pairs[pair_index, 1] = cluster_j[t]
                        pair_densities[pair_index] = local_density[j]
                        pair_index += 1

        unique_diff_pairs, indices, pair_indices = np.unique(pairs, axis=0, return_index=True, return_inverse=True)
        node1 = unique_diff_pairs[:, 0]
        node2 = unique_diff_pairs[:, 1]

        # Debugging: Check max indices of node1 and node2
        print(f'Max node1 index: {np.max(node1)}, Max node2 index: {np.max(node2)}, root_samples size: {root_samples.shape[0]}')

        if np.max(node1) >= root_samples.shape[0] or np.max(node2) >= root_samples.shape[0]:
            raise IndexError("Index out of bounds for root_samples")

        num_non_zero_elements = len(node1)
        pair_weights = np.zeros(num_non_zero_elements)
        pair_counts = np.zeros(num_non_zero_elements)
        for t in range(len(pair_indices)):
            pair_weights[pair_indices[t]] += pair_densities[t]
            pair_counts[pair_indices[t]] += 1

        distance_values = np.zeros(num_non_zero_elements)
        for t in range(num_non_zero_elements):
            distance = np.sqrt(np.sum((root_samples[node1[t]] - root_samples[node2[t]]) ** 2))
            distance_values[t] = distance / pair_weights[t] / pair_counts[t]

        graph = nx.Graph()
        for i in range(num_non_zero_elements):
            graph.add_edge(node1[i], node2[i], weight=distance_values[i])
        minimum_spanning_tree_graph = nx.minimum_spanning_tree(graph)

        parent = np.zeros(num_roots, dtype=int)
        for i, component in enumerate(nx.connected_components(minimum_spanning_tree_graph)):
            for node in component:
                parent[node] = i
        edge_weights = np.zeros(num_roots)
        for i in range(num_roots):
            edge_weights[i] = minimum_spanning_tree_graph.edges[i, parent[i]]['weight'] if minimum_spanning_tree_graph.has_edge(i, parent[i]) else 0

        new_parent, new_root_indices = cut_edges(parent, edge_weights, cluster_labels, min_cluster_size, num_clusters)

        supernode_labels = np.zeros(num_roots, dtype=int)
        for i in range(num_roots):
            supernode_labels[i] = new_parent[new_root_indices[i]]
        cluster_labels = supernode_labels[cluster_labels]

    total_time = time.time() - start_time  # 计算总运行时间
    return cluster_labels, total_time

def compute_local_density_peaks(data, knn_method, num_neighbors):
    """
    Local Density Peaks (LDP) function to calculate density and distances.
    """
    num_samples, num_features = data.shape
    if knn_method == 'kd_tree':
        neighbors = NearestNeighbors(n_neighbors=num_neighbors, algorithm='kd_tree').fit(data)
    else:
        neighbors = NearestNeighbors(n_neighbors=num_neighbors, algorithm='auto').fit(data)
    distances, indices = neighbors.kneighbors(data)  # 计算每个点的K近邻

    local_density = np.zeros(num_samples)  # 局部密度
    min_distance = np.zeros(num_samples) + np.inf  # 最小距离
    for i in range(num_samples):
        local_density[i] = np.sum(distances[i, 1:])  # 计算局部密度

    for i in range(1, num_samples):
        for j in range(i):
            distance = np.linalg.norm(data[i] - data[j])  # 计算点之间的距离
            if local_density[j] > local_density[i] and distance < min_distance[i]:
                min_distance[i] = distance
            elif local_density[j] < local_density[i] and distance < min_distance[j]:
                min_distance[j] = distance

    gamma = local_density * min_distance  # 密度和距离的乘积
    sorted_indices = np.argsort(-gamma)  # 按gamma值降序排序

    peak_indices = sorted_indices[:num_neighbors]  # 选取前K个密度峰值点
    cluster_labels = np.zeros(num_samples, dtype=int)
    cluster_labels[peak_indices] = np.arange(1, num_neighbors + 1)

    return indices, local_density, peak_indices, cluster_labels

def cut_edges(parent, edge_weights, cluster_labels, min_cluster_size, num_clusters):
    """
    Function to cut edges in the MST to form desired number of clusters.
    """
    num_roots = len(edge_weights)
    num_samples = len(cluster_labels)
    node_weights = np.zeros(num_roots)
    for i in range(num_samples):
        node_weights[cluster_labels[i]] += 1  # 计算每个节点的权重

    in_neighbors = [[] for _ in range(num_roots)]  # 存储每个节点的入度邻居
    for i in range(num_roots):
        if parent[i] != i:
            in_neighbors[parent[i]].append(i)

    node_indices = np.zeros(num_roots, dtype=int)
    current_index = 0
    for j in range(num_roots):
        if parent[j] == j:
            node_indices[current_index] = j
            current_index += 1

    for j in range(num_roots - 1, -1, -1):
        i = node_indices[j]
        if parent[i] != i:
            node_weights[parent[i]] += node_weights[i]

    sorted_edge_indices = np.argsort(-edge_weights)  # 按边权重降序排序
    num_edges_to_cut = 0
    t = 0
    initial_num_roots = np.sum(parent == np.arange(num_roots))
    num_of_edges_required_to_remove = max(num_clusters - initial_num_roots, 0)

    while num_edges_to_cut != num_of_edges_required_to_remove and t < len(sorted_edge_indices):
        start_node = sorted_edge_indices[t]
        end_node = parent[start_node]
        if node_weights[start_node] > min_cluster_size:
            passed_nodes = []
            while end_node != parent[end_node]:
                passed_nodes.append(end_node)
                end_node = parent[end_node]
            passed_nodes.append(end_node)

            root_node_reached = end_node
            if node_weights[root_node_reached] - node_weights[start_node] > min_cluster_size:
                for node in passed_nodes:
                    node_weights[node] -= node_weights[start_node]
                parent[start_node] = start_node
                num_edges_to_cut += 1
        t += 1

    new_parent = parent
    new_root_indices = np.where(new_parent == np.arange(num_roots))[0]  # 获取所有根节点的索引
    return new_parent, new_root_indices


def plot_clusters(data, labels):
    """
    Function to plot the clustered data.
    """
    plt.figure()
    unique_labels = np.unique(labels)  # 获取唯一的标签
    for label in unique_labels:
        cluster_data = data[labels == label]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {label}')
    plt.legend()
    plt.show()


# 演示代码
if __name__ == "__main__":
    iris = load_iris()
    data = iris.data

    labels, total_time = fast_ldp_mst(data, num_clusters=3, min_cluster_size=None, num_neighbors=None)
    plot_clusters(data, labels)
    print(f"Total time: {total_time:.2f} seconds")

