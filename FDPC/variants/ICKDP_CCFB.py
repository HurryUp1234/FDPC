# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import networkx as nx


def calculate_local_density(data, K):
    nbrs = NearestNeighbors(n_neighbors=K + 1).fit(data)
    distances, indices = nbrs.kneighbors(data)
    local_density = K / np.sum(distances[:, 1:], axis=1)
    return local_density, distances, indices


def calculate_Si(i, LC, local_density, indices, K):
    if len(set(indices[i, 1:K + 1]).intersection(LC)) == 0 and local_density[i] >= np.mean(
            local_density[indices[i, 1:K + 1]]):
        return 1
    return 0


def extract_local_centers(data, K):
    local_density, distances, indices = calculate_local_density(data, K)
    LC = set()
    LC.add(np.argmax(local_density))
    for i in range(len(data)):
        if calculate_Si(i, LC, local_density, indices, K):
            LC.add(i)
    return LC, local_density, distances, indices


def calculate_ICK_distance(data, LC, local_density):
    """
    Calculate the Improved Connectivity Kernel (ICK) distance between local centers.
    """
    LC = list(LC)
    num_centers = len(LC)
    ICK_d = np.zeros((num_centers, num_centers))
    # print(LC)
    for i in range(num_centers):
        for j in range(i + 1, num_centers):
            pt1 = LC[i]
            pt2 = LC[j]
            path = [pt1]
            # 使用列表推导式找到所有符合条件的中间点，并添加到路径中
            intermediate_points = [pt for pt in LC if pt not in path and local_density[pt] < local_density[pt2]]
            path.extend(intermediate_points)
            path.append(pt2)
            # print(f'path: {path}')
            distances = [np.linalg.norm(data[path[k]] - data[path[k + 1]]) for k in range(len(path) - 1)]
            ICK_d[i, j] = max(distances)
            ICK_d[j, i] = ICK_d[i, j]

    return ICK_d

def calculate_relative_distance(local_density, ICK_distance, LC):
    LC_array = np.array(list(LC))
    relative_distance = np.zeros_like(local_density[LC_array])
    sorted_density_indices = np.argsort(-local_density[LC_array])
    # print(f"sorted_density_indices: {sorted_density_indices}")
    for i in range(1, len(sorted_density_indices)):
        relative_distance[sorted_density_indices[i]] = np.max(ICK_distance[sorted_density_indices[i], :])
    relative_distance[sorted_density_indices[0]] = max(relative_distance)
    relative_distance_dict = {LC_array[i]: relative_distance[i] for i in range(len(LC_array))}
    # print(relative_distance_dict)
    return relative_distance_dict
def ICKDP(data, K, cluster_num=2):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    LC, local_density, distances, indices = extract_local_centers(data, K)
    # print(f'Local centers: {LC}')
    ICK_distance = calculate_ICK_distance(data, LC, local_density)
    relative_distance = calculate_relative_distance(local_density, ICK_distance, LC)

    sorted_LC = sorted(relative_distance, key=relative_distance.get, reverse=True)
    cluster_centers = sorted_LC[:cluster_num]
    cluster_labels = -np.ones(len(data))
    for i, center in enumerate(cluster_centers):
        cluster_labels[center] = i

    sorted_indices = np.argsort(-local_density)
    for i in sorted_indices:
        if cluster_labels[i] == -1:
            neighbors = indices[i, 1:K + 1]
            neighbors = [neighbor for neighbor in neighbors if local_density[neighbor] > local_density[i]]
            if neighbors:
                nearest_center = neighbors[np.argmax(local_density[neighbors])]
                cluster_labels[i] = cluster_labels[nearest_center]

    print(cluster_labels)
    return cluster_labels, local_density, relative_distance, cluster_centers


def plot_clusters(data, labels, cluster_centers):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(data[cluster_centers, 0], data[cluster_centers, 1], s=300, c='red', marker='*')
    plt.show()


# 演示代码
if __name__ == "__main__":
    iris = load_iris()
    data = iris.data

    labels, local_density, relative_distance, cluster_centers = ICKDP(data, K=5)
    plot_clusters(data, labels, cluster_centers)
    print("Cluster Centers:", cluster_centers)
