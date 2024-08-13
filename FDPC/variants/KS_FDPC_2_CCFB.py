# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KDTree

from FDPC import get_varience
from FDPC.get_varience import read_data
from mesure.comment import balance, compute_SC


class KS_FDPC:
    def __init__(self, X, f_attr_vals, k=5, C=3):
        self.X = X

        self.k = k
        self.C = C
        self.fair_attr_vals = f_attr_vals
        # 各个敏感属性值在数据集中的占比
        self.fattr_val_ratio = get_varience.cal_fair_attr_val_ratio(self)
        self.n, self.m = X.shape
        self.dist_matrix = None
        self.rho = None
        self.delta = None
        self.sub_clusters = None
        self.labels_ = None
        self.peaks = None
        self.cluster_num = cluster_num
        self.dataset_cluster = X

    def compute_2KNN(self):
        tree = KDTree(self.X)
        _, ind = tree.query(self.X, k=self.k * 2 + 1)
        return ind[:, 1:]

    def compute_DSKNN(self, distances, indices):
        DS_knn = [set() for _ in range(self.n)]
        dislevel = np.zeros(self.n)

        for i in range(self.n):
            for j in indices[i]:
                if i in indices[j]:
                    DS_knn[i].add(j)

        for i in range(self.n):
            if 0 < len(DS_knn[i]) < self.k:
                dislevel[i] = max([distances[i, indices[i].tolist().index(j)] for j in DS_knn[i]])
            elif len(DS_knn[i]) >= self.k:
                dislevel[i] = max(
                    [distances[i, indices[i].tolist().index(j)] for j in DS_knn[i].intersection(indices[i])])
            else:
                dislevel[i] = np.inf

        return DS_knn, dislevel

    def compute_IMKNN(self, distances, indices, dislevel):
        IMKNN = [set() for _ in range(self.n)]

        for i in range(self.n):
            for j in indices[i]:
                if distances[i, indices[i].tolist().index(j)] < min(dislevel[i], dislevel[j]):
                    IMKNN[i].add(j)

        return IMKNN

    def compute_local_density_and_distance(self, IMKNN):
        rho = np.zeros(self.n)
        delta = np.full(self.n, np.inf)

        for i in range(self.n):
            IM_knn_i = np.array(list(IMKNN[i]))
            if len(IM_knn_i) > 0 and np.sum(self.dist_matrix[i, IM_knn_i]) > 0:
                rho[i] = len(IM_knn_i) / np.power(np.sum(self.dist_matrix[i, IM_knn_i]), 1.5)
            else:
                rho[i] = 0

        for i in range(self.n):
            higher_density = np.where(rho > rho[i])[0]
            if len(higher_density) > 0:
                delta[i] = np.min(self.dist_matrix[i, higher_density])
            else:
                delta[i] = np.max(self.dist_matrix[i])

        return rho, delta

    def initial_sub_clusters(self, rho, delta, IMKNN):
        peaks = []
        sub_clusters = -1 * np.ones(self.n, dtype=int)

        # 找到局部中心
        for i in range(self.n):
            if all(rho[i] > rho[j] for j in IMKNN[i]):
                peaks.append(i)
                sub_clusters[i] = len(peaks) - 1

        # 分配非中心点到最近的中心点
        for i in range(self.n):
            if sub_clusters[i] == -1:
                closest_peak = np.argmin([self.dist_matrix[i, peak] for peak in peaks])
                sub_clusters[i] = sub_clusters[peaks[closest_peak]]

        peaks = np.array(peaks)

        return sub_clusters, peaks

    def compute_similarity(self, sub_clusters):
        n_clusters = len(np.unique(sub_clusters))
        SIM = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                points_i = np.where(sub_clusters == i)[0]
                points_j = np.where(sub_clusters == j)[0]
                intersection = [(p, q) for p in points_i for q in points_j]

                if intersection:
                    M = len(intersection)
                    CON = np.sum([1 / self.dist_matrix[p, q] for p, q in intersection]) / np.sqrt(M)
                    #
                    Pavg_i = np.sum(self.rho[points_i]) / len(points_i)
                    Pavg_j = np.sum(self.rho[points_j]) / len(points_j)
                    Pavg_ij = 2 * np.sqrt(Pavg_i * Pavg_j) / (Pavg_i + Pavg_j)
                    # 合并后所有点的密度标准差
                    combined_points = np.concatenate((points_i, points_j))
                    combined_densities = self.rho[combined_points]
                    Std_ij = np.std(combined_densities)
                    STD = Pavg_ij / Std_ij

                    SIM[i, j] = CON * Pavg_ij + Pavg_ij * STD + STD * CON
                    SIM[j, i] = SIM[i, j]

        # print(f"Similarity matrix: \n{SIM}")  # 调试信息
        return SIM

    def merge_sub_clusters(self, sub_clusters, SIM):
        while len(np.unique(sub_clusters)) > self.C:
            max_sim = np.max(SIM)
            if max_sim == 0:
                break
            i, j = np.unravel_index(np.argmax(SIM), SIM.shape)
            sub_clusters[sub_clusters == j] = i
            SIM[i, :] = 0
            SIM[:, i] = 0
            SIM[j, :] = 0
            SIM[:, j] = 0

            # print(f"Clusters merged: {i}, {j}")  # 调试信息

        return sub_clusters

    def final_merge(self, sub_clusters):
        unique_clusters = np.unique(sub_clusters)
        if len(unique_clusters) > self.C:
            cluster_sizes = {cluster: np.sum(sub_clusters == cluster) for cluster in unique_clusters}
            sorted_clusters = sorted(cluster_sizes.items(), key=lambda item: item[1], reverse=True)

            retained_clusters = [cluster for cluster, size in sorted_clusters[:self.C]]
            merged_clusters = [cluster for cluster, size in sorted_clusters[self.C:]]

            for cluster in merged_clusters:
                similar_clusters = []
                for retained in retained_clusters:
                    points_cluster = np.where(sub_clusters == cluster)[0]
                    points_retained = np.where(sub_clusters == retained)[0]

                    Pavg_cluster = np.sum(self.rho[points_cluster]) / len(points_cluster)
                    Pavg_retained = np.sum(self.rho[points_retained]) / len(points_retained)
                    Pavg_ij = min(Pavg_cluster, Pavg_retained) / max(Pavg_cluster, Pavg_retained)

                    Std_cluster = np.std(self.rho[points_cluster])
                    Std_retained = np.std(self.rho[points_retained])
                    Std_ij = min(Std_cluster, Std_retained) / max(Std_cluster, Std_retained)

                    similarity = Pavg_ij * Std_ij / np.std(
                        np.concatenate((self.rho[points_cluster], self.rho[points_retained])))
                    similar_clusters.append((retained, similarity))

                best_match = max(similar_clusters, key=lambda item: item[1])
                sub_clusters[sub_clusters == cluster] = best_match[0]

        return sub_clusters

    def fit(self):
        indices = self.compute_2KNN()
        self.dist_matrix = squareform(pdist(self.X, metric='euclidean'))
        # print(self.dist_matrix)
        DS_knn, dislevel = self.compute_DSKNN(self.dist_matrix, indices)

        IMKNN = self.compute_IMKNN(self.dist_matrix, indices, dislevel)
        self.rho, self.delta = self.compute_local_density_and_distance(IMKNN)
        # 先完成初始子簇的初始化，不考虑合并
        self.sub_clusters, self.peaks = self.initial_sub_clusters(self.rho, self.delta, IMKNN)
        # 再需要分别计算三个指标，并作为该类的属性，暂时不考虑后续的合并
        SIM = self.compute_similarity(self.sub_clusters)
        self.labels_ = self.merge_sub_clusters(self.sub_clusters, SIM)
        self.labels_ = self.final_merge(self.labels_)
        self.group_label = self.labels_
        # print(f"Final labels: {self.labels_}")  # 调试信息

    def visualize_clusters(self, title="Clustering Results"):
        X = self.X
        if X.shape[1] > 2:
            X = PCA(n_components=2).fit_transform(X)
        unique_labels = np.unique(self.labels_)
        colors = plt.cm.get_cmap("tab10", len(unique_labels))

        plt.figure(figsize=(10, 6))
        for i, label in enumerate(unique_labels):
            plt.scatter(X[self.labels_ == label, 0], X[self.labels_ == label, 1], c=[colors(i)],
                        label=f'Cluster {label}')
        # plt.scatter(X[self.peaks, 0], X[self.peaks, 1], c='red', marker='x', s=200, label='Peaks')
        plt.title(title)
        plt.legend()
        plt.show()

    def evaluate(self, data_name, k):

        self.balance_val = balance(self)
        self.sc = compute_SC(self)

            # Log to Excel
        self.log_to_excel(data_name, k)

    def log_to_excel(self, data_name, k):
        # Prepare data for Excel log
        data = {
            'Data Name': [data_name],
            'Cluster Count': [self.cluster_num],
            'K': [k],
            'SC Value': [self.sc],
            'Balance': [self.balance_val],
        }

        # Create or append to the Excel file
        excel_file = f"{data_name}_KS_FDPC.xlsx"
        try:
            df = pd.read_excel(excel_file)
            df = df.append(pd.DataFrame(data), ignore_index=True)
        except FileNotFoundError:
            df = pd.DataFrame(data)

        # Write to Excel
        df.to_excel(excel_file, index=False)


all_datasets = ['obesity']

# 'Wholesale', 'hcvdat0',  'obesity',
#                 'drug_consumption', 'Room_Occupancy_Estimation',
#                 'liver_disorder', 'Rice',
#                 'adult', 'bank', 'dermatology'

for data in all_datasets:
    for cluster_num in range(4, 5):
        for k in range(5, 6):
            cluster_attr_vals, f_attr_vals, f_attr = read_data(data)
            model = KS_FDPC(cluster_attr_vals, f_attr_vals=f_attr_vals,
                            k=k, C=cluster_num)
            model.fit()
            # 可视化聚类结果
            model.visualize_clusters(cluster_attr_vals)
            # model.evaluate(data, k)
