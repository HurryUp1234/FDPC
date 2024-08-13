import numpy as np
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from FDPC import get_varience
from FDPC import remain_cluster
import networkx as nx

# 中心点选取的fair_dc限制和分配策略加入公平约束
class FDPC5:
    def __init__(self, cluster_attr_vals, f_attr_vals, f_attr, data_name,
                 dc_rate, fair_dc_min_rate, fair_dc_max_rate,cluster_num=3):
        self.data_name = data_name
        self.dataset_cluster = cluster_attr_vals
        self.fair_attr_vals = f_attr_vals
        self.fair_attr = f_attr
        # 读数据库名字得到数据集，单属性所有敏感属性组，数据集各敏感属性组的占比
        # 距离矩阵(马氏距离)
        # cov_matrix = np.cov(self.dataset_cluster.T)
        # self.inv_cov_matrix = np.linalg.inv(cov_matrix)
        # self.dist_matrix = get_varience.compute_mahalanobis_distance_matrix(self)
        self.dist_matrix = squareform(pdist(self.dataset_cluster, metric='euclidean'))
        # 各个敏感属性组占比
        self.fattr_val_ratio = get_varience.cal_fair_attr_val_ratio(self)

        # dc用于密度计算
        self.dc = get_varience.select_dc_orin(self, dc_rate)
        # 各点局部密度(原密度),默认用截断核, 可用 Gaussion
        self.rho = get_varience.get_local_density(self, method="Gaussion")
        # 各点的相对距离
        self.deltas, self.nearest_neiber = get_varience.get_deltas(self)

        # fair_dc用于中心点间的距离约束
        # self.fair_dc = get_varience.select_dc_lag(self, dc_min=fair_dc_min_rate, dc_max=fair_dc_max_rate)

        self.fair_dc = get_varience.select_dc_lag(self, fair_dc_min_rate, fair_dc_max_rate)

        # 选取聚类中心(改动)
        self.group_centers, self.wasted_centers = get_varience.find_centers_fairDc(self, cluster_num)
        # 考虑公平约束
        self.group_label = remain_cluster.cluster_PD_fairDcChain(self)