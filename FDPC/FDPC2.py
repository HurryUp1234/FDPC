import numpy as np
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from FDPC import get_varience
from FDPC import remain_cluster

# 对照组
class FDPC2:
    def __init__(self, data_name,  dataset_cluster,  fair_attr_vals, fair_attr, dc_rate, cluster_num=None):

        # 读数据库名字得到数据集，单属性所有敏感属性组，数据集各敏感属性组的占比
        self.data_name = data_name
        self.dataset_cluster, self.fair_attr_vals, self.fair_attr = \
            dataset_cluster, fair_attr_vals, fair_attr
        # 距离矩阵
        self.dist_matrix = squareform(pdist(self.dataset_cluster, metric='euclidean'))
        # dc: 不进行优化
        self.fattr_val_ratio = get_varience.cal_fair_attr_val_ratio(self)
        self.dc = get_varience.select_dc_orin(self, dc_rate)
        # print(f"dc of DPC : {self.dc}")

        # 各点局部密度(原密度),默认用截断核, 可用 Gaussion
        self.rho = get_varience.get_local_density(self, method="Gaussion")
        # 计算各点的dc公平损失
        self.fair_loss = get_varience.cal_dc_fair_loss(self, self.dc)
        # print('-----------------------')
        # 各点的中心偏移距离
        self.deltas, self.nearest_neiber = get_varience.get_deltas(self)
        # print(self.fair_rho)
        # 选取聚类中心
        self.group_centers = remain_cluster.find_centers_auto(self, cluster_num)
        # 聚类分组
        self.group_label = remain_cluster.cluster_PD(self)

    # 根据密度和相对距离来选出指定的k各聚类中心
    def find_centers_auto(self, k=None):
        # rho = self.fair_rho
        rho = self.rho
        deltas = self.deltas
        centers = []

        if k == 0 or k == None:
            rho_threshold = (np.min(rho) + np.max(rho)) / 2
            delta_threshold = (np.min(deltas) + np.max(deltas)) / 2
            N = np.shape(rho)[0]

            for i in range(N):
                if rho[i] >= rho_threshold and deltas[i] > delta_threshold:
                    centers.append(i)
            return np.array(centers)

        else:
            rho_delta = self.rho * self.deltas
            centers = np.argsort(-rho_delta)
            centers = centers[:k]

        return centers
