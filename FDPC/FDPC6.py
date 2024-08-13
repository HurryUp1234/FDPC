import numpy as np
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from FDPC import get_varience
from FDPC import remain_cluster
import networkx as nx

# ���ĵ�ѡȡ��fair_dc���ƺͷ�����Լ��빫ƽԼ��
class FDPC6:
    def __init__(self, cluster_attr_vals, f_attr_vals, f_attr, data_name,
                 dc_rate, fair_dc_min_rate, fair_dc_max_rate, cluster_num=3,
                 fair_tolerance=0.1
                 ):
        self.data_name = data_name
        self.dataset_cluster = cluster_attr_vals
        self.fair_attr_vals = f_attr_vals
        self.fair_attr = f_attr
        # �����ݿ����ֵõ����ݼ����������������������飬���ݼ��������������ռ��
        # �������(���Ͼ���)
        # cov_matrix = np.cov(self.dataset_cluster.T)
        # self.inv_cov_matrix = np.linalg.inv(cov_matrix)
        # self.dist_matrix = get_varience.compute_mahalanobis_distance_matrix(self)
        self.dist_matrix = squareform(pdist(self.dataset_cluster, metric='euclidean'))
        # ��������������ռ��
        self.fattr_val_ratio = get_varience.cal_fair_attr_val_ratio(self)

        # dc�����ܶȼ���
        self.dc = get_varience.select_dc_orin(self, dc_rate)
        # ����ֲ��ܶ�(ԭ�ܶ�),Ĭ���ýضϺ�, ���� Gaussion
        self.rho = get_varience.get_local_density(self, method="Gaussion")
        # �������Ծ���
        self.deltas, self.nearest_neiber = get_varience.get_deltas(self)

        # fair_dc�������ĵ��ľ���Լ��
        # self.fair_dc = get_varience.select_dc_lag(self, dc_min=fair_dc_min_rate, dc_max=fair_dc_max_rate)

        self.fair_dc = get_varience.select_dc_lag(self, fair_dc_min_rate, fair_dc_max_rate)

        # ѡȡ��������(�Ķ�)
        self.group_centers, self.wasted_centers = get_varience.find_centers_fairDc(self, cluster_num)
        # ���ǹ�ƽԼ��
        self.group_label = remain_cluster.cluster_PD_fair_constraint(self, fair_tolerance=fair_tolerance)