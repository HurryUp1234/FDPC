import numpy as np
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from FDPC import get_varience
from FDPC import remain_cluster


# 中心点选取：
# 密度*相对距离 排名大小选出当前点，并与已经选出的先前所有中心点进行距离比较，
# 若都大于fair_dc，则当前点做为中心点
# 否则，按排名对下一个进行判断,直到选出所k个中心点个中心点是人为去看决策图来决定的)

# 非中心点分配：
# 1.先分配中心点选取中被淘汰的点，对于每个淘汰点，直接分配给最近的中心点
# 2.对于其他点，先选一个候选点，对于当前要分配的点，在fair_dc之内选密度较大点，若有多个，选距离最近的作为候选点
# 若候选点没有被分配，则对候选点选它的候选点，直到某一个候选点有簇标记，就将这条候选链上的点都标记
# 若在成链的过程中，找不到候选点了，则按照原本的DPC方法进行分配，再给链上的点都分配成那个簇标记
class FDPC3:
    def __init__(self, data_name: str,  cluster_num=None,
                 dc_max_rate=0.1, dc_min_rate=0.01):
        self.data_name = data_name
        # 读数据库名字得到数据集，单属性所有敏感属性组，数据集各敏感属性组的占比

        self.dataset_cluster, self.fair_attr_vals, self.fair_attr = \
            get_varience.read_data(data_name)
        # 距离矩阵
        self.dist_matrix = squareform(pdist(self.dataset_cluster, metric='euclidean'))
        # dc: 用拉格朗日函数最优化目标函数:最小化所有点的dc公平损失

        self.fattr_val_ratio = get_varience.cal_fair_attr_val_ratio(self)
        self.fair_dc = get_varience.select_dc_lag(self, dc_min=dc_min_rate, dc_max=dc_max_rate, learning_rate=0.01,
                                                  max_iter=10, epsilon=1e-4)
        # 原dc:用于密度和相对距离计算
        self.dc = get_varience.select_dc_ratio(self, 0.02)
        # 各点局部密度(原密度),默认用截断核, 可用 Gaussion
        self.rho = get_varience.get_local_density(self, method="Gaussion")
        # 打印各点的fair_dc公平损失之和
        # fair_loss = get_varience.cal_dc_fair_loss(self, self.dc)
        # print(f'fair_loss_of_FDPC: {np.sum(self.fair_loss)}, fair_dc: {self.dc}')

        # 各点的中心偏移距离
        self.deltas, self.nearest_neiber = get_varience.get_deltas(self)
        # 选取聚类中心
        self.group_centers = remain_cluster.find_centers_auto(self, cluster_num)
        # 聚类分组
        self.group_label = remain_cluster.cluster_PD(self)

