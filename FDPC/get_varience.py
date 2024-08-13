from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from data_process.data import specialized_process
from data_process.data.specialized_process import *


# FDPC3
def read_data(data_name):
    if data_name == 'obesity':
        data_processed, fair_attr = get_data_obesity()
    elif data_name == 'bank':
        data_processed, fair_attr = get_data_bank()
    elif data_name == 'census1990':
        data_processed, fair_attr = get_data_census1990()
    elif data_name == 'creditcard':
        data_processed, fair_attr = get_data_creditcard()
    elif data_name == 'drug':
        data_processed, fair_attr = get_data_drug()
    elif data_name == 'dabestic':
        data_processed, fair_attr = get_data_dabestic()
    elif data_name == 'hcvdat0':
        data_processed, fair_attr = get_data_hcvdat0()
    elif data_name == 'athlete':
        data_processed, fair_attr = get_data_athlete()
    elif data_name == 'adult':
        data_processed, fair_attr = get_data_adult()
    elif data_name == 'drug_consumption':
        data_processed, fair_attr = get_data_drug_consumption()
    elif data_name == 'abalone':
        data_processed, fair_attr = get_data_abalone()
    elif data_name == 'Room_Occupancy_Estimation':
        data_processed, fair_attr = get_data_Room_Occupancy_Estimation()
    elif data_name == 'Rice':
        data_processed, fair_attr = get_data_Rice()
    elif data_name == 'Wholesale':
        data_processed, fair_attr = get_data_Wholesale()
    elif data_name == 'student':
        data_processed, fair_attr = get_data_student()
    elif data_name == 'parkinsons':
        data_processed, fair_attr = get_data_parkinsons()
    elif data_name == 'vertebral':
        data_processed, fair_attr = get_data_vertebral()
    elif data_name == 'liver_disorder':
        data_processed, fair_attr = get_data_liver_disorder()
    elif data_name == 'heart_failure_clinical':
        data_processed, fair_attr = get_data_heart_failure_clinical()
    elif data_name == 'chronic_kidney_disease':
        data_processed, fair_attr = get_data_chronic_kidney_disease()
    elif data_name == 'dermatology':
        data_processed, fair_attr = get_data_dermatology()
    elif data_name == 'glass':
        data_processed, fair_attr = get_data_glass()
    elif data_name == 'wdbc':
        data_processed, fair_attr = get_data_wdbc()
    elif data_name == 'wine':
        data_processed, fair_attr = get_data_wine()
    elif data_name == 'seeds':
        data_processed, fair_attr = get_data_seeds()
    else:
        data_processed, fair_attr = get_data_iris()

    need_subgraph = []
    # "abalone", "athlete", "Rice",
    # "adult", "census1990","Room_Occupancy_Estimation","creditcard"
    # 特定数据集的最佳阈值
    best_thresh = {
        # "abalone": 3.802,
        # "bank": 0.198,  # 0.17
        # "Rice": 0.155,
        # "Room_Occupancy_Estimation": 4.3,
        # "athlete": 0.09,  # 0.3: # 0.09:20%
        # "adult": 0.33,
        # "creditcard": 0.25,  # 0.3
        # "census1990": 5.68  # 8
    }

    # 创建一个默认值为1的字典
    thresholds = defaultdict(lambda: 1, best_thresh)

    # 查看数据集是否需要寻找最大联通子图
    if data_name in need_subgraph:
        print(f"need_subgraph: {data_name}, initial size: {len(data_processed)}")
        G = create_graph_from_data(data_processed, threshold=thresholds[data_name])
        selected_data = extract_largest_connected_subgraph(G, data_processed)
        print(f"after_subgraph size: {len(selected_data)}")
        # 保存处理后的数据
        output_path = f"{project_path}{data_name}_processed.csv"
        selected_data.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
    else:
        selected_data = data_processed

    # 分离敏感属性
    fair_attribute = selected_data[fair_attr]
    fair_attribute_array = fair_attribute.values

    # 分离聚类属性
    cluster_attributes = selected_data.drop(fair_attr, axis=1)
    cluster_attributes_array = cluster_attributes.values

    # 对聚类属性的数据进行归一化处理
    scaler = StandardScaler()
    cluster_attributes_normalized = scaler.fit_transform(cluster_attributes_array)

    # 敏感属性组-与对应点的编号
    return cluster_attributes_normalized, fair_attribute_array, fair_attr


# 计算数据集点之间的马氏距离
def compute_mahalanobis_distance_matrix(self):
    """计算马氏距离矩阵"""
    n_samples = self.dataset_cluster.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):  # 利用对称性减少计算量
            dist = mahalanobis(self.dataset_cluster[i], self.dataset_cluster[j], self.inv_cov_matrix)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # 对称矩阵

    return dist_matrix


# FDPC3 敏感属性组在数据集中占比
def cal_fair_attr_val_ratio(fdpc):
    # 敏感属性的每个取值对应的点的数目与数据集点数的比值
    # 已经有data_cluster, data_fair
    data_fair = fdpc.fair_attr_vals
    tot_cnt = len(data_fair)
    fattr_val_ratio = {val: 0 for val in set(data_fair)}
    for i in range(tot_cnt):
        fattr_val_ratio[data_fair[i]] += 1

    for val, cnt in fattr_val_ratio.items():
        fattr_val_ratio[val] /= tot_cnt
    return fattr_val_ratio


# FDPC3
# 所有点dc邻域公平损失
def cal_dc_fair_loss(fdpc, dc):
    dis_matrix = fdpc.dist_matrix
    data_cluster = fdpc.dataset_cluster

    fair_loss = np.zeros(len(data_cluster))
    for p in range(len(data_cluster)):
        # 得到点p的dc近邻点
        distances = dis_matrix[p]
        dc_neighbors = [dc_p for dc_p, dis in enumerate(distances) if dis <= dc]
        fair_loss[p] = dc_fair_loss_of_p(fdpc, dc_neighbors)
        # print(f'P--fair_loss:{p} : {fair_loss[p]}')
    return fair_loss


# FDPC3
# 单个点的dc领域公平损失
def dc_fair_loss_of_p(fdpc, dc_nbs: []):
    ratio_all = fdpc.fattr_val_ratio
    fair_attr_vals = fdpc.fair_attr_vals

    dc_fatrr_val_cnt = {val: 0 for val in set(fair_attr_vals)}
    for dc_p in dc_nbs:
        dc_fatrr_val_cnt[fair_attr_vals[dc_p]] += 1

    tot_dc_cnt = len(dc_nbs)
    p_fair_loss = 0.0

    # 计算公平损失：敏感属性组的实际占比与期望占比的平方差
    for val, dc_fattr_val_cnt in dc_fatrr_val_cnt.items():
        ratio_all_ = ratio_all[val]
        ratio_dc_ = dc_fattr_val_cnt / tot_dc_cnt if tot_dc_cnt > 0 else 0
        p_fair_loss += (ratio_dc_ - ratio_all_) ** 2

    return p_fair_loss


#
# dc: 用拉格朗日函数最优化目标函数:最小化所有点的dc公平损失

# 优化目标: min f(dc) = L(dc)
# 约束条件: g₁(dc) = dc - dc_min >= 0 g₂(dc) = dc_max - dc >= 0
# 对于这个问题，满足以下KKT条件的dc由定义是此问题的局部最优:
# 梯度条件: ∇f(dc) - λ₁∇g₁(dc) - λ₂∇g₂(dc) = 0
# 原始条件: g₁(dc) ≥ 0, g₂(dc) ≥ 0
# 对偶条件: λ₁ ≥ 0, λ₂ ≥ 0
# 互补松弛条件: λ₁g₁(dc) = 0, λ₂g₂(dc) = 0
# dc: 用拉格朗日函数最优化目标函数:最小化所有点的dc公平损失
# 优化目标: min f(dc) = L(dc)
# 只需要考虑上界约束 ( g_2(dc) = dc_{\text{max}} - dc \geq 0 )
# 梯度条件: [ \nabla f(dc) - \lambda_2 \nabla g_2(dc) = 0 ]
# 原始条件: [ g_2(dc) \geq 0 ]
# 对偶条件: [ \lambda_2 \geq 0 ]
# 互补松弛条件: [ \lambda_2 g_2(dc) = 0 ]
def select_dc_lag(fdpc, dc_min, dc_max, learning_rate=1e-3, max_iter=20, epsilon=1e-2):
    """
    通过梯度下降和拉格朗日乘子法查找最佳dc
    """
    # 初始化 dc 和拉格朗日乘数

    dc_min = select_dc_orin(fdpc, dc_min)
    dc_max = select_dc_orin(fdpc, dc_max)
    dc = (dc_min + dc_max) / 2.0
    lambda_1, lambda_2 = 0.0, 0.0
    for i in range(max_iter):
        # 计算函数在dc + epsilon和 dc - epsilon处的值
        fair_loss_plus = np.sum(cal_dc_fair_loss(fdpc, dc + epsilon))
        fair_loss_minus = np.sum(cal_dc_fair_loss(fdpc, dc - epsilon))

        # 使用中央差分法计算梯度
        gradient = (fair_loss_plus - fair_loss_minus) / (2 * epsilon)
        # 更新 dc ,因为lambda_1和lambda_2实时变化，lambda_1 - lambda_2可正可负，dc可加可减
        dc -= learning_rate * (gradient + lambda_1 - lambda_2)

        # 保证 dc 在 [dc_min, dc_max] 内并更新拉格朗日乘数
        if dc < dc_min:
            dc = dc_min
            lambda_1 += learning_rate

        elif dc > dc_max:
            dc = dc_max
            lambda_2 += learning_rate

        # 检查KKT条件
        g1, g2 = dc - dc_min, dc_max - dc  # 计算约束函数值
        if abs(gradient - lambda_1 + lambda_2) <= epsilon \
                and g1 >= 0 and g2 >= 0 \
                and lambda_1 >= 0 and lambda_2 >= 0 and \
                lambda_1 * g1 <= epsilon and lambda_2 * g2 <= epsilon:
            # print("KKT conditions satisfied")
            return dc

    return dc


# DPC: 按照比例选出dc
def select_dc_orin(fdpc, percent):
    dists = fdpc.dist_matrix
    N = np.shape(dists)[0]
    tt = np.reshape(dists, N * N)  # 转一维方便排序
    position = int(N * (N - 1) * percent)
    dc = np.sort(tt)[position]
    return dc


# DPC: 局部密度(原密度)
# 方法：默认为截断核,可切换为高斯核
def get_local_density(fdpc, method=None):
    dists = fdpc.dist_matrix
    dc = fdpc.dc
    N = np.shape(dists)[0]
    rho = np.zeros(N)

    for i in range(N):
        if method == None:  # 截断核
            rho[i] = np.where(dists[i, :] < dc)[0].shape[0] - 1
        else:
            rho[i] = np.sum(np.exp(-(dists[i, :] / dc) ** 2)) - 1
    return rho


# DPC: 原密度较大点相对距离
def get_deltas(fdpc):
    dists = fdpc.dist_matrix
    # rho = fdpc.fair_rho
    rho = fdpc.rho

    N = np.shape(dists)[0]
    deltas = np.zeros(np.shape(dists)[0])
    nearest_neiber = np.zeros(N)
    # 将密度从大到小排序
    rho_sorted_point_id = np.argsort(-rho)
    for i, point in enumerate(rho_sorted_point_id):  # i:表示大小顺序  point ：表示元素，此处代表点的编号
        # 对于密度最大的点
        if i == 0:
            continue

        point_higher_rho = rho_sorted_point_id[:i]
        # 存距离
        deltas[point] = np.min(dists[point, point_higher_rho])

        # 存点编号(可以删)
        point_mindist_to = np.argmin(dists[point, point_higher_rho])
        nearest_neiber[point] = point_higher_rho[point_mindist_to].astype(int)

    deltas[rho_sorted_point_id[0]] = np.max(deltas)
    return deltas, nearest_neiber


# 中心点选取通过fair_dc距离限制
def find_centers_fairDc(fdpc, k):
    ps = np.argsort(-fdpc.rho)
    selected_centers = []
    wasted_centers = []
    for p in ps:
        if len(selected_centers) < k:
            if all(fdpc.dist_matrix[p, selected_centers] > 2 * fdpc.fair_dc):
                selected_centers.append(p)
            else:
                wasted_centers.append(p)

    return np.array(selected_centers), np.array(wasted_centers)
