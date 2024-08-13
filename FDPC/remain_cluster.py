import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import silhouette_score, davies_bouldin_score
from platypus import NSGAII, Problem, Real

def find_centers_auto(fdpc, k=None):
    # rho = fdpc.fair_rho
    rho = fdpc.rho
    deltas = fdpc.deltas
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
        rho_delta = fdpc.rho * fdpc.deltas
        centers = np.argsort(-rho_delta)
        centers = centers[:k]

    return centers


# 基于相对距离的正向选点（原方法）
def cluster_PD(fdpc):
    centers = fdpc.group_centers
    rho = fdpc.rho
    nearest_neiber = fdpc.nearest_neiber

    K = np.shape(centers)[0]
    if K == 0:
        print("can not find centers")
        return

    N = np.shape(rho)[0]
    label = -1 * np.ones(N).astype(int)

    # 有几个聚类中心就分为几个簇
    for i, center in enumerate(centers):
        label[center] = i

    # 将密度从大到小排序
    index_rho = np.argsort(-rho)
    for i, point in enumerate(index_rho):

        # 从密度大的点进行标号
        if label[point] == -1:
            # 如果没有被标记过
            # 那么聚类标号与距离其最近且密度比其大的点的标号相同
            # 密度比当前点大的，一定已经先排好位置了
            label[point] = label[int(nearest_neiber[point])]
    return label

# FDPC6
# 公平约束限制的聚类方法
def cluster_PD_fair_constraint(fdpc, fair_tolerance=0.01):
    centers = fdpc.group_centers
    rho = fdpc.rho
    nearest_higher_rho = fdpc.nearest_neiber

    K = np.shape(centers)[0]

    N = np.shape(rho)[0]
    label = -1 * np.ones(N).astype(int)

    # 确保fair_attr_vals是一个包含敏感属性值的列表或数组
    fair_attr_values = np.array(fdpc.fair_attr_vals)

    # 计算每个敏感属性值在数据集中的比例
    total_count = len(fdpc.dataset_cluster)
    f_attr_distri_P = {
        val: (fair_attr_values == val).sum() / total_count
        for val in np.unique(fair_attr_values)
    }

    # 初始化每个簇的敏感属性分布计数
    clusters_f_attr_cnt = [
        {val: 0 for val in np.unique(fair_attr_values)}
        for _ in range(K)
    ]

    # 初始化每个簇的点数
    cluster_sizes = [0] * K

    # 有几个聚类中心就分为几个簇
    for i, center in enumerate(centers):
        label[center] = i
        clusters_f_attr_cnt[i][fair_attr_values[center]] += 1
        cluster_sizes[i] += 1

    dist_matrix = fdpc.dist_matrix
    # 第一步，淘汰点密度较大，需要先分配给最近的簇
    for p in fdpc.wasted_centers:
        f_attr_val = fair_attr_values[p]
        distances = [dist_matrix[p, center] for center in centers]
        label[p] = label[centers[np.argmin(distances)]]
        clusters_f_attr_cnt[label[p]][f_attr_val] += 1

    # 第二步，将密度从大到小排序，先给密度大的分配
    index_rho = np.argsort(-rho)
    for i, point in enumerate(index_rho):
        if label[point] == -1:
            # 获取最近邻的聚类标签
            nearest_higher_rho_label = label[int(nearest_higher_rho[point])]
            # 检查当前点的敏感属性
            f_attr_val = fair_attr_values[point]
            # 公平偏差
            fairness_deviation = calculate_fairness_deviation(clusters_f_attr_cnt, nearest_higher_rho_label, f_attr_val,
                                                              f_attr_distri_P)
            # 密度较大点最近邻在fair_dc内 并且 公平偏差在公平容忍度以内
            if dist_matrix[point][int(nearest_higher_rho[point])] < fdpc.fair_dc \
                    and fairness_deviation < fair_tolerance:
                label[point] = nearest_higher_rho_label
                clusters_f_attr_cnt[nearest_higher_rho_label][f_attr_val] += 1
            # 否则，使用pareto进行簇分配
            else:
                best_cluster = nsga2_assignment(fdpc, point, f_attr_val, f_attr_distri_P, clusters_f_attr_cnt, K, label)
                label[point] = best_cluster
                clusters_f_attr_cnt[best_cluster][f_attr_val] += 1
    return label


# FDPC5
def cluster_PD_fairDcChain(fdpc, fair_tolerance=0.05):
    centers = fdpc.group_centers
    rho = fdpc.rho
    nearest_neiber = fdpc.nearest_neiber

    K = np.shape(centers)[0]

    N = np.shape(rho)[0]
    label = -1 * np.ones(N).astype(int)

    # 确保fair_attr_vals是一个包含敏感属性值的列表或数组
    fair_attr_values = np.array(fdpc.fair_attr_vals)

    # 计算每个敏感属性值在数据集中的比例
    total_count = len(fdpc.dataset_cluster)
    f_attr_distri_P = {
        val: (fair_attr_values == val).sum() / total_count
        for val in np.unique(fair_attr_values)
    }

    # 初始化每个簇的敏感属性分布计数
    clusters_f_attr_cnt = [
        {val: 0 for val in np.unique(fair_attr_values)}
        for _ in range(K)
    ]

    # 初始化每个簇的点数
    cluster_sizes = [0] * K

    # 有几个聚类中心就分为几个簇
    for i, center in enumerate(centers):
        label[center] = i
        clusters_f_attr_cnt[i][fair_attr_values[center]] += 1
        cluster_sizes[i] += 1

    dist_matrix = fdpc.dist_matrix
    # 第一步，淘汰点密度较大，需要先分配给最近的簇
    for p in fdpc.wasted_centers:
        distances = [dist_matrix[p, center] for center in centers]
        label[p] = label[centers[np.argmin(distances)]]

    # 第二步，将密度从大到小排序，先给密度大的分配
    index_rho = np.argsort(-rho)
    for i, point in enumerate(index_rho):
        if label[point] == -1:
            # 获取最近邻的聚类标签
            nearest_label = label[int(nearest_neiber[point])]
            # 检查当前点的敏感属性
            f_attr_val = fair_attr_values[point]
            # fair_dc内不考虑公平性
            if dist_matrix[point][int(nearest_neiber[point])] < fdpc.fair_dc:
                    label[point] = nearest_label
                    clusters_f_attr_cnt[nearest_label][f_attr_val] += 1

    # 第三步，使用pareto进行簇分配
    for point in range(N):
        if label[point] == -1:
            f_attr_val = fair_attr_values[point]
            best_cluster = nsga2_assignment(fdpc, point, f_attr_val, f_attr_distri_P, clusters_f_attr_cnt, K, label)
            label[point] = best_cluster
            clusters_f_attr_cnt[best_cluster][f_attr_val] += 1
    return label


def nsga2_assignment(fdpc, point, f_attr_val, f_attr_distri_P, clusters_f_attr_cnt, cluster_num, label):
    distances = []
    deviations = []

    for cluster_index in range(cluster_num):
        distance_metric = calculate_distance_metric(fdpc, point, cluster_index, label)
        fairness_deviation = calculate_fairness_deviation(clusters_f_attr_cnt, cluster_index, f_attr_val,
                                                          f_attr_distri_P)
        distances.append(distance_metric)
        deviations.append(fairness_deviation)

    def objectives(vars):
        cluster_index = int(vars[0])
        return [distances[cluster_index], deviations[cluster_index]]

    problem = Problem(1, 2)
    problem.types[:] = Real(0, cluster_num - 1)
    problem.function = objectives

    algorithm = NSGAII(problem)
    algorithm.run(15)
    # 选择 Pareto 前沿上公平性偏差最小的解
    feasible_solutions = [s for s in algorithm.result if s.feasible]
    best_solution = min(feasible_solutions, key=lambda s: s.objectives[1])
    best_cluster = int(best_solution.variables[0])

    return best_cluster



# 用于fdc和公平偏差上界的分配
def calculate_fairness_deviation(clusters_f_attr_cnt, cluster_index, f_attr_val, f_attr_distri_P):
    cluster_total = sum(clusters_f_attr_cnt[cluster_index].values())
    cluster_ratio = clusters_f_attr_cnt[cluster_index][f_attr_val] / cluster_total
    # 整个数据集对于分配点的敏感属性比例
    global_sensitive_ratio = f_attr_distri_P[f_attr_val]
    # 计算偏差
    deviation = (cluster_ratio - global_sensitive_ratio)

    return deviation


# 用于帕累托，NG-2智能优化算法的分配
def calculate_fairness_deviation2(clusters_f_attr_cnt, cluster_index, f_attr_val, f_attr_distri_P):
    cluster_total = sum(clusters_f_attr_cnt[cluster_index].values())
    cluster_ratio = clusters_f_attr_cnt[cluster_index][f_attr_val] / cluster_total
    # 整个数据集对于分配点的敏感属性比例
    global_sensitive_ratio = f_attr_distri_P[f_attr_val]
    # 计算偏差
    deviation = abs(cluster_ratio - global_sensitive_ratio)

    return deviation

def calculate_distance_metric(fdpc, point, cluster_index, label):
    cluster_indices = np.where(label == cluster_index)[0]
    distances_to_cluster = fdpc.dist_matrix[point, cluster_indices]
    return np.mean(distances_to_cluster)