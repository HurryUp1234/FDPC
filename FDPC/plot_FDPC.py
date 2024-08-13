import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA




def draw_cluster(dpc):
    datas = dpc.dataset_cluster
    centers = dpc.group_centers
    # 将 NumPy 数组转换为 DataFrame 对象
    df = pd.DataFrame(datas)
    # 问 DataFrame 对象的 columns 属性
    dimension = len(df.columns) - 1
    plt.cla()

    label = dpc.group_label
    unique_labels = np.unique(label)
    colors = np.array(["red", "blue", "green", "orange", "purple", "cyan",
                       "magenta", "beige", "hotpink", "#88c999", "black"])

    if dimension > 2:
        datas = PCA(n_components=2).fit_transform(datas)  # 如果属性数量大于2，降维

    for i, label_id in enumerate(unique_labels):
        if label_id == -1:  # 噪声点
            plt.scatter(datas[label == label_id, 0], datas[label == label_id, 1], c='black', s=7, label='Noise')
        else:
            color = colors[i % len(colors)]
            plt.scatter(datas[label == label_id, 0], datas[label == label_id, 1], c=color, s=7, label=f'Cluster {label_id}')
    # 特别处理聚类中心，使用'+'标记
    plt.scatter(datas[centers, 0], datas[centers, 1], color='k', marker='+', s=200, label='Cluster Centers')
    # 添加标题
    plt.title(dpc.data_name)
    plt.legend()
    plt.show()


# 绘制决策图
def draw_decision(fdpc):
    rho = fdpc.rho
    deltas = fdpc.deltas
    data_matrix = fdpc.dataset_cluster

    plt.cla()
    for i in range(np.shape(data_matrix)[0]):
        plt.scatter(rho[i], deltas[i], s=16., color=(0, 0, 0))
        plt.annotate(str(i), xy=(rho[i], deltas[i]), xytext=(rho[i], deltas[i]))
        plt.xlabel("rho")
        plt.ylabel("deltas")
    # plt.savefig(filename+"_decision.jpg")
    plt.show()