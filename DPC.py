import pandas as pd
import numpy as np
import math

#数据预处理（归一化）
def Z_Score(data):
    df = pd.read_csv(data)
    column_headers = list(df.columns.values)  # 获取列数
    a = len(column_headers)
    b = len(df)
    np_D = np.zeros((len(column_headers), len(df)))
    for i in range(len(column_headers)):
        r = np.loadtxt(data, skiprows=1, delimiter=',', usecols=i)
        ave = float(len(r)) / sum(r)  # 获取均值
        std = np.std(r)  # 获取标准差
        total = 0
        for j in range(len(r)):
            total += (j - ave) ** 2
        stddev = math.sqrt(total / len(r))
        np_D[i] = [(x - ave) / stddev for x in r]
    np_D=np.reshape(np_D, (b, a))#行列转换
    return (np_D)

# 计算距离矩阵
def getDistanceMatrix(datas):
    N, D = np.shape(datas)
    dists = np.zeros([N, N])

    for i in range(N):
        for j in range(N):
            vi = datas[i, :]
            vj = datas[j, :]
            dists[i, j] = np.sqrt(np.dot((vi - vj), (vi - vj)))
    return dists

#计算dc
def select_dc(dists):
    '''算法1'''
    N = np.shape(dists)[0]
    tt = np.reshape(dists, N * N)
    percent = 2.0
    position = int(N * (N - 1) * percent / 100)
    dc = np.sort(tt)[position + N]
    return dc

#计算局部密度
def get_density(dists, dc, method=None):
    N = np.shape(dists)[0]
    rho = np.zeros(N)

    for i in range(N):
        if method == None:
            rho[i] = np.where(dists[i, :] < dc)[0].shape[0] - 1
        else:
            rho[i] = np.sum(np.exp(-(dists[i, :] / dc) ** 2)) - 1
    return rho

#计算密度距离
def get_deltas(dists, rho):
    N = np.shape(dists)[0]
    deltas = np.zeros(N)
    nearest_neiber = np.zeros(N)
    # 将密度从大到小排序
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        # 对于密度最大的点
        if i == 0:
            continue

        # 对于其他的点
        # 找到密度比其大的点的序号
        index_higher_rho = index_rho[:i]
        # 获取这些点距离当前点的距离,并找最小值
        deltas[index] = np.min(dists[index, index_higher_rho])

        # 保存最近邻点的编号
        index_nn = np.argmin(dists[index, index_higher_rho])
        nearest_neiber[index] = index_higher_rho[index_nn].astype(int)

    deltas[index_rho[0]] = np.max(deltas)
    return deltas, nearest_neiber

 #获取聚类中心索引
def find_centers_K(rho, deltas, K):
    rho_delta = rho * deltas
    centers = np.argsort(-rho_delta)
    return  centers[:K]


#整合数据处理和DPC各部分，返回聚类中心所在行所有数据
def union_func(data):
    df = pd.read_csv(data)
    column_headers = list(df.columns.values)
    data = Z_Score(data)
    dists = getDistanceMatrix(data)
    dc = select_dc(dists)
    rho = get_density(dists, dc, method="Gaussion")
    deltas, nearest_neiber = get_deltas(dists, rho)  # 计算密度距离
    index = find_centers_K(rho, deltas, 7)  # 获取聚类中心，作为记忆矩阵
    rows = len(index)
    cols = len(column_headers)
    centers = np.zeros((rows, cols))
    for i in range(len(index)):
        a = index[i]
        centers[i] = df.iloc[a]
    return (centers)






















