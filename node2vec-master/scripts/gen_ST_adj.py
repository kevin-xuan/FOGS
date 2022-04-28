from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import numpy
import pickle
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp


def get_weighted_adjacency_matrix(distance_df, sensor_ids):  # 用W_s表示
    """

    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :return:
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind or len(row) != 3:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]
        dist_mx[sensor_id_to_ind[row[1]], sensor_id_to_ind[row[0]]] = row[2]
    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()

    adj_mx = np.exp(-np.square(dist_mx / std))

    return sensor_ids, sensor_id_to_ind, adj_mx


def get_time_volume_matrix(data_filename, period=12 * 24 * 7):  # 用W_v表示
    data = np.load(data_filename)['data'][:, :, 0]  # 26208 * 358 * 1
    num_samples, num_nodes = data.shape
    num_train = int(num_samples * 0.7)
    num_ave = int(num_train / period) * period
    time_volume_mx = np.zeros((num_nodes, 7, 288), dtype=np.float32)

    for node in range(num_nodes):
        for i in range(7):  # 星期一~星期天
            for t in range(288):  # 一天有288个时间段  将所有星期一的288个时间段的流量求均值。同理, 所有星期二, 星期三
                time_volume = []  # i*288+t表示星期XXX的0点时数据所对应的行数
                for j in range(i * 288 + t, num_ave, period):
                    time_volume.append(data[j][node])

                time_volume_mx[node][i][t] = np.array(time_volume).mean()  # 0也算在里面

    time_volume_mx = time_volume_mx.reshape(num_nodes, -1)  # (num_nodes, 7*288)

    # 计算l2-norm
    similarity_mx = np.ones((num_nodes, num_nodes), dtype=np.float32)  # 用于路网中,不能存在自环,因此对角线要为0
    similarity_mx[:] = np.inf  # DCRNN需要它,另外2种方法不用
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            similarity_mx[i][j] = similarity_mx[j][i] = np.sqrt(np.sum((time_volume_mx[i] - time_volume_mx[j]) ** 2))

    # DCRNN用的归一化 W_ij = exp(-(dist(vi,vj)/σ)^2)
    distances = similarity_mx[~np.isinf(similarity_mx)].flatten()
    std = distances.std()
    similarity_mx = np.exp(-np.square(similarity_mx / std))  # 主对角线为0

    # 线性比例变换法 y_i = x_i / max(x)的改版y_i = 1- x_i / max(x)
    # max_value = similarity_mx.max()
    # similarity_mx = 1 - similarity_mx / max_value
    # for i in range(similarity_mx.shape[0]):
    #     similarity_mx[i, i] = 0

    # 极差变换法 y_i = (x_i - min(x)) / (max(x) - min(x))的改版y_i = 1- (x_i - min(x)) / (max(x) - min(x)) 当min(x)=0时等价于线性比例变换法
    # max_value = similarity_mx.max()
    # min_value = similarity_mx.min()
    # print(max_value, min_value)
    # similarity_mx = 1 - (similarity_mx - min_value) / (max_value - min_value)
    # for i in range(similarity_mx.shape[0]):  # 尝试主对角线为1
    #     similarity_mx[i, i] = 1

    return time_volume_mx, similarity_mx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 只有03有id,所以sensor_ids_filename为'../data/PEMS03/PEMS03.txt'
    parser.add_argument('--sensor_ids_filename', type=str, default='',
                        help='File containing sensor ids separated by comma.')
    parser.add_argument('--num_of_vertices', type=int, default=883)  # 加了个顶点数,为04,07,08服务 03:358, 04:307, 07:883, 08:170
    parser.add_argument('--distances_filename', type=str, default='../data/PEMS08/PEMS08.csv',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--data_filename', type=str, default='../data/PEMS08/PEMS08.npz',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--edgelist_filename', type=str, default='../graph/PEMS08.edgelist',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--T_filename', type=str, default='../graph/PEMS08_graph_T.npz',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--thresh_T', type=float, default=20,
                        help='Entries that become lower than normalized_k after normalization are set to zero for sparsity.')
    parser.add_argument('--thresh_cos', type=float, default=10,
                        help='Entries that become lower than normalized_k after normalization are set to zero for sparsity.')
    # parser.add_argument('--normalized_k', type=float, default=0.1,
    #                     help='Entries that become lower than normalized_k after normalization are set to zero for sparsity.')
    parser.add_argument('--output_pkl_filename', type=str, default='../data/PEMS08/learn_two_10_mx.pkl',
                        help='Path of the output file.')
    args = parser.parse_args()

    if args.sensor_ids_filename != '':
        with open(args.sensor_ids_filename) as f:
            sensor_ids = f.read().strip().split('\n')  # 原先是‘,’,而PEMS03是'\n'
    else:
        sensor_ids = [str(i) for i in range(args.num_of_vertices)]
    distance_df = pd.read_csv(args.distances_filename, dtype={'from': 'str', 'to': 'str'})

    _, sensor_id_to_ind, adj_mx = get_weighted_adjacency_matrix(distance_df, sensor_ids)
    time_volume_mx, sim_mx = get_time_volume_matrix(args.data_filename)
    ### 注意, 这里要记得改数据集的名字!!!

    # learn_mx_1 = learn_matrix(adj_mx, sim_mx)
    # Save to pickle file.
    # with open(args.output_pkl_filename, 'wb') as f:
    #     pickle.dump([sensor_ids, sensor_id_to_ind, learn_mx], f, protocol=2)
    # T = construct_T(sim_mx)
    # consrtuct_edgelist(distance_df, sensor_ids)
    # sensor_id_to_ind = {}
    # for i, sensor_id in enumerate(sensor_ids):
    #     sensor_id_to_ind[sensor_id] = i
    # learn_mx_2 = learn_another_matrix()
    # with open(args.output_pkl_filename, 'wb') as f:
    #     pickle.dump([sensor_ids, sensor_id_to_ind, learn_mx_2], f, protocol=2)
