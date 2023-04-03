import pickle
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import os


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

                time_volume_mx[node][i][t] = np.array(time_volume).mean()

    time_volume_mx = time_volume_mx.reshape(num_nodes, -1)  # (num_nodes, 7*288)

    # 计算l2-norm
    similarity_mx = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    similarity_mx[:] = np.inf
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            similarity_mx[i][j] = similarity_mx[j][i] = np.sqrt(np.sum((time_volume_mx[i] - time_volume_mx[j]) ** 2))

    distances = similarity_mx[~np.isinf(similarity_mx)].flatten()
    std = distances.std()
    similarity_mx = np.exp(-np.square(similarity_mx / std))  # 主对角线为0

    return time_volume_mx, similarity_mx


def construct_T(sim_mx, threshold, filename, direct):  # 用W_V构造T,用knn原理选择每行前threshold为True
    num_nodes = sim_mx.shape[0]
    temporal_graph = np.zeros((num_nodes, num_nodes), dtype=bool)
    for row in range(num_nodes):  # 主对角线为0
        indices = np.argsort(sim_mx[row])[::-1][:threshold]  # 取top k个为True,sim_mx主对角线为0,因此top k不会出现在主对角线上
        temporal_graph[row, indices] = True

    if not direct:  # 构造对称矩阵
        temporal_graph = np.maximum.reduce([temporal_graph, temporal_graph.T])
        print('构造的时间相似性矩阵是对称的')

    np.savez(filename, data=temporal_graph)
    return temporal_graph


def consrtuct_edgelist(distance_df, sensor_ids, filename, weighted=False):
    G = nx.Graph()
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # 在node2vec里默认设置为无向图,所以这里构造有向图是可以的！！！
    if weighted:
        for row in distance_df.values:
            if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind or len(row) != 3:
                continue
            G.add_edge(sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]], weight=row[2])
        nx.write_weighted_edgelist(G, filename)
    else:
        for row in distance_df.values:
            if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind or len(row) != 3:
                continue
            G.add_edge(sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]])
        nx.write_edgelist(G, filename, data=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 只有03有id,所以sensor_ids_filename为'../data/PEMS03/PEMS03.txt'
    parser.add_argument('--sensor_ids_filename', type=str, default='',
                        help='File containing sensor ids separated by comma.')
    parser.add_argument('--num_of_vertices', type=int, default=170)  # 加了个顶点数,为04,07,08服务 03:358, 04:307, 07:883, 08:170
    parser.add_argument('--distances_filename', type=str, default='../../data/PEMS08/PEMS08.csv',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--data_filename', type=str, default='../../data/PEMS08/PEMS08.npz',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--edgelist_filename', type=str, default='../graph/PEMS08.edgelist',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--filename_T', type=str, default='../graph/PEMS08_graph_T.npz',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--flow_mean', type=str, default='../../data/PEMS08/PEMS08_flow_count.pkl',
                        help='store average flow.')
    parser.add_argument('--thresh_T', type=int, default=10,
                        help='Threshold used in constructing temporal graph.')
    parser.add_argument('--direct_T', type=bool, default=False,
                        help='Whether is the temporal graph directed or undirected.')
    args = parser.parse_args()

    if args.sensor_ids_filename != '':
        with open(args.sensor_ids_filename) as f:
            sensor_ids = f.read().strip().split('\n')
    else:
        sensor_ids = [str(i) for i in range(args.num_of_vertices)]

    distance_df = pd.read_csv(args.distances_filename, dtype={'from': 'str', 'to': 'str'})
    print('Constructing spatial matrix......')
    _, sensor_id_to_ind, adj_mx = get_weighted_adjacency_matrix(distance_df, sensor_ids)

    if not os.path.exists(args.edgelist_filename):
        print('Constructing temporal matrix......')
        time_volume_mx, sim_mx = get_time_volume_matrix(args.data_filename)  # 构造时间相似性矩阵

        # 存起流量均值数据！！！
        print(args.flow_mean)
        with open(args.flow_mean, 'wb') as f:
            pickle.dump(time_volume_mx, f, protocol=2)

        print('Constructing temporal graph......')
        T = construct_T(sim_mx, threshold=args.thresh_T, filename=args.filename_T, direct=args.direct_T)  # 变成稀疏矩阵，生成时间图

        print('Constructing spatial graph......')
        consrtuct_edgelist(distance_df, sensor_ids, filename=args.edgelist_filename)  # 根据路网构造空间图

