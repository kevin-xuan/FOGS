from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import pickle


def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    """

    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
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
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return sensor_ids, sensor_id_to_ind, adj_mx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 只有03有id,所以sensor_ids_filename为'../data/PEMS03/PEMS03.txt'
    parser.add_argument('--sensor_ids_filename', type=str, default='',
                        help='File containing sensor ids separated by comma.')
    parser.add_argument('--num_of_vertices', type=int, default=358)  # 加了个顶点数,为04,07,08服务
    parser.add_argument('--distances_filename', type=str, default='../data/PEMS04/PEMS04.csv',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--data_filename', type=str, default='../data/PEMS04/PEMS04.npz',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--normalized_k', type=float, default=0.1,
                        help='Entries that become lower than normalized_k after normalization are set to zero for sparsity.')
    parser.add_argument('--output_pkl_filename', type=str, default='../data/PEMS04/adj_mx.pkl',
                        help='Path of the output file.')
    args = parser.parse_args()

    if args.sensor_ids_filename != '':
        with open(args.sensor_ids_filename) as f:
            sensor_ids = f.read().strip().split('\n')  # 原先是‘,’,而PEMS03是'\n'
    else:
        sensor_ids = [str(i) for i in range(args.num_of_vertices)]
    distance_df = pd.read_csv(args.distances_filename, dtype={'from': 'str', 'to': 'str'})
    _, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, sensor_ids)
    # Save to pickle file.
    # with open(args.output_pkl_filename, 'wb') as f:
    #     pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)
    print(type(adj_mx))


