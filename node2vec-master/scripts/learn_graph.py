import pickle
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import os


def get_cos_similar(v1, v2):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0  # 转换为[0,1]之间


def learn_final_graph(threshold, filename, direct):
    def readEmbedFile(embedFile):
        input = open(embedFile, 'r')
        lines = []
        for line in input:
            lines.append(line)

        embeddings = {}
        for lineId in range(1, len(lines)):  # 因为第一行是统计信息，所以用第二行
            splits = lines[lineId].split(' ')
            # embedId赋值
            embedId = int(splits[0])
            embedValue = splits[1:]
            new_embedValue = [float(x) for x in embedValue]
            embeddings[embedId] = new_embedValue

        return embeddings

    embeddings = readEmbedFile(filename)  # 字典形式

    index_list = []
    for index in embeddings.keys():
        index_list.append(index)

    num_nodes = len(index_list)
    cos_mx = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for i in range(num_nodes):  # 主对角线为0
        for j in range(i + 1, num_nodes):
            embedding_i = np.asarray(embeddings[i])
            embedding_j = np.asarray(embeddings[j])
            cos_value = get_cos_similar(embedding_i, embedding_j)
            cos_mx[i][j] = cos_mx[j][i] = cos_value

    learn_mx = np.zeros((num_nodes, num_nodes), dtype=np.float32)  # 主对角线要不要为1呢?
    for row in range(num_nodes):  # 有向图
        indices = np.argsort(cos_mx[row])[::-1][:threshold]  # 每行取前top k个最大值，返回对应的一维索引数组
        norm = cos_mx[row, indices].sum()
        for index in indices:
            learn_mx[row, index] = cos_mx[row, index] / norm

    if not direct:
        learn_mx = np.maximum.reduce([learn_mx, learn_mx.T])
        print('最终学的的图是无向图')
    else:
        print('最终学的的图是非对称矩阵')

    return learn_mx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename_emb', type=str, default='../emb/PEMS08.emb',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--output_pkl_filename', type=str, default='../../data/PEMS08',
                        help='Path of the output file.')
    parser.add_argument('--thresh_cos', type=int, default=10,
                        help='Threshold used in constructing final graph.')
    parser.add_argument('--direct_L', type=bool, default=True,
                        help='Whether is the final graph directed or undirected.')
    args = parser.parse_args()

    output_pkl_filename = args.output_pkl_filename + '/' + 'learn_mx.pkl'
    print(output_pkl_filename)

    if os.path.exists(args.filename_emb):
        learn_graph = learn_final_graph(args.thresh_cos, filename=args.filename_emb,
                                        direct=args.direct_L)
        with open(output_pkl_filename, 'wb') as f:
            pickle.dump(learn_graph, f, protocol=2)
