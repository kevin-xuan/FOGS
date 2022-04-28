import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
import pickle
import copy
import scipy.sparse as sp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def log_string(log, string):
    """打印log"""
    log.write(string + '\n')
    log.flush()
    print(string)


def count_parameters(model):
    """统计模型参数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_seed(seed):
    """Disable cudnn to maximize reproducibility 禁用cudnn以最大限度地提高再现性"""
    torch.cuda.cudnn_enabled = False
    """
    cuDNN使用非确定性算法，并且可以使用torch.backends.cudnn.enabled = False来进行禁用
    如果设置为torch.backends.cudnn.enabled =True，说明设置为使用使用非确定性算法
    然后再设置：torch.backends.cudnn.benchmark = True，当这个flag为True时，将会让程序在开始时花费一点额外时间，
    为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
    但由于其是使用非确定性算法，这会让网络每次前馈结果略有差异,如果想要避免这种结果波动，可以将下面的flag设置为True
    """
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


"""图相关"""


def get_adjacency_matrix(distance_df_filename, num_of_vertices, type_='connectivity', id_filename=None):
    """
    :param distance_df_filename: str, csv边信息文件路径
    :param num_of_vertices:int, 节点数量
    :param type_:str, {connectivity, distance}
    :param id_filename:str 节点信息文件， 有的话需要构建字典
    """
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 建立映射列表
        df = pd.read_csv(distance_df_filename)
        for row in df.values:
            if len(row) != 3:
                continue
            i, j = int(row[0]), int(row[1])
            A[id_dict[i], id_dict[j]] = 1
            A[id_dict[j], id_dict[i]] = 1

        return A

    df = pd.read_csv(distance_df_filename)
    for row in df.values:
        if len(row) != 3:
            continue
        i, j, distance = int(row[0]), int(row[1]), float(row[2])
        if type_ == 'connectivity':
            A[i, j] = 1
            A[j, i] = 1
        elif type == 'distance':
            A[i, j] = 1 / distance
            A[j, i] = 1 / distance
        else:
            raise ValueError("type_ error, must be "
                             "connectivity or distance!")

    return A


def construct_adj(A, steps):
    """
    构建local 时空图
    :param A: np.ndarray, adjacency matrix, shape is (N, N)
    :param steps: 选择几个时间步来构建图
    :return: new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)
    """
    N = len(A)  # 获得行数
    adj = np.zeros((N * steps, N * steps))

    for i in range(steps):
        """对角线代表各个时间步自己的空间图，也就是A"""
        adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A

    for i in range(N):
        for k in range(steps - 1):
            """每个节点只会连接相邻时间步的自己"""
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1

    for i in range(len(adj)):
        """加入自回"""
        adj[i, i] = 1

    return adj


def construct_adj_fusion(A, A_dtw, steps):
    '''
    construct a bigger adjacency matrix using the given matrix

    Parameters
    ----------
    A: np.ndarray, adjacency matrix, shape is (N, N)

    steps: how many times of the does the new adj mx bigger than A

    Returns
    ----------
    new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)

    ----------
    This is 4N_1 mode:

    [T, 1, 1, T
     1, S, 1, 1
     1, 1, S, 1
     T, 1, 1, T]

    '''

    N = len(A)
    adj = np.zeros([N * steps] * 2)  # "steps" = 4 !!!

    for i in range(steps):
        if (i == 1) or (i == 2):
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A
        else:
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A_dtw
    #'''
    for i in range(N):
        for k in range(steps - 1):
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1
    #'''
    adj[3 * N: 4 * N, 0:  N] = A_dtw  # adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[0: N, 3 * N: 4 * N] = A_dtw  # adj[0 * N : 1 * N, 1 * N : 2 * N]

    adj[2 * N: 3 * N, 0: N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
    adj[0: N, 2 * N: 3 * N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
    adj[1 * N: 2 * N, 3 * N: 4 * N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
    adj[3 * N: 4 * N, 1 * N: 2 * N] = adj[0 * N: 1 * N, 1 * N: 2 * N]


    for i in range(len(adj)):
        adj[i, i] = 1

    return adj


"""数据加载器"""


class DataLoader(object):
    def __init__(self, xs, ys, x_ts, y_ts, batch_size, y_dist=None, pad_with_last_sample=True):
        """
        数据加载器
        :param xs:训练数据
        :param ys:标签数据
        :param batch_size:batch大小
        :param pad_with_last_sample:剩余数据不够时，是否复制最后的sample以达到batch大小
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            x_slot_padding = np.repeat(x_ts[-1:], num_padding, axis=0)
            y_slot_padding = np.repeat(y_ts[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            x_ts = np.concatenate([x_ts, x_slot_padding], axis=0)
            y_ts = np.concatenate([y_ts, y_slot_padding], axis=0)
            if y_dist is not None:
                y_dist_padding = np.repeat(y_dist[-1:], num_padding, axis=0)
                y_dist = np.concatenate([y_dist, y_dist_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.x_ts = x_ts
        self.y_ts = y_ts
        self.y_dist = y_dist

    def shuffle(self):
        """洗牌"""
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        y_ts = self.y_ts[permutation]
        x_ts = self.x_ts[permutation]
        # y_dist = self.y_dist[permutation]

        self.xs = xs
        self.ys = ys
        self.x_ts = x_ts
        self.y_ts = y_ts
        # self.y_dist = y_dist

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                x_ts_i = self.x_ts[start_ind: end_ind, ...]
                y_ts_i = self.y_ts[start_ind: end_ind, ...]

                if self.y_dist is not None:
                    y_dist_i = self.y_dist[start_ind: end_ind, ...]
                    yield x_i, y_i, x_ts_i, y_ts_i, y_dist_i
                else:
                    yield x_i, y_i, x_ts_i, y_ts_i
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    """标准转换器"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class NScaler:
    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class MinMax01Scaler:
    """最大最小值01转换器"""
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min


class MinMax11Scaler:
    """最大最小值11转换器"""
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min


def load_dataset(dataset_dir, normalizer, batch_size, valid_batch_size=None, test_batch_size=None, column_wise=False):
    """
    加载数据集
    :param dataset_dir: 数据集目录
    :param normalizer: 归一方式
    :param batch_size: batch大小
    :param valid_batch_size: 验证集batch大小
    :param test_batch_size: 测试集batch大小
    :param column_wise: 是指列元素的级别上进行归一，否则是全样本取值
    """
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        data['y_slot_' + category] = cat_data['y_slot']  # 标签对应的时间段
        data['x_slot_' + category] = cat_data['x_slot']  # 标签对应的时间段
        # if category == 'train':
        #     data['y_distribution'] = cat_data['y_distribution']

    if normalizer == 'max01':
        if column_wise:
            minimum = data['x_train'].min(axis=0, keepdims=True)
            maximum = data['x_train'].max(axis=0, keepdims=True)
        else:
            minimum = data['x_train'].min()
            maximum = data['x_train'].max()

        scaler = MinMax01Scaler(minimum, maximum)
        print('Normalize the dataset by MinMax01 Normalization')

    elif normalizer == 'max11':
        if column_wise:
            minimum = data['x_train'].min(axis=0, keepdims=True)
            maximum = data['x_train'].max(axis=0, keepdims=True)
        else:
            minimum = data['x_train'].min()
            maximum = data['x_train'].max()

        scaler = MinMax11Scaler(minimum, maximum)
        print('Normalize the dataset by MinMax11 Normalization')

    elif normalizer == 'std':
        if column_wise:
            mean = data['x_train'].mean(axis=0, keepdims=True)  # 获得每列元素的均值、标准差
            std = data['x_train'].std(axis=0, keepdims=True)
        else:
            mean = data['x_train'].mean()
            std = data['x_train'].std()

        scaler = StandardScaler(mean, std)
        print('Normalize the dataset by Standard Normalization')

    elif normalizer == 'None':
        scaler = NScaler()
        print('Does not normalize the dataset')
    else:
        raise ValueError

    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], data['x_slot_train'], data['y_slot_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], data['x_slot_val'], data['y_slot_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], data['x_slot_test'], data['y_slot_test'], test_batch_size)
    data['scaler'] = scaler

    return data


"""指标"""


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)

    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()

    return mae, mape, rmse


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def compute_loss(y_true, y_predicted, null_val=np.nan):
    return masked_mae_loss(y_predicted, y_true, null_val)


def masked_mae_loss(y_pred, y_true, null_val=np.nan):
    mask = (y_true != null_val).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()


def change_input(x_truth):
    # with open('data/processed/PEMS04/PEMS04_flow_count.pkl', 'rb') as f:
    #     pickle_data = pickle.load(f)[0]
    # num_nodes, timeslot = pickle_data.shape
    # pickle_data = np.reshape(pickle_data, (num_nodes, 7, 288))
    # pickle_data = np.mean(pickle_data, axis=1)  # (num_nodes, 288)  # 每个结点在288个时间段的均值
    # pickle_data = torch.from_numpy(pickle_data).to(self.device)
    x_train_value = copy.deepcopy(x_truth)  # 深拷贝 (B, T, N, C)
    batch_size, seq_len, num_sensor, input_dim = x_train_value.shape
    # x_train_value = torch.reshape(x_train_value, (batch_size, seq_len, num_sensor))  # (B, T, N)
    # 往前替换非0值
    for bat in range(batch_size):
        indices = []
        for node in range(num_sensor):  # 确定x_train_val最后一个时段为0值的元素所对应的索引
            if x_train_value[bat, -1, node, 0] == 0:
                indices.append(node)

        for ind in indices:
            for t in range(seq_len - 1)[::-1]:  # 把x_train_val最后一个时段为0值所对应的元素往前替换为最新的那个
                if x_train_value[bat, t, ind, 0] != 0:
                    x_train_value[bat, -1, ind, 0] = x_train_value[bat, t, ind, 0]
                    break

        # # 用这个时段的均值替换0
        # cur_timeslot = copy.deepcopy(x_ts[bat][-1])  # 最后一个时刻对应的时间段, 整数
        # for ind in indices:
        #     if x_train_value[bat, -1, ind] == 0:
        #         x_train_value[bat][-1][ind] = pickle_data[ind][cur_timeslot]

    x_truth = x_train_value
    # x_truth = torch.reshape(x_truth, (batch_size, seq_len, num_sensor, input_dim))
    return x_truth


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)  # adj转换为稀疏矩阵
    d = np.array(adj.sum(1))  # Sum the matrix elements over a given axis.
    d_inv_sqrt = np.power(d, -0.5).flatten()  # D^-1/2的铺开形式
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # 测试元素是否为正无穷大或负无穷大,把无穷大元素变为0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # 从对角线构造一个稀疏矩阵, D^-1/2
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian  # L


if __name__ == '__main__':
    adj = get_adjacency_matrix("./data/PEMS04/PEMS04.csv", 307, id_filename=None)
    print(adj)
    A = construct_adj(adj, 3)
    print(A.shape)
    print(A)

    dataloader = load_dataset('./data/processed/PEMS04/', 'std', batch_size=64, valid_batch_size=64, test_batch_size=64)
    print(dataloader)






