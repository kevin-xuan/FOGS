import os
import time
import argparse
import configparser
import numpy as np
import torch
import torch.nn as nn
import tqdm

import utils
from engine import trainer
from utils import *
from model import STSGCN
import ast

DATASET = 'PEMS08'  # PEMS03, PEMS04, PEMS07, or PEMS08

# config_file = './config/{}.conf'.format(DATASET)  # exists path error for debugging
dirname, filename = os.path.split(os.path.abspath(__file__))
config_file = os.path.join(dirname, 'config/{}.conf').format(DATASET)
config = configparser.ConfigParser()
config.read(config_file)

parser = argparse.ArgumentParser(description='arguments')

parser.add_argument('--no_cuda', action="store_true", help="没有GPU")
parser.add_argument('--data', type=str, default=config['data']['data'], help='data path')
parser.add_argument('--sensors_distance', type=str, default=config['data']['sensors_distance'], help='节点距离文件')
parser.add_argument('--dtw_distance', type=str, default=config['data']['dtw_distance'], help='dtw文件')
parser.add_argument('--learn_graph', type=str, default=config['data']['learn_graph'], help='learned graph')
parser.add_argument('--column_wise', type=eval, default=config['data']['column_wise'],
                    help='是指列元素的级别上进行归一，否则是全样本取值')
parser.add_argument('--normalizer', type=str, default=config['data']['normalizer'], help='归一化方式')
parser.add_argument('--batch_size', type=int, default=config['data']['batch_size'], help="batch大小")

parser.add_argument('--num_of_vertices', type=int, default=config['model']['num_of_vertices'], help='传感器数量')
parser.add_argument('--construct_type', type=str, default=config['model']['construct_type'],
                    help="构图方式  {connectivity, distance}")
parser.add_argument('--in_dim', type=int, default=config['model']['in_dim'], help='输入维度')
parser.add_argument('--hidden_dims', type=list, default=ast.literal_eval(config['model']['hidden_dims']),
                    help='中间各STSGCL层的卷积操作维度')

parser.add_argument('--first_layer_embedding_size', type=int, default=config['model']['first_layer_embedding_size'],
                    help='第一层输入层的维度')

parser.add_argument('--out_layer_dim', type=int, default=config['model']['out_layer_dim'], help='输出模块中间层维度')

parser.add_argument("--history", type=int, default=config['model']['history'], help="每个样本输入的离散时序")

parser.add_argument("--horizon", type=int, default=config['model']['horizon'], help="每个样本输出的离散时序")

parser.add_argument("--strides", type=int, default=config['model']['strides'], help="滑动窗口步长，local时空图使用几个时间步构建的，默认为3")

parser.add_argument("--temporal_emb", type=eval, default=config['model']['temporal_emb'], help="是否使用时间嵌入向量")

parser.add_argument("--spatial_emb", type=eval, default=config['model']['spatial_emb'], help="是否使用空间嵌入向量")

parser.add_argument("--use_mask", type=eval, default=config['model']['use_mask'], help="是否使用mask矩阵优化adj")

parser.add_argument("--activation", type=str, default=config['model']['activation'], help="激活函数 {relu, GlU}")

parser.add_argument('--seed', type=int, default=config['train']['seed'], help='种子设置')

parser.add_argument("--learning_rate", type=float, default=config['train']['learning_rate'], help="初始学习率")

parser.add_argument("--lr_decay", type=eval, default=config['train']['lr_decay'], help="是否开启初始学习率衰减策略")
parser.add_argument("--lr_decay_step", type=str, default=config['train']['lr_decay_step'], help="在几个epoch进行初始学习率衰减")

parser.add_argument("--lr_decay_rate", type=float, default=config['train']['lr_decay_rate'], help="学习率衰减率")

parser.add_argument('--epochs', type=int, default=config['train']['epochs'], help="训练代数")

parser.add_argument('--print_every', type=int, default=config['train']['print_every'], help='几个batch报训练损失')

parser.add_argument('--save', type=str, default=config['train']['save'], help='保存路径')

parser.add_argument('--expid', type=int, default=config['train']['expid'], help='实验 id')

parser.add_argument('--max_grad_norm', type=float, default=config['train']['max_grad_norm'], help="梯度阈值")

parser.add_argument('--patience', type=int, default=config['train']['patience'], help='等待代数')

parser.add_argument('--log_file', default=config['train']['log_file'], help='log file')
parser.add_argument('--trend_embedding', default=False, type=bool, help='Set to true to train trend embedding.')
parser.add_argument('--use_trend', default=True, type=bool, help='Set to true to use trend to train model.')
parser.add_argument('--direct', default=False, type=bool, help='Set to true to use directed graph to train model.')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

log = open(args.log_file, 'w')
log_string(log, str(args))


def main():
    # load data
    if DATASET == 'PEMS03':
        adj = get_adjacency_matrix(distance_df_filename=args.sensors_distance,
                                   num_of_vertices=args.num_of_vertices,
                                   type_=args.construct_type,
                                   id_filename='data/PEMS03/PEMS03.txt')
    else:
        adj = get_adjacency_matrix(distance_df_filename=args.sensors_distance,
                                   num_of_vertices=args.num_of_vertices,
                                   type_=args.construct_type,
                                   id_filename=None)

    # adj_dtw = np.array(pd.read_csv(args.dtw_distance, header=None))
    adj_dtw = load_pickle(args.learn_graph)  # 换成我们的graph

    # 将有向图变成无向图
    if not args.direct:
        log_string(log, 'Use undirected graph')
        adj_dtw = np.maximum.reduce([adj_dtw, adj_dtw.T])

    # local_adj = construct_adj(A=adj, steps=args.strides)  # STSGCN
    local_adj = construct_adj_fusion(adj, adj_dtw, steps=args.strides)  # STFGNN
    local_adj = torch.FloatTensor(local_adj)

    dataloader = load_dataset(dataset_dir=args.data,
                              normalizer=args.normalizer,
                              batch_size=args.batch_size,
                              valid_batch_size=args.batch_size,
                              test_batch_size=args.batch_size,
                              column_wise=args.column_wise)

    scaler = dataloader['scaler']

    log_string(log, 'loading data...')

    log_string(log, "The shape of localized adjacency matrix: {}".format(local_adj.shape))

    log_string(log, f'trainX: {torch.tensor(dataloader["train_loader"].xs).shape}\t\t '
                    f'trainY: {torch.tensor(dataloader["train_loader"].ys).shape}')
    log_string(log, f'valX:   {torch.tensor(dataloader["val_loader"].xs).shape}\t\t'
                    f'valY:   {torch.tensor(dataloader["val_loader"].ys).shape}')
    log_string(log, f'testX:   {torch.tensor(dataloader["test_loader"].xs).shape}\t\t'
                    f'testY:   {torch.tensor(dataloader["test_loader"].ys).shape}')
    log_string(log, f'mean:   {scaler.mean:.4f}\t\tstd:   {scaler.std:.4f}')
    log_string(log, 'data loaded!')

    engine = trainer(args=args,
                     scaler=scaler,
                     adj=local_adj,
                     history=args.history,
                     num_of_vertices=args.num_of_vertices,
                     in_dim=args.in_dim,
                     hidden_dims=args.hidden_dims,
                     first_layer_embedding_size=args.first_layer_embedding_size,
                     out_layer_dim=args.out_layer_dim,
                     log=log,
                     lrate=args.learning_rate,
                     device=device,
                     dataloader=dataloader,
                     activation=args.activation,
                     use_mask=args.use_mask,
                     max_grad_norm=args.max_grad_norm,
                     lr_decay=args.lr_decay,
                     temporal_emb=args.temporal_emb,
                     spatial_emb=args.spatial_emb,
                     horizon=args.horizon,
                     strides=args.strides)

    # 开始训练
    if args.use_trend:
        log_string(log, 'Use trend')
    else:
        log_string(log, 'Use graph')

    log_string(log, 'compiling model...')
    his_loss = []
    val_time = []
    train_time = []
    best_epoch = 0
    wait = 0
    val_loss_min = float('inf')
    # best_model_wts = None

    for i in range(1, args.epochs + 1):
        if wait >= args.patience:
            log_string(log, f'early stop at epoch: {i:04d}')
            break

        train_loss = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()

        for iter, (x, y, x_ts, y_ts) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            # [B, T, N, C]

            trainy = torch.Tensor(y[:, :, :, 0]).to(device)
            # [B, T, N]
            trainx_slot = torch.LongTensor(x_ts).to(device)  # (B, T)
            trainy_slot = torch.LongTensor(y_ts).to(device)  # (B, T)
            # trainy_dist = torch.LongTensor(y_dist).to(device)  # (B, T, N, C)

            loss = engine.train(trainx, trainy, trainx_slot, trainy_slot)
            train_loss.append(loss)

        if args.lr_decay:
            engine.lr_scheduler.step()

        t2 = time.time()
        train_time.append(t2 - t1)

        # valid_loss = []

        s1 = time.time()
        val_loss, _ = engine.evaluate(dataset='val')

        s2 = time.time()
        # logs = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        # log_string(log, logs.format(i, (s2 - s1)))

        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)

        his_loss.append(val_loss)

        logs = 'Epoch: {:03d}, Train Loss: {:.4f},  Valid Loss: {:.4f}, Training Time: {:.4f}/epoch'
        log_string(log, logs.format(i, mtrain_loss, val_loss, (t2 - t1)))

        if not os.path.exists(args.save):
            os.makedirs(args.save)

        if val_loss <= val_loss_min:
            log_string(
                log,
                f'val loss decrease from {val_loss_min:.4f} to {val_loss:.4f}, '
                f'save model to {args.save + "exp_" + str(args.expid) + "_" + str(round(val_loss, 2)) + "_best_model.pth"}'
            )
            wait = 0
            val_loss_min = val_loss
            best_model_wts = engine.model.state_dict()
            torch.save(best_model_wts,
                       args.save + "exp_" + str(args.expid) + "_" + str(round(val_loss_min, 2)) + "_best_model.pth")

            # 测试
            realy = torch.Tensor(dataloader['y_test'][:, :, :, 0]).to(device)  # 这里是因为你在做batch的时候，可能会padding出新的sample以满足batch_size的要求
            test_loss, result = engine.evaluate(dataset='test')
            prediction = result['prediction'][:realy.size(0)]  # shape为(len, horizon, num_sensor * output_dim)
            test_label = result['truth'][:realy.size(0)]
            prediction = torch.from_numpy(prediction)  # 转换为tensor
            test_label = torch.from_numpy(test_label)
            log_string(log, "Training finished")
            log_string(log, "The valid loss on best model is " + str(round(val_loss_min, 4)))
            tmp_info = []
            for t in range(args.horizon):
                pred = prediction[:, :t + 1, :]
                real = test_label[:, :t + 1, :]

                mae, mape, rmse = metric(pred, real)
                tmp_info.append((mae, mape, rmse))

            mae, mape, rmse = tmp_info[-1]
            # print('\n')
            print('-------------------------------------------------------------')
            logs = 'Test: Epoch: {}, Test MAE: {:.3f}, Test MAPE: {:.4f}, Test RMSE: {:.3f}'
            best_epoch = i
            log_string(log, logs.format(i, mae, mape * 100, rmse))
            print('-------------------------------------------------------------')
            print('\n')
        else:
            wait += 1

        np.save('./history_loss' + f'_{args.expid}', his_loss)

    log_string(log, "Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    log_string(log, "Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    print('\n')
    print('-------------------------------------------------------------')
    log_string(log, "Best epoch: {}".format(best_epoch))


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()

    log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
    log.close()
