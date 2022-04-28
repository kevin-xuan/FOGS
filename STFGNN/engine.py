import torch
import torch.optim as optim
from model import *
import utils
import numpy as np
import copy
import os
import pickle


class trainer():
    def __init__(self, args, scaler, adj, history, num_of_vertices,
                 in_dim, hidden_dims, first_layer_embedding_size, out_layer_dim,
                 log, lrate, device, dataloader, activation='GLU', use_mask=True, max_grad_norm=5,
                 lr_decay=False, temporal_emb=True, spatial_emb=True, horizon=12, strides=3):
        """
        训练器
        :param args: 参数脚本
        :param scaler: 转换器
        :param adj: local时空间矩阵
        :param history: 输入时间步长
        :param num_of_vertices: 节点数量
        :param in_dim: 输入维度
        :param hidden_dims: lists, 中间各STSGCL层的卷积操作维度
        :param first_layer_embedding_size: 第一层输入层的维度
        :param out_layer_dim: 输出模块中间层维度
        :param log: 日志
        :param lrate: 初始学习率
        :param device: 计算设备
        :param activation:激活函数 {relu, GlU}
        :param use_mask: 是否使用mask矩阵优化adj
        :param max_grad_norm: 梯度阈值
        :param lr_decay: 是否采用初始学习率递减策略
        :param temporal_emb: 是否使用时间嵌入向量
        :param spatial_emb: 是否使用空间嵌入向量
        :param horizon: 预测时间步长
        :param strides: 滑动窗口步长，local时空图使用几个时间步构建的，默认为3
        """
        super(trainer, self).__init__()

        self.model = STSGCN(
            adj=adj,
            history=history,
            num_of_vertices=num_of_vertices,
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            first_layer_embedding_size=first_layer_embedding_size,
            out_layer_dim=out_layer_dim,
            activation=activation,
            use_mask=use_mask,
            temporal_emb=temporal_emb,
            spatial_emb=spatial_emb,
            horizon=horizon,
            strides=strides
        )
        # train embedding
        if args.trend_embedding:
            self.trend_embedding = args.trend_embedding
            self.trend_bias_embeddings = nn.Embedding(288, num_of_vertices * horizon)
            self.device = device
            self._data = dataloader

        self.trend_embedding = args.trend_embedding
        self.device = device
        self._data = dataloader
        self._args = args
        self.num_nodes = num_of_vertices
        self.horizon = horizon

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.model.to(device)

        self.model_parameters_init()
        # train embedding
        if args.trend_embedding:
            self.optimizer = optim.Adam([{'params': self.model.parameters()},
                                         {'params': self.trend_bias_embeddings.parameters()}], lr=lrate, eps=1.0e-8,
                                        weight_decay=0, amsgrad=False)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, eps=1.0e-8, weight_decay=0, amsgrad=False)

        if lr_decay:
            utils.log_string(log, 'Applying learning rate decay.')
            lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                                     milestones=lr_decay_steps,
                                                                     gamma=args.lr_decay_rate)
        self.loss = torch.nn.SmoothL1Loss()
        # self.loss = torch.nn.MSELoss(reduction='none')
        self.scaler = scaler
        self.clip = max_grad_norm

        utils.log_string(log, "模型可训练参数: {:,}".format(utils.count_parameters(self.model)))
        utils.log_string(log, 'GPU使用情况:{:,}'.format(
            torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0))

    def model_parameters_init(self):
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.0003)
            else:
                nn.init.uniform_(p)

    def train(self, input, real_val, realx_slot, realy_slot):
        """
        x shape:  (16969, 12, 307, 1) , y shape:  (16969, 12, 307)
        :param input: B, T, N, C
        :param real_val: B, T, N
        :param realx_slot: B, T
        :param realy_slot: B, T
        :param realy_dist: B, T, N, C
        """

        self.model.train()
        if self.trend_embedding:
            self.trend_bias_embeddings.train()
        self.optimizer.zero_grad()

        output = self.model(input)  # B, T, N  # 换成趋势后,output变成趋势;不换之前，output是归一化的流量值
        if self.trend_embedding:
            trend_time_bias = self.trend_bias_embeddings(realy_slot[:, 0]).to(self.device)  # (B, N * T)
            # (B, N, T) -> (B, T, N)
            trend_time_bias = torch.reshape(trend_time_bias, (-1, self.num_nodes, self.horizon)).permute(0, 2, 1)
            loss = utils.masked_mae(output, real_val) + self._compute_embedding_loss(input, real_val, output, trend_time_bias)
        else:
            if self._args.use_trend:
                loss = utils.masked_mae(output, real_val)  # ①下次试试转换为具体流量值计算MAE
            else:
                output = self.scaler.inverse_transform(output)
                loss = self.loss(output, real_val)

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            if self.trend_embedding:
                torch.nn.utils.clip_grad_norm_(self.trend_bias_embeddings.parameters(), self.clip)

        self.optimizer.step()
        return loss.item()

    # def evel(self, input, real_val):
    #     """
    #     x shape:  (16969, 12, 307, 1) , y shape:  (16969, 12, 307)
    #     :param input: B, T, N, C
    #     :param real_val:B, T, N
    #     """
    #     self.model.eval()
    #
    #     output = self.model(input)  # B, T, N
    #
    #     B, T, N = output.shape
    #     x_truth = self.scaler.inverse_transform(input).reshape(B, T, -1)  # 把x_truth也变成(B,T,N)
    #     x_truth = x_truth.permute(1, 0, 2)  # (B,T,N)->(T,B,N)
    #     output = output.permute(1, 0, 2)  # (T, B, N)
    #     if self.trend_embedding:
    #         inputs = [list(range(N)) for _ in range(B)]
    #         bias = self.trend_bias_embeddings(torch.LongTensor(inputs))  # (B, N, T)
    #         bias = bias.permute(0, 2, 1)  # (B, N, T)->(B, T, N)
    #         bias = bias.to(self.device)
    #         predict = (1 + output) * x_truth[-1] + bias  # 将预测趋势变成流量值
    #     else:
    #         predict = (1 + output) * x_truth[-1]  # 将预测趋势变成流量值
    #
    #     predict = predict.permute(1, 0, 2)  # (B, T, N)
    #
    #     mae = utils.masked_mae(predict, real_val, 0.0).item()  # 验证集和测试集的标签未改成趋势，所以real_val不需改变
    #     mape = utils.masked_mape(predict, real_val, 0.0).item()
    #     rmse = utils.masked_rmse(predict, real_val, 0.0).item()
    #
    #     return mae, mape, rmse

    def evaluate(self, dataset='val'):
        with torch.no_grad():
            self.model.eval()
            if self.trend_embedding:
                self.trend_bias_embeddings.eval()
            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            losses = []
            y_truths = []
            y_preds = []
            for _, (x, y, x_ts, y_ts) in enumerate(val_iterator):
                valx = torch.Tensor(x).to(self.device)  # B, T, N, C
                valy = torch.Tensor(y[:, :, :, 0]).to(self.device)  # B, T, N
                valx_slot = torch.LongTensor(x_ts)  # (B, T)
                valy_slot = torch.LongTensor(y_ts)  # (B, T)

                output = self.model(valx)  # B, T, N  # 趋势或者归一化的流量值
                if self._args.use_trend:
                    B, T, N = output.shape
                    x_truth = self.scaler.inverse_transform(valx).reshape(B, T, -1)  # 把x_truth也变成(B,T,N)
                    x_truth = x_truth.permute(1, 0, 2)  # (B,T,N)->(T,B,N)
                    output = output.permute(1, 0, 2)  # (T, B, N)

                    if self.trend_embedding:
                        bias = self.trend_bias_embeddings(valy_slot[:, 0])  # (B, N * T)
                        bias = torch.reshape(bias, (-1, self.num_nodes, self.horizon))  # (B, N, T)
                        bias = bias.permute(2, 0, 1)  # (B, N, T)->(T, B, N)
                        bias = bias.to(self.device)
                        predict = (1 + output) * x_truth[-1] + bias  # 将预测趋势变成流量值  (T, B, N)
                    else:
                        predict = (1 + output) * x_truth[-1]  # 将预测趋势变成流量值

                    predict = predict.permute(1, 0, 2)  # (T, B, N)->(B, T, N)
                else:
                    predict = self.scaler.inverse_transform(output)

                loss = utils.masked_mae(predict, valy, 0.0).item()

                losses.append(loss)
                y_truths.append(valy.cpu())
                y_preds.append(predict.cpu())

            mean_loss = np.mean(losses)
            y_preds = np.concatenate(y_preds, axis=0)  # 是numpy矩阵, shape是(len, horizon, num_sensor * output_dim)
            y_truths = np.concatenate(y_truths, axis=0)  # concatenate on batch dimension
            return mean_loss, {'prediction': y_preds, 'truth': y_truths}

    def _compute_embedding_loss(self, x, y_true, y_pred, bias, null_val=np.nan):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim) tensor
        :param y_pred: shape (batch_size, seq_len, num_sensor * input_dim) tensor
        :param y_true: shape (batch_size, horizon, num_sensor * input_dim)
        :param bias: shape (batch_size, horizon, num_sensor * input_dim)
        """
        x = x.squeeze()  # (B, T, N)
        x_truth = self.scaler.inverse_transform(x)  # 把x逆标准化转换为真实流量值
        # (B, T, N) -> (T, B, N)
        x_truth = x_truth.permute(1, 0, 2)
        y_true = y_true.permute(1, 0, 2)
        y_pred = y_pred.permute(1, 0, 2)

        labels = (1 + y_true) * x_truth[-1] - (1 + y_pred) * x_truth[-1]  # (T, B, N)
        bias = bias.to(self.device)
        labels = labels.to(self.device)
        # (T, B, N) -> (B, T, N)
        labels = labels.permute(1, 0, 2)

        return utils.masked_mae_loss(bias, labels, null_val)



