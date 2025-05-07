from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from infobatch import InfoBatch
import math

import copy
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # def _select_criterion(self):
    #     criterion = nn.MSELoss()
    #     return criterion
    def _select_criterion(self):
        # 【Modified】：当使用概率模型时，根据用户选择的损失函数决定 criterion
        if self.args.probabilistic:
            loss_func = self.args.probabilistic_loss_function
            if loss_func == "torchGaussianNLL":
                def gaussian_nll_loss_wrapper(pred, true):
                    # pred 是一个二元组 [mu, sigma]
                    mu, sigma = pred
                    # 确保sigma不小于eps
                    sigma = torch.clamp(sigma, min=1e-6)
                    # 注意：torch.nn.GaussianNLLLoss要求输入: mean, target, variance (sigma^2)
                    # 这里设置 reduction="mean" 且 eps=1e-6
                    loss_fn = torch.nn.GaussianNLLLoss(eps=1e-6, reduction="mean")
                    # loss_fn = torch.nn.GaussianNLLLoss(eps=1e-6, reduction="mean")
                    # loss = loss_fn(mu, true, sigma ** 2)
                    return loss_fn(mu, true, sigma ** 2)
                criterion = gaussian_nll_loss_wrapper

            elif loss_func == "CRPS":
                # 采用高斯CRPS的闭式解: CRPS(mu, sigma, y) = sigma * [1/sqrt(pi) - 2*phi(z) - z*(2*Phi(z)-1)]
                def gaussian_crps_loss(pred, true):
                    mu, sigma = pred
                    sigma = torch.clamp(sigma, min=1e-6)  # 防止sigma过小
                    z = (true - mu) / sigma
                    pdf = 1.0 / math.sqrt(2 * math.pi) * torch.exp(-0.5 * z**2)
                    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
                    crps = sigma * (1/math.sqrt(math.pi) - 2 * pdf - z * (2 * cdf - 1))
                    return crps.mean()
                criterion = gaussian_crps_loss
            else:
                def custom_gaussian_nll_loss(pred, true):
                    mu, sigma = pred
                    sigma = torch.clamp(sigma, min=1e-6)  # 不允许sigma过小
                    # NLL loss: 0.5*log(2*pi*sigma^2) + ((y-mu)^2)/(2*sigma^2)
                    loss = 0.5 * torch.log(2 * math.pi * sigma**2) + ((true - mu)**2) / (2 * sigma**2)
                    return loss.mean()
                criterion = custom_gaussian_nll_loss
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        MSE_loss = []
        MAE_loss = []
        criterion_MSE = nn.MSELoss(reduction='none')
        self.model.eval()

        sum_mse = torch.tensor(0.0, device=self.device)
        sum_mae = torch.tensor(0.0, device=self.device)
        total_samples = 0

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, sample_id, sample_weight) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.probabilistic:
                    # 【Modified】处理概率模式：提取 mu, sigma，并取最后 pred_len 时间步
                    mu, sigma = outputs
                    mu = mu[:, -self.args.pred_len:, :]
                    sigma = sigma[:, -self.args.pred_len:, :]
                    loss = criterion([mu, sigma], batch_y[:, -self.args.pred_len:, :])
                    loss_mse = criterion_MSE(mu, batch_y[:, -self.args.pred_len:, :])
                    loss_mae = torch.mean(torch.abs(mu - batch_y[:, -self.args.pred_len:, :]))

                else:
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss_mse = criterion_MSE(outputs, batch_y)
                    loss_mse = loss_mse.mean(dim=(1, 2))
                    loss_mae = torch.abs(outputs - batch_y)
                    loss_mae = loss_mae.mean(dim=(1, 2))
                    # loss = loss_mse
                    # loss = criterion(outputs, batch_y)

                    sum_mse += loss_mse.sum()
                    sum_mae += loss_mae.sum()
                    total_samples += loss_mse.numel()  # B

                # 5) 在 GPU 上算平均
                avg_mse = sum_mse / total_samples
                avg_mae = sum_mae / total_samples


                # total_loss.append(loss.item())

                # MSE_loss.extend(loss_mse.tolist())
                # MAE_loss.extend(loss_mae.tolist())

        # total_loss = np.average(total_loss)
        MSE_loss = avg_mse.cpu().numpy()
        MAE_loss = avg_mae.cpu().numpy()
        total_loss = MSE_loss
        self.model.train()
        # 1. MSE loss, 2. MAE, 3. pre-selected loss,
        return MSE_loss, MAE_loss, total_loss

    def infer_train_set(self, vali_data, vali_loader, criterion, iterations, setting):

        criterion_MSE = nn.MSELoss(reduction='none')
        loss_all_sample_all_variable_all_token = np.zeros((len(vali_data), self.args.pred_len, self.args.enc_in))
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, sample_id, sample_weight) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.probabilistic:
                    # 【Modified】处理概率模式：提取 mu, sigma，并取最后 pred_len 时间步
                    mu, sigma = outputs
                    mu = mu[:, -self.args.pred_len:, :]
                    sigma = sigma[:, -self.args.pred_len:, :]
                    loss = criterion([mu, sigma], batch_y[:, -self.args.pred_len:, :])




                else:
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss_mse = criterion_MSE(outputs, batch_y)
                    sample_loss_metric = loss_mse.detach().cpu().numpy()
                    for j, sid in enumerate(sample_id):  # 遍历当前batch的样本id
                        loss_all_sample_all_variable_all_token[int(sid)] = sample_loss_metric[j]



        epoch_results_path = os.path.join("/mnt/ssd/zi/itransformer_results/new/", setting)
        if not os.path.exists(epoch_results_path):
            os.makedirs(epoch_results_path)
        epoch_results_path = os.path.join(epoch_results_path,
                                          f'iter_{iterations}_train_set_all_sample_all_tokens.npy')
        np.save(epoch_results_path, loss_all_sample_all_variable_all_token)


        self.model.train()
        # 1. MSE loss, 2. MAE, 3. pre-selected loss,


    def train(self, setting):

        if self.args.pruning_method in (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19):

            train_data, _ = self._get_data(flag='train')

            steps_per_epoch = len(train_data) // self.args.batch_size  # 丢弃不能填充完整 batch 的数据
            total_epoch = self.args.train_iterations / steps_per_epoch if steps_per_epoch > 0 else 0
            total_epoch = math.ceil(total_epoch)

            train_data = InfoBatch(train_data, total_epoch,
                                   prune_ratio=self.args.pruning_rate,
                                   delta=self.args.infobatch_delta,
                                   args=self.args)

            from torch.utils.data import DataLoader
            train_loader = DataLoader(
                train_data,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,  # 让数据固定在内存，加快 GPU 访问
                prefetch_factor=2,  # 提前加载数据
                drop_last=True,
                sampler=train_data.sampler)
        else:
            train_data, train_loader = self._get_data(flag='train')

        steps_per_epoch = len(train_data) // self.args.batch_size  # 丢弃不能填充完整 batch 的数据
        total_epoch = self.args.train_iterations / steps_per_epoch if steps_per_epoch > 0 else 0
        total_epoch = math.ceil(total_epoch)

        infer_data, infer_loader = self._get_data(flag='infer')

        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')



        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        criterion_train = nn.MSELoss(reduction='none')

        if self.args.save_per_sample_loss:
            # if self.args.data_path=='electricity.csv':
            loss_all_epoch = np.zeros((len(train_data), total_epoch+1))
            loss_all_epoch[:, 0] = np.arange(len(train_data))  # 【Modified】保存样本id到第一列
            attention_all_epoch = np.zeros((len(train_data), self.args.train_epochs + 1, 15000))
            loss_all_epoch_all_variable_all_stamp = np.zeros((len(train_data), total_epoch, self.args.pred_len, self.args.enc_in))
            # attention_all_epoch = np.zeros((len(train_data), total_epoch, 1875))


        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        global_step = 0
        use_early_stop = False
        # for epoch in range(self.args.train_epochs):

        for epoch in range(total_epoch):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, sample_id, sample_weight) in enumerate(train_loader):
                iter_count += 1
                global_step += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    pass
                else:
                    if self.args.output_attention:
                        outputs, attentions = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    if self.args.probabilistic:
                        # 【Modified】提取概率模式输出
                        mu, sigma = outputs
                        mu = mu[:, -self.args.pred_len:, f_dim:]
                        sigma = sigma[:, -self.args.pred_len:, f_dim:]
                        batch_y_proc = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion([mu, sigma], batch_y_proc)
                    else:
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y_proc = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        if self.args.pruning_method in (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19):

                            loss = criterion_train(outputs, batch_y_proc)

                            if self.args.pruning_method == 5:
                                loss = loss.mean(dim=(1, 2))
                                loss = train_data.update(-loss)
                                loss = -loss


                            elif self.args.pruning_method in (9, 10, 11, 12, 16):
                                loss = train_data.update(loss)
                            # 第一个epoch不计算weighted loss值 只记录当前的loss
                            elif self.args.pruning_method in (13, 14, 15, 17):
                                if epoch == 0:
                                    loss = train_data.update(loss, only_update_saved_loss_metric=True)
                                else:
                                    loss = train_data.update(loss, only_update_saved_loss_metric=False)


                            else:
                                loss = loss.mean(dim=(1, 2))
                                loss = train_data.update(loss)
                        else:
                            loss = criterion(outputs, batch_y_proc)

                    # else:
                    #     if self.args.probabilistic:
                    #         loss = criterion([mu, sigma], batch_y_proc)
                    #     else:
                    #         loss = criterion(outputs, batch_y_proc)

                    train_loss.append(loss.item())
                    # 【Modified】记录每个样本的loss
                    if self.args.save_per_sample_loss:
                        sample_loss_metric = ((outputs - batch_y_proc) ** 2)
                        sample_loss = (sample_loss_metric.mean(dim=[1,2])).detach().cpu().numpy()
                        sample_loss_metric = sample_loss_metric.detach().cpu().numpy()# 计算每个样本的MSE
                        for j, sid in enumerate(sample_id):  # 遍历当前batch的样本id
                            loss_all_epoch[int(sid), epoch+1] = sample_loss[j]
                            loss_all_epoch_all_variable_all_stamp[int(sid), epoch]=sample_loss_metric[j]


                            if self.args.output_attention:
                                attention_all_epoch[int(sid), epoch] = torch.cat([item.view(32, -1) for item in attentions], dim=1).detach().cpu().numpy()[j]
                                # attention_all_epoch[int(sid), epoch] = torch.cat([item.mean(dim=1).view(32, -1) for item in attentions], dim=1).detach().cpu().numpy()[j]
                            # 保存loss到对应epoch的列

                if (global_step) % 100 == 0:

                    # self.infer_train_set(infer_data, infer_loader, criterion, global_step, setting)

                    vali_MSE_loss, vali_MAE_loss, vali_total_loss = self.vali(vali_data, vali_loader, criterion)
                    test_MSE_loss, test_MAE_loss, test_total_loss = self.vali(test_data, test_loader, criterion)


                    iteration_results_path = os.path.join(self.args.checkpoints, setting, f'iter_{global_step}_results.npy')

                    np.save(iteration_results_path, np.array([global_step,
                                                          epoch + 1, 0, vali_MSE_loss, vali_MAE_loss,
                                                          vali_total_loss,
                                                          test_MSE_loss, test_MAE_loss, test_total_loss]))
                    print("\titers: {0}, epoch: {1} | loss: {2:.6f}, val loss: {3:.6f}, test loss: {4:.6f}"
                          .format(global_step, epoch, loss, vali_total_loss, test_total_loss))

                if (i + 1) % 300 == 0:
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)



            # # 1. MSE loss, 2. MAE, 3. pre-selected loss,

            vali_MSE_loss, vali_MAE_loss, vali_total_loss = self.vali(vali_data, vali_loader, criterion)
            test_MSE_loss, test_MAE_loss, test_total_loss = self.vali(test_data, test_loader, criterion)


            epoch_results_path = os.path.join(self.args.checkpoints, setting, f'epoch_{epoch}_results.npy')

            np.save(epoch_results_path, np.array([global_step,
                                                  epoch + 1, train_loss, vali_MSE_loss, vali_MAE_loss, vali_total_loss,
                                                  test_MSE_loss, test_MAE_loss, test_total_loss]))

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_total_loss, test_total_loss))


            # vali_MSE_loss, vali_MAE_loss, vali_total_loss = self.vali(vali_data, vali_loader, criterion)
            # test_MSE_loss, test_MAE_loss, test_total_loss = self.vali(test_data, test_loader, criterion)
            #
            # epoch_results_path = os.path.join(self.args.checkpoints, setting, f'epoch_{epoch}_results.npy')
            #
            # np.save(epoch_results_path, np.array([global_step,
            #                                       epoch + 1, train_loss, vali_MSE_loss, vali_MAE_loss, vali_total_loss,
            #                                       test_MSE_loss, test_MAE_loss, test_total_loss]))
            #
            # print("Epoch: {0}, Steps: {1} | loss: {2:.6f}, val loss: {3:.6f}, test loss: {4:.6f}"
            #       .format(global_step, epoch, train_loss, vali_total_loss, test_total_loss))





            # 1,

            # if self.args.output_attention:
            #     import matplotlib.pyplot as plt
            #
            #     # 假设 result 是你的 tensor 列表
            #     result = attentions  # 这里需要替换为你的实际数据
            #
            #     # 根目录
            #     output_root = "output_images"
            #     os.makedirs(output_root, exist_ok=True)
            #
            #     # 遍历 list，分别保存每个 batch 的 25x25 图片
            #     for list_idx, tensor in enumerate(result):
            #         tensor = tensor.cpu().detach()  # 确保 tensor 在 CPU 上
            #         for batch_idx in range(tensor.shape[0]):  # 32 batches
            #             # 选取一个通道的 25x25 图片（这里选择通道平均）
            #             image = tensor[batch_idx].mean(dim=0).numpy()  # 计算 8 个通道的均值
            #
            #             # 也可以选择单个通道，例如通道 0
            #             # image = tensor[batch_idx, 0].numpy()
            #
            #             # 创建画布
            #             fig, ax = plt.subplots()
            #             img = ax.imshow(image, cmap="gray")  # 以灰度图显示
            #             plt.axis("off")  # 不显示坐标轴
            #
            #             # 添加 color bar
            #             cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
            #             cbar.ax.tick_params(labelsize=8)  # 调整 color bar 标签大小
            #
            #             # 构造文件名
            #             filename = os.path.join(output_root, f"layer{list_idx}_epoch{epoch + 1}_batch{batch_idx}.png")
            #             plt.savefig(filename, bbox_inches="tight", pad_inches=0.1, dpi=300)
            #             plt.close()
            #
            #     print("图片保存完成！")

            if use_early_stop:
                early_stopping(vali_total_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        if self.args.save_per_sample_loss:
            print("Saving model weights")
            torch.save(self.model.state_dict(), path + '/' + f'checkpoint_epoch_{epoch}_seed_{self.args.seed}.pth')

            print(f'Saving loss_all_epoch_seed_{self.args.seed}.npy')
            np.save(os.path.join(self.args.checkpoints, setting, f'loss_all_epoch_seed_{self.args.seed}.npy'),
                    loss_all_epoch)

            print(f'Saving attention_all_epoch_seed_{self.args.seed}.npy')
            np.save(os.path.join(self.args.checkpoints, setting, f'attention_all_epoch_seed_{self.args.seed}.npy'),
                    attention_all_epoch)

            print(f'Saving loss_all_epoch_all_variable_all_stamp{self.args.seed}.npy')
            np.save(os.path.join(self.args.checkpoints, setting, f'loss_all_epoch_all_variable_all_stamp{self.args.seed}.npy'),
                    loss_all_epoch_all_variable_all_stamp)


        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, sample_id, sample_weight) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                if self.args.probabilistic:
                    # 【Modified】在概率模式下，使用预测均值作为点预测
                    mu, _ = outputs
                    outputs = mu[:, -self.args.pred_len:, :]
                else:
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, sample_id, sample_weight) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return


class Exp_Long_Term_Forecast_GRPO(Exp_Basic):
    def __init__(self, args):
        """
        初始化实验类，加载基础设置并构建GRPO的选择网络。
        """
        super(Exp_Long_Term_Forecast_GRPO, self).__init__(args)
        # 样本总数，用于选择网络输入输出维度
        self.num_samples = len(self.train_data)
        hidden_dim = args.selector_hidden_dim
        # 选择网络：输入每个样本的历史损失，输出每个样本被选中概率
        self.selector_net = nn.Sequential(
            nn.Linear(1, hidden_dim),  # 每个样本输入其历史损失值
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 输出每个样本的优先级得分
            nn.Sigmoid()  # 输出选择概率
        ).to(self.device)
        # 选择网络的优化器
        self.optimizer_selector = optim.Adam(
            self.selector_net.parameters(),
            lr=args.selector_lr
        )

    def train(self, setting):
        """
        训练函数：整合GRPO策略与iTransformer训练
        """
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        train_loader = self._get_data_loader(
            self.train_data,
            batch_size=self.args.batch_size,
            shuffle=True
        )
        vali_loader = self._get_data_loader(
            self.vali_data,
            batch_size=self.args.batch_size,
            shuffle=False
        )

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 记录每个样本的历史损失
        loss_record = torch.ones(len(self.train_data), device=self.device)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            # ===== GRPO策略选择样本 =====
            subset_indices_list = []
            log_probs = []
            rewards = []

            # 多候选采样
            for g in range(self.args.num_candidates):
                # 获取每个样本的选择概率
                probs = self.selector_net(loss_record.unsqueeze(1)).squeeze()

                # 采样子集
                indices = torch.nonzero(
                    torch.bernoulli(probs)
                ).squeeze(1)
                if len(indices) < self.args.batch_size:  # 确保至少有一个批次
                    indices = torch.randperm(len(self.train_data))[:self.args.batch_size]

                log_prob = torch.sum(torch.log(probs[indices]))
                subset_indices_list.append(indices)
                log_probs.append(log_prob)

                # 评估子集
                subset_loader = self._get_data_loader(
                    Subset(self.train_data, indices.cpu().numpy()),
                    batch_size=self.args.batch_size,
                    shuffle=True
                )

                val_loss = self.vali(vali_loader)
                rewards.append(-val_loss)

            # 选择最佳子集
            best_idx = np.argmax(rewards)
            best_subset = subset_indices_list[best_idx]
            best_loader = self._get_data_loader(
                Subset(self.train_data, best_subset.cpu().numpy()),
                batch_size=self.args.batch_size,
                shuffle=True
            )

            # 更新选择网络
            advantages = torch.tensor(rewards, device=self.device) - torch.mean(
                torch.tensor(rewards, device=self.device))
            selector_loss = -sum(adv * lp for adv, lp in zip(advantages, log_probs)) / self.args.num_candidates

            self.optimizer_selector.zero_grad()
            selector_loss.backward()
            self.optimizer_selector.step()

            # 使用最佳子集训练模型
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, _, _) in enumerate(best_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

            # 更新loss记录
            with torch.no_grad():
                for idx in best_subset:
                    loss_record[idx] = torch.mean(torch.tensor(train_loss, device=self.device))

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader)
            test_loss = self.vali(self.test_loader)

            print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def _get_data_loader(self, dataset, batch_size, shuffle=True):
        """辅助函数：创建数据加载器"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            drop_last=True
        )
