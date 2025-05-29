for epoch in range(total_epoch):
    iter_count = 0
    train_loss = []
    self.model.train()
    epoch_time = time.time()

    if self.args.pruning_method in (100,):  # 您的条件
        for i, (train_batch_x_orig, train_batch_y_orig, train_batch_x_mark_orig, train_batch_y_mark_orig,
                sample_id, sample_weight) in enumerate(train_loader):
            iter_count += 1
            global_step += 1

            # 记录原始训练批次大小
            current_train_batch_size = train_batch_x_orig.size(0)

            # 1. 数据准备与合并 (确保在 .to(device) 和 .float() 之后)
            train_batch_x = train_batch_x_orig.float().to(self.device)
            train_batch_y = train_batch_y_orig.float().to(self.device)  # 用于后续和 val_batch_y 拼接

            # 合并输入 X
            combined_batch_x = torch.cat((train_batch_x, val_batch_x), dim=0)

            # 处理和合并目标 Y (batch_y_proc)
            f_dim = -1 if self.args.features == 'MS' else 0
            # 训练目标 (用于计算个体损失和批次训练损失)
            train_batch_y_proc = train_batch_y[:, -self.args.pred_len:, f_dim:]
            # TODO 验证目标 (val_batch_y 已经是处理好的形式)
            # combined_batch_y_proc = torch.cat((train_batch_y_proc, val_batch_y), dim=0) # 用于loss_train_batch_for_optim

            # 处理和合并 X_mark
            if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                combined_batch_x_mark = None
            else:
                train_batch_x_mark = train_batch_x_mark_orig.float().to(self.device)
                # val_batch_x_mark 需要在此作用域可用
                combined_batch_x_mark = torch.cat((train_batch_x_mark, val_batch_x_mark), dim=0)

            # --- 只进行一次前向传播 ---
            all_outputs_raw = self.model(combined_batch_x, combined_batch_x_mark, 0, 0)
            all_outputs = all_outputs_raw[:, -self.args.pred_len:, f_dim:]

            # 获取模型参数
            model_params = list(self.model.parameters())
            # model_param_names = [name for name, _ in self.model.named_parameters()] # 如果需要名字映射

            # 2. 分离输出
            outputs_train_all_batch = all_outputs[:current_train_batch_size]
            outputs_val_all = all_outputs[current_train_batch_size:]

            # 3. 计算损失
            # 验证集损失 (标量)
            loss_val_only = criterion(outputs_val_all, val_batch_y)

            # 训练批次的总损失 (用于优化器更新) - 确保目标对应正确
            # 如果优化器只应基于训练数据:
            # loss_train_batch_for_optim = criterion(outputs_train_all_batch, train_batch_y_proc)
            # 如果优化器基于合并数据 (如您之前的代码):
            # combined_batch_y_target_for_optim = torch.cat((train_batch_y_proc, val_batch_y), dim=0)
            # loss_train_batch_for_optim = criterion(all_outputs, combined_batch_y_target_for_optim)
            # 为清晰起见，我们假设只用训练数据更新：
            # train_loss.append(loss_train_batch_for_optim.item()) # 累加训练损失

            # 使用 reduction='none' 计算每个训练样本的损失向量
            # 确保 criterion_train 和 criterion 类型相同，只是 reduction 不同
            # 例如: criterion_train = nn.MSELoss(reduction='none')
            train_sample_losses_vector = criterion_train(outputs_train_all_batch, train_batch_y_proc)
            train_sample_losses_vector = train_sample_losses_vector.mean(dim=[1, 2])
            loss_train_batch_for_optim = train_sample_losses_vector.mean()

            # train_sample_losses_vector 的形状通常是 (current_train_batch_size,) 或 (current_train_batch_size, seq_len * features)
            # 如果是后者，通常需要 .mean(dim=[1,2,...]) 或 .sum(dim=[1,2,...]) 来得到每个样本的标量损失
            # 假设它已经是 (current_train_batch_size,)
            # 如果不是，例如对于序列 MSE，可能是 criterion_train(outputs_train_all_batch.reshape(current_train_batch_size, -1), train_batch_y_proc.reshape(current_train_batch_size, -1)).mean(dim=1)
            # 具体取决于您的损失函数和数据形状。为简化，假设 train_sample_losses_vector[k] 是第k个样本的标量损失。

            # 4. 计算梯度
            model_optim.zero_grad()

            # 4a. 计算验证集梯度 (dL_val / d_params)
            grad_val_tuple = torch.autograd.grad(loss_val_only, model_params, retain_graph=True,
                                                 create_graph=False)
            active_param_indices = [idx for idx, grad_tensor in enumerate(grad_val_tuple) if
                                    grad_tensor is not None]
            grad_val_list_for_flat = [grad_val_tuple[idx] for idx in active_param_indices]

            # --- 计算验证集梯度的 L2 范数 (在内循环之前计算一次) ---
            flat_grad_val = torch.tensor(0.0, device=self.device)  # 默认值
            l2_norm_flat_val_grad = torch.tensor(0.0, device=self.device)  # 初始化验证集梯度的L2范数

            if grad_val_list_for_flat:
                valid_grads_for_vec = [g for g in grad_val_list_for_flat if isinstance(g, torch.Tensor)]
                if valid_grads_for_vec:
                    flat_grad_val = parameters_to_vector(valid_grads_for_vec)
                    if flat_grad_val.numel() > 0:
                        l2_norm_flat_val_grad = torch.linalg.norm(flat_grad_val)
            # -----------------------------------------------------------

            # 4b. 计算每个训练样本的梯度并计算相关度量
            for batch_idx in range(current_train_batch_size):
                single_loss_i = train_sample_losses_vector[batch_idx]
                grad_train_i_tuple = torch.autograd.grad(single_loss_i, model_params, retain_graph=True,
                                                         create_graph=False)

                grad_list_i_for_flat = []
                for active_idx in active_param_indices:
                    grad_component_train_i = grad_train_i_tuple[active_idx]
                    if grad_component_train_i is not None:
                        grad_list_i_for_flat.append(grad_component_train_i)
                    else:
                        corresponding_val_grad_for_shape = grad_val_tuple[active_idx]
                        grad_list_i_for_flat.append(torch.zeros_like(corresponding_val_grad_for_shape))

                # 初始化本次迭代的度量值
                cosine_similarity_value = torch.tensor(0.0, device=self.device)
                actual_dot_product_value = torch.tensor(0.0, device=self.device)
                l2_norm_flat_train_grad_value = torch.tensor(0.0, device=self.device)

                if grad_list_i_for_flat:
                    valid_train_grads_for_vec = [g for g in grad_list_i_for_flat if isinstance(g, torch.Tensor)]
                    if valid_train_grads_for_vec:
                        current_flat_grad_train_i = parameters_to_vector(valid_train_grads_for_vec)

                        # 确保扁平化后的梯度有效，以及与 flat_grad_val 兼容
                        if isinstance(flat_grad_val, torch.Tensor) and flat_grad_val.numel() > 0 and \
                                current_flat_grad_train_i.numel() == flat_grad_val.numel() and \
                                current_flat_grad_train_i.numel() > 0:

                            # 1. 计算 L2 Norm (训练梯度)
                            l2_norm_flat_train_grad_value = torch.linalg.norm(current_flat_grad_train_i)

                            # 2. 计算实际的点积
                            actual_dot_product_value = torch.dot(current_flat_grad_train_i, flat_grad_val)

                            # 3. 手动计算余弦相似度
                            denominator = l2_norm_flat_train_grad_value * l2_norm_flat_val_grad
                            # 添加一个小的 epsilon 防止除以零
                            if denominator > 1e-9:
                                cosine_similarity_value = actual_dot_product_value / denominator
                            # else: 保持为 0.0

                # 获取样本在完整数据集中的索引
                idx_in_full_dataset = sample_id[batch_idx]

                # 存储原始的 Shapley 分数 (基于缩放后的余弦相似度)
                data_shapley_scores[idx_in_full_dataset, epoch, 0] = 100.0 * l2_norm_flat_train_grad_value.item()
                data_shapley_scores[idx_in_full_dataset, epoch, 1] = 100.0 * actual_dot_product_value.item()
                data_shapley_scores[idx_in_full_dataset, epoch, 2] = 100.0 * cosine_similarity_value.item()

                # --- 添加代码：保存新增的度量值 ---
                # (与之前一样，假设您定义了 self.data_actual_dot_products 等)
                # if hasattr(self, 'data_actual_dot_products'):
                #     self.data_actual_dot_products[idx_in_full_dataset, epoch] = actual_dot_product_value.item()
                #
                # if hasattr(self, 'data_l2_train_grad_norms'):
                #     self.data_l2_train_grad_norms[idx_in_full_dataset, epoch] = l2_norm_flat_train_grad_value.item()
                #
                # if hasattr(self, 'data_raw_cosine_similarity'):
                #     self.data_raw_cosine_similarity[idx_in_full_dataset, epoch] = cosine_similarity_value.item()
                # ------------------------------------

            # 5. 为优化器准备梯度
            self.model.zero_grad()
            loss_train_batch_for_optim.backward()  # retain_graph 默认为 False，将释放计算图

            # 6. 执行优化器步骤
            model_optim.step()
