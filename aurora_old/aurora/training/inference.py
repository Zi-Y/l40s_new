import torch
from torch.amp import autocast
import time
import os, pickle
from aurora_old.training.loss import AuroraMeanAbsoluteError
from aurora_old.training.logging import log_metrics, print_time, log_message
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import plotly.graph_objects as go

# The following settings are to solve the problem:
# RuntimeError: CUDA error: invalid configuration argument (if image size >= 1024x1024)
# refer to https://stackoverflow.com/questions/77343471/pytorch-cuda-error-invalid-configuration-argument
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# set cudnn to be deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

VARIABLE_UNITS = {
    "msl": "hPa ",
    "2t": "°C",
    "10u": "m/s",
    "10v": "m/s",
    "tp": "mm",
}

VARIABLE_NAMES = {
    "msl": "Mean Sea Level Pressure",
    "2t": "2m Temperature",
    "10u": "10m U Wind Component",
    "10v": "10m V Wind Component",
    "tp": "Total Precipitation",
}


# def unnormalise(stats: dict[str, tuple[float, float]]) -> "Batch":
#     """Unnormalise all variables in the batch.
#
#     Args:
#         stats (dict[str, tuple[float, float]]): Custom variable statistics. For variables where no
#             statistics are provided, the Aurora default statistics are used.
#
#     Returns:
#         :class:`.Batch`: Unnormalised batch.
#     """
#     return Batch(
#         surf_vars={
#             k: unnormalise_surf_var(v, k, stats=stats) for k, v in self.surf_vars.items()
#         },
#         static_vars={
#             k: unnormalise_surf_var(v, k, stats=stats) for k, v in self.static_vars.items()
#         },
#         atmos_vars={
#             k: unnormalise_atmos_var(v, k, self.metadata.atmos_levels, stats=stats)
#             for k, v in self.atmos_vars.items()
#         },
#         metadata=self.metadata,
#     )

def inference(model, dataloader, cfg, device):
    """
    Train the given model using mixed precision and an adjustable learning rate schedule.

    Args:
        model (torch.nn.Module): The model to be trained.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the training dataset.
        cfg (DictConfig): Hydra configuration object containing hyperparameters.
        device (torch.device): The device to which the model and data is moved.
    Returns:
        None: The function trains the model in-place

    Notes:
        - Mixed precision is used for improved performance on supported hardware.

    Example:
        >>> inference(model, test_dataloader, cfg, torch.device("cuda:0"))
    """

    criterion = AuroraMeanAbsoluteError(variable_weights=cfg.variable_weights)
    stats = dataloader.dataset.stats
    ckpt_path = os.path.join(cfg.checkpoint.ckpt_dir, cfg.checkpoint.ckpt_file)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)

    model.eval()

    start_time = time.time()
    log_message("Data is loaded")
    print_time("inference_start", start_time)

    for i, sample in enumerate(dataloader):
        batch = sample["input"].to(device, non_blocking=True)
        target = sample["target"].to(device, non_blocking=True)
        # sample['input'].metadata.time
        # (datetime.datetime(2007, 1, 2, 6, 0, tzinfo=datetime.timezone.utc),)
        # sample['input'].metadata.time[0].strftime('%Y-%m-%d %H:%M:%S')
        # '2007-01-02 06:00:00'
        print('sample time: ' + sample['input'].metadata.time[0].strftime('%Y-%m-%d %H:%M:%S'))
        # Mixed precision forward pass
        with torch.no_grad():
            with autocast("cuda", dtype=torch.bfloat16):
                prediction = model(batch)
                loss = criterion(prediction, target)

                input_batch = batch.unnormalise(stats)
                pred_batch = prediction.to(torch.float32).unnormalise(stats)
                gt_batch = target.unnormalise(stats)

            for variable_name in list(VARIABLE_UNITS.keys()):
                input_img_list = []
                for i in range(input_batch.surf_vars[variable_name].shape[1]):
                    input_img_list.append(input_batch.surf_vars[variable_name][0, i, :,:].cpu().detach().squeeze().numpy())
                imput_time = input_batch.metadata.time[0].strftime('%Y-%m-%d-%Hh')
                pred_img = pred_batch.surf_vars[variable_name].cpu().detach().to(torch.float32).squeeze().numpy()
                target_img = gt_batch.surf_vars[variable_name].cpu().detach().squeeze().numpy()
                target_time = gt_batch.metadata.time[0].strftime('%Y-%m-%d-%Hh')

                # Re-plot with adjusted title size to prevent truncation
                fig, axs = plt.subplots(1, 5, figsize=(16, 3.5))

                # Plot matrix A
                im1 = axs[0].imshow(input_img_list[0], cmap='coolwarm')
                axs[0].set_title('Input (first frame)', fontsize=10)
                axs[0].axis('off')

                axs[1].imshow(input_img_list[1], cmap='coolwarm')
                axs[1].set_title('Input (second frame)', fontsize=10)
                axs[1].axis('off')

                # Plot matrix C
                axs[2].imshow(pred_img, cmap='coolwarm')
                axs[2].set_title('Predicted result', fontsize=10)
                axs[2].axis('off')

                # Plot matrix B
                axs[3].imshow(target_img, cmap='coolwarm')
                axs[3].set_title('Ground truth', fontsize=10)
                axs[3].axis('off')

                # Plot the difference (A - B)
                D = target_img - pred_img
                min_value = np.min(D)
                max_value = np.max(D)
                cmap = plt.cm.bwr
                norm = mcolors.TwoSlopeNorm(vmin=min_value, vcenter=0, vmax=max_value)
                im4 = axs[4].imshow(D, cmap=cmap, norm=norm)
                axs[4].set_title('Error', fontsize=10)
                axs[4].axis('off')

                # Add a larger color bar to represent the scale
                cbar1 = fig.colorbar(im1, ax=axs[0:4], orientation='horizontal', fraction=0.05, pad=0.05)
                cbar1.ax.tick_params(labelsize=8)

                # Add a separate color bar for the fourth plot
                cbar2 = fig.colorbar(im4, ax=axs[4], orientation='horizontal', fraction=0.05, pad=0.05)
                cbar2.ax.tick_params(labelsize=8)

                saved_path = './inference_results_test/aurora_large/'
                saved_path += 'time_' + target.metadata.time[0].strftime('%Y-%m-%d-%H') + '/'


                fig.suptitle('Variable of ' + variable_name + f' at {target_time}, loss = {loss.item():.3f}', fontsize=14)

                os.makedirs(saved_path, exist_ok=True)
                plt.savefig(saved_path + f'{variable_name}_inference_results', dpi=300, bbox_inches='tight')
                print('save to: ' + saved_path + f'{variable_name}_inference_results.jpg')
                plt.close()

                if variable_name =='tp':
                    # # 1. 明确你要显示的数值范围（可以用 np.min(D) / np.max(D)，
                    # #    这里以 -20 到 80 为例）
                    # min_val = np.min(D)
                    # max_val = np.max(D)
                    #
                    # # 2. 计算 0 在整个区间中的位置（占比）
                    # zero_position = (0 - min_val)/(max_val - min_val)  # 0 相对于 -20 的距离除以总宽度 => 0.2
                    #
                    # # 3. 自定义颜色梯度（colorscale）
                    # #   - [0.0, "blue"] 表示在最小值（-20）对应的颜色是蓝色
                    # #   - [zero_position, "white"] 表示在 0 对应颜色是白色
                    # #   - [1.0, "red"] 表示在最大值（80）对应的颜色是红色
                    # custom_colorscale = [
                    #     [0.0, "blue"],
                    #     [zero_position, "white"],
                    #     [1.0, "red"]
                    # ]
                    #
                    # # 4. 创建 Heatmap
                    # fig_html = go.Figure(
                    #     data=go.Heatmap(
                    #         z=D,
                    #         x=list(range(D.shape[1])),
                    #         y=list(range(D.shape[0])),
                    #         # 使用自定义的颜色表
                    #         colorscale=custom_colorscale,
                    #         # 设置固定的 zmin 和 zmax，保证 colorbar 范围一致
                    #         zmin=min_val,
                    #         zmax=max_val,
                    #         # 颜色条标题
                    #         colorbar=dict(title="Values")
                    #     )
                    # )
                    #
                    # # 5. 设置布局，例如标题、坐标轴等
                    # fig_html.update_layout(
                    #     title="Error of TP",
                    #     title_x=0.5,
                    #     xaxis=dict(title="Columns"),
                    #     yaxis=dict(
                    #         title="Rows",
                    #         autorange="reversed"  # y轴翻转
                    #     )
                    # )
                    #
                    # # 6. 保存为 HTML
                    # html_file = saved_path + "Error_of_TP.html"
                    # fig_html.write_html(html_file)
                    save_all = True
                    if save_all:
                        from plotly.subplots import make_subplots

                        # 全局最小/最大值（前四幅图共享色尺）
                        all_four = np.concatenate([
                            input_img_list[0].ravel(),
                            # input_img_list[1].ravel(),
                            # pred_img.ravel(),
                            # target_img.ravel()
                        ])
                        min_val_1 = all_four.min()
                        max_val_1 = all_four.max()

                        # 差异图单独的最小/最大值
                        min_val_2 = D.min()
                        max_val_2 = D.max()
                        zero_position = (0 - min_val_2) / (max_val_2 - min_val_2) if max_val_2 != min_val_2 else 0.5

                        # 自定义色标
                        approx_coolwarm = [
                            [0.0, 'rgb(59,76,192)'], [0.25, 'rgb(127,156,207)'],
                            [0.5, 'rgb(196,196,196)'], [0.75, 'rgb(214,125,85)'], [1.0, 'rgb(180,4,38)']
                        ]
                        diff_colorscale = [
                            [0.0, "blue"], [max(0, min(zero_position, 1)), "white"], [1.0, "red"]
                        ]

                        # 创建子图
                        fig = make_subplots(
                            rows=1, cols=5,
                            subplot_titles=[
                                'Input (first frame)', 'Input (second frame)',
                                'Predicted result', 'Ground truth', 'Error'
                            ],
                            horizontal_spacing=0.03
                        )

                        # 自定义悬停模板
                        hover_template = 'Value: %{z:.2f}<br>Row: %{y}<br>Column: %{x}<extra></extra>'

                        # 前四幅图（共享色尺和colorbar）
                        for i, img in enumerate([input_img_list[0], input_img_list[1], pred_img, target_img], start=1):
                            fig.add_trace(
                                go.Heatmap(
                                    z=img,
                                    colorscale=approx_coolwarm,
                                    zmin=min_val_1,
                                    zmax=max_val_1,
                                    hovertemplate=hover_template,  # 自定义悬停显示
                                    showscale=(i == 1),  # 只在第一个图显示 colorbar
                                    colorbar=dict(
                                        title='Scale (4 images)',
                                        orientation='h',
                                        x=0.4,  # 横向位置（整体下方居中）
                                        y=-0.22,  # 下移
                                        thickness=10,  # 调低高度
                                        len=0.5,  # colorbar 长度
                                        xanchor='center'
                                    ) if i == 1 else None
                                ),
                                row=1, col=i
                            )

                        # 差异图
                        fig.add_trace(
                            go.Heatmap(
                                z=D,
                                colorscale=diff_colorscale,
                                zmin=min_val_2,
                                zmax=max_val_2,
                                hovertemplate=hover_template,  # 自定义悬停显示
                                showscale=True,
                                colorbar=dict(
                                    title='Error',
                                    orientation='h',
                                    x=0.9,  # 横向位置
                                    y=-0.22,  # 下移
                                    thickness=10,  # 调低高度
                                    len=0.2,  # colorbar 长度
                                    xanchor='center'
                                )
                            ),
                            row=1, col=5
                        )

                        # 翻转所有子图的 Y 轴
                        for i in range(1, 6):
                            fig.update_yaxes(autorange="reversed", row=1, col=i)
                            fig.update_xaxes(showticklabels=False, visible=False, row=1, col=i)
                            fig.update_yaxes(showticklabels=False, visible=False, row=1, col=i)

                        # 调整整体布局，添加标题
                        fig.update_layout(
                            width=1500,  # 宽度增大
                            height=400,  # 高度增大
                            margin=dict(l=20, r=20, t=80, b=100),
                            title=dict(
                                text="Visualization of "
                                     f"Variable {variable_name} at {target_time}, loss = {loss.item():.3f}",
                                x=0.5,  # 标题居中
                                font=dict(size=18)
                            )
                        )

                        # 导出 HTML
                        fig.write_html(saved_path + "All_of_TP.html")
            #             saved_path + "Error_of_TP.html"


            print('test')






    inference_duration = time.time() - start_time
    print_time("inference_duration", inference_duration)
