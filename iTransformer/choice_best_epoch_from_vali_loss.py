import numpy as np
import glob
import os
import re
#
# # 定义数据文件存放的目录（可变变量）
# data_dir = ("/home/local/zi/research_project/iTransformer/all_saved_seed0_trained_per_token_loss/"
#             "all_saved_seed0_pm0_pr0_low10_high10_start0_int20_test1_iTransformer_custom_ftM_sl96_ll48_pl720_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0/")  # 可以修改为其他路径
#
# # 定义文件名
# file_names = [
#     "epoch_0_results.npy",
#     "epoch_1_results.npy",
#     "epoch_2_results.npy",
#     "epoch_3_results.npy",
#     "epoch_5_results.npy",
#     "epoch_6_results.npy",
#     "epoch_7_results.npy",
#     "epoch_8_results.npy",
#     "epoch_9_results.npy",
#     "epoch_10_results.npy",
# ]
#
# # 构造完整路径
# file_paths = [data_dir + file_name for file_name in file_names]
# # 读取所有 .npy 文件，并合并为一个矩阵
# matrices = [np.load(file_path) for file_path in file_paths]


data_dir = ("/mnt/ssd/zi/itransformer_results/"
            "seed0_pm0_pr0_low10_high10_start0_int20_tr50_"
            "test101_iTransformer_custom_ftM_sl96_ll48_pl96_"
            "dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0")



# 构造匹配文件的路径模式
pattern = os.path.join(data_dir, "iter_*_results.npy")
# 使用 glob 获取所有匹配的文件路径
file_paths = glob.glob(pattern)

# iter_1446_results


# 读取所有 .npy 文件，并合并为一个矩阵
matrices = [np.load(file_path) for file_path in file_paths]
combined_matrix = np.vstack(matrices)  # 每一行对应一个文件的数据

# 定义列名
column_names = [
    "global_step", "epoch", "train_loss",
    "vali_MSE_loss", "vali_MAE_loss", "vali_total_loss",
    "test_MSE_loss", "test_MAE_loss", "test_total_loss"
]

# 选择某几列数据
def select_columns(matrix, column_names, selected_columns):
    """根据列名选择对应的数据"""
    indices = [column_names.index(col) for col in selected_columns]  # 获取对应列索引
    return matrix[:, indices]  # 选取指定列的数据

# 示例：选择 `epoch`, `train_loss`, `vali_total_loss`
# selected_columns = ["epoch", "train_loss", "vali_MSE_loss", "test_MSE_loss",]
selected_columns = ["global_step", "epoch", "vali_MSE_loss", "test_MSE_loss",]
selected_data = select_columns(combined_matrix, column_names, selected_columns)

# 输出所选列数据
print(f"选定列 {selected_columns} 的数据：\n", selected_data)

loss_all_epoch_all_variable_all_stamp = np.load('/home/local/zi/research_project/iTransformer/'
        'all_saved_seed0_trained_per_token_loss/'
        'all_saved_seed0_pm0_pr0_low10_high10_start0_int20_test1'
        '_iTransformer_custom_ftM_sl96_ll48_pl96_dm512_nh8_el3_'
        'dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0/'
        'loss_all_epoch_all_variable_all_stamp0.npy')
print('test')

# all_saved_seed0_pm0_pr0_low10_high10_start0_int20_test1_iTransformer_custom_ftM_sl96_ll48_pl96_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0
# 选择(epoch+1)为6，即loss_all_epoch_all_variable_all_stamp[int(sid), 5]

# all_saved_seed0_pm0_pr0_low10_high10_start0_int20_test1_iTransformer_custom_ftM_sl96_ll48_pl192_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0
# 选择(epoch+1)为4，即loss_all_epoch_all_variable_all_stamp[int(sid), 3]

# all_saved_seed0_pm0_pr0_low10_high10_start0_int20_test1_iTransformer_custom_ftM_sl96_ll48_pl336_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0
# 选择(epoch+1)为2，即loss_all_epoch_all_variable_all_stamp[int(sid), 1]

# all_saved_seed0_pm0_pr0_low10_high10_start0_int20_test1_iTransformer_custom_ftM_sl96_ll48_pl720_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0
# 选择(epoch+1)为7，即loss_all_epoch_all_variable_all_stamp[int(sid), 6]



