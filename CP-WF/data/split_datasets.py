import numpy as np
import os

data_path = 'processed_data'
data_type = {'train','val','test'}
defense_method = {'WT', 'Adv','mockingbird'}

for  data_type in data_type:
    adv_datasets = [f'{data_path}/cw100_{method}_{data_type}.npz' for method in defense_method]

    # ---------- 读取原始数据 ----------
    orig_train = np.load(f'{data_path}/cw100_{data_type}.npz')
    orig_sequences = orig_train['burst_sequences']
    orig_labels = orig_train['labels']

    # ---------- 定义类别集合 ----------
    C_common    = np.arange(0, 40)    # 交集类别
    C_A_extra   = np.arange(40, 60)   # A 独有类别
    C_B_extra   = np.arange(60, 80)  # B 独有类别
    C_C_extra   = np.arange(80,100)
    # ---------- 索引划分 ----------
    A_classes = set(C_common) | set(C_A_extra)
    B_classes = set(C_common) | set(C_B_extra)
    C_classes = set(C_common) | set(C_C_extra)
    idx_A = np.where(np.isin(orig_labels, list(A_classes)))[0]
    idx_B = np.where(np.isin(orig_labels, list(B_classes)))[0]
    idx_C = np.where(np.isin(orig_labels, list(C_classes)))[0]
    # ---------- 保存原始数据集划分 ----------
    np.savez(f'{data_path}/splited_data/cw100_{data_type}_A.npz',
            burst_sequences=orig_sequences[idx_A],
            labels=orig_labels[idx_A])

    np.savez(f'{data_path}/splited_data/cw100_{data_type}_B.npz',
            burst_sequences=orig_sequences[idx_B],
            labels=orig_labels[idx_B])

    np.savez(f'{data_path}/splited_data/cw100_{data_type}_C.npz',
            burst_sequences=orig_sequences[idx_C],
            labels=orig_labels[idx_C])
    # ---------- 对每个防御数据集做索引划分 ----------
    for adv_file in adv_datasets:
        adv_data = np.load(adv_file)
        adv_sequences = adv_data['burst_sequences']
        adv_labels = adv_data['labels']
        
        # 确保标签序列与原始数据集完全对应
        assert np.array_equal(adv_labels, orig_labels), \
            f"{adv_file} labels do not match original dataset!"
        
        method_name = os.path.basename(adv_file).replace('.npz', '')
        
        # 划分 A
        np.savez(f'{data_path}/splited_data/{method_name}_A.npz',
                burst_sequences=adv_sequences[idx_A],
                labels=adv_labels[idx_A])
        
        # 划分 B
        np.savez(f'{data_path}/splited_data/{method_name}_B.npz',
                burst_sequences=adv_sequences[idx_B],
                labels=adv_labels[idx_B])

        np.savez(f'{data_path}/splited_data/{method_name}_C.npz',
                burst_sequences=adv_sequences[idx_C],
                labels=adv_labels[idx_C])

    print("✅ 数据划分完成：原始及所有防御数据集已按A/B/C索引同步保存")
