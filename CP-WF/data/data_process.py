import numpy as np
import os
from tqdm import tqdm

# 将原始数据转化为固定长度的brust序列，并将其分割为训练集、验证集、测试集
def process_and_save(
    data_path, 
    output_dir, 
    base_filename='processed', 
    max_length=512,
    label_mapping=False  # 新增：是否启用标签映射（继承原脚本功能）
):
    """
    数据处理并保存主函数
    参数:
        data_path: 原始数据路径
        output_dir: 输出目录（自动创建train/test/val子目录）
        base_filename: 输出文件的基础名称（自动添加_train/_val/_test后缀）
        max_length: 序列最大长度
        label_mapping: 是否将标签映射为连续整数索引（默认为False）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载原始数据
    data = np.load(data_path, allow_pickle=True)
    X_data = data['data'] if 'data' in data else data['X']
    y_data = data['labels'] if 'labels' in data else data['y']

    # 随机打乱
    indices = np.random.permutation(len(X_data))
    X_data, y_data = X_data[indices], y_data[indices]

    # 标签转换
    if label_mapping:
        unique_labels = sorted(set(y_data))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        y_data = np.array([label_to_idx[label] for label in y_data])
        print(f"标签映射完成，唯一标签数: {len(unique_labels)}")

    # 分割并保存
    for split_name, (start, end) in {
        'train': (0, 0.7),
        'val': (0.7, 0.8),
        'test': (0.8, 1)
    }.items():
        # 切片数据
        split_data = X_data[int(len(X_data)*start):int(len(X_data)*end)]
        split_labels = y_data[int(len(X_data)*start):int(len(X_data)*end)]

        # 转换突发序列
        bursts = []
        for x in tqdm(split_data, desc=f"Processing {split_name} set"):
            bursts.append(_convert_to_burst(x, max_length))

        # 保存文件（使用基础名称+后缀）
        output_path = os.path.join(
            output_dir, 
            f"{base_filename}_{split_name}.npz"  # 例如: cwx_train.npz
        )
        save_dict = {
            'burst_sequences': np.array(bursts),
            'labels': split_labels
        }
        if label_mapping:
            save_dict['label_mapping'] = np.array(list(label_to_idx.items()), dtype=object)
        
        np.savez(output_path, **save_dict)
        print(f"Saved {len(bursts)} samples to {output_path}")

def _convert_to_burst(sequence, max_length):
    """同之前的严格转换函数"""
    seq = np.array(sequence)
    seq = seq[seq != 0]
    if len(seq) == 0:
        return np.zeros(max_length)
    
    directions = np.sign(seq)
    bursts = []
    current_dir = directions[0]
    count = 1
    
    for dir_val in directions[1:]:
        if dir_val == current_dir:
            count += 1
        else:
            bursts.append(count * current_dir)
            current_dir = dir_val
            count = 1
    bursts.append(count * current_dir)
    
    bursts = np.array(bursts)
    if len(bursts) > max_length:
        return bursts[:max_length]
    return np.pad(bursts, (0, max_length - len(bursts)), 'constant')

if __name__ == "__main__":
    # 示例：将标签转换为二进制（0/1）
   

    process_and_save(
        data_path='/home/wyh/DTPN/CW100/tor_100w_2500tr.npz',
        output_dir='/home/wyh/DTPN/processed_data',
        base_filename='cw100',  # 指定基础名称
        max_length=512,
        label_mapping=True  # 启用标签映射
    )
