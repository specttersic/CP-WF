"""
数据加载模块

提供数据集类和数据加载器
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class CleanTrafficDataset(Dataset):
    """纯净流量数据集（用于预训练encoder）"""
    
    def __init__(self, dataset='A', data_path='../../processed_data', use_full=False):
        """
        Args:
            dataset: 数据集名称（A/B/C）
            data_path: 数据路径
            use_full: 是否使用完整数据集（True）或分类数据集（False）
        """
        self.dataset = dataset
        self.data_path = data_path
        self.use_full = use_full
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        if self.use_full:
            # 使用完整数据集（train + val + test）
            train_file = os.path.join(self.data_path, 'cw100_train.npz')
            val_file = os.path.join(self.data_path, 'cw100_val.npz')
            test_file = os.path.join(self.data_path, 'cw100_test.npz')
            
            # 加载并合并
            train_dict = np.load(train_file)
            val_dict = np.load(val_file)
            test_dict = np.load(test_file)
            
            self.data = np.concatenate([
                train_dict['burst_sequences'],
                val_dict['burst_sequences'],
                test_dict['burst_sequences']
            ], axis=0)
            
            self.labels = np.concatenate([
                train_dict['labels'],
                val_dict['labels'],
                test_dict['labels']
            ], axis=0)
            
            dataset_type = "完整数据集"
        else:
            # 使用分类数据集（splited_data中的A/B/C）
            train_file = os.path.join(self.data_path, 'splited_data', f'cw100_train_{self.dataset}.npz')
            val_file = os.path.join(self.data_path, 'splited_data', f'cw100_val_{self.dataset}.npz')
            test_file = os.path.join(self.data_path, 'splited_data', f'cw100_test_{self.dataset}.npz')
            
            # 加载并合并
            train_dict = np.load(train_file)
            val_dict = np.load(val_file)
            test_dict = np.load(test_file)
            
            self.data = np.concatenate([
                train_dict['burst_sequences'],
                val_dict['burst_sequences'],
                test_dict['burst_sequences']
            ], axis=0)
            
            self.labels = np.concatenate([
                train_dict['labels'],
                val_dict['labels'],
                test_dict['labels']
            ], axis=0)
            
            dataset_type = "分类数据集"
        
        print(f"CleanTrafficDataset loaded:")
        print(f"  Type: {dataset_type}")
        print(f"  Dataset: {self.dataset}")
        print(f"  Data shape: {self.data.shape}")
        print(f"  Labels shape: {self.labels.shape}")
        print(f"  Num classes: {len(np.unique(self.labels))}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            data: (1, 1, 512) 纯净流量
            label: 类别标签
        """
        data = torch.FloatTensor(self.data[idx]).unsqueeze(0).unsqueeze(0)  # (1, 1, 512)
        label = torch.LongTensor([self.labels[idx]])[0]
        
        return data, label


class AdversarialTrafficDataset(Dataset):
    """对抗流量数据集（用于训练denoiser）"""
    
    def __init__(self, datasets=None, defense='WT', data_path='../../processed_data', use_full=False):
        """
        Args:
            datasets: 数据集名称列表（['A'], ['A', 'B'], ['A', 'B', 'C']等）或None（使用完整数据集）
            defense: 防御方法（WT/Adv）
            data_path: 数据路径
            use_full: 是否使用完整数据集（不划分A/B/C）
        """
        self.datasets = datasets
        self.defense = defense
        self.data_path = data_path
        self.use_full = use_full
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        if self.use_full:
            # 使用完整数据集（不划分A/B/C）
            if self.defense == 'WT':
                adv_train = os.path.join(self.data_path, 'cw100_WT_train.npz')
                adv_val = os.path.join(self.data_path, 'cw100_WT_val.npz')
                adv_test = os.path.join(self.data_path, 'cw100_WT_test.npz')
            elif self.defense == 'Adv' or self.defense == 'adv':
                adv_train = os.path.join(self.data_path, 'cw100_adv_train.npz')
                adv_val = os.path.join(self.data_path, 'cw100_adv_val.npz')
                adv_test = os.path.join(self.data_path, 'cw100_adv_test.npz')
            elif self.defense == 'mockingbird' or self.defense == 'Mockingbird':
                adv_train = os.path.join(self.data_path, 'cw100_mockingbird_train.npz')
                adv_val = os.path.join(self.data_path, 'cw100_mockingbird_val.npz')
                adv_test = os.path.join(self.data_path, 'cw100_mockingbird_test.npz')
            else:
                raise ValueError(f"Unknown defense: {self.defense}")
            
            # 加载并合并对抗流量
            train_dict = np.load(adv_train)
            val_dict = np.load(adv_val)
            test_dict = np.load(adv_test)
            
            self.adv_data = np.concatenate([
                train_dict['burst_sequences'],
                val_dict['burst_sequences'],
                test_dict['burst_sequences']
            ], axis=0)
            
            # 加载原始流量和标签
            clean_train = os.path.join(self.data_path, 'cw100_train.npz')
            clean_val = os.path.join(self.data_path, 'cw100_val.npz')
            clean_test = os.path.join(self.data_path, 'cw100_test.npz')
            
            train_dict = np.load(clean_train)
            val_dict = np.load(clean_val)
            test_dict = np.load(clean_test)
            
            self.clean_data = np.concatenate([
                train_dict['burst_sequences'],
                val_dict['burst_sequences'],
                test_dict['burst_sequences']
            ], axis=0)
            
            self.labels = np.concatenate([
                train_dict['labels'],
                val_dict['labels'],
                test_dict['labels']
            ], axis=0)
            
            dataset_type = "完整数据集"
            
        else:
            # 使用分类数据集（单个或多个A/B/C的组合）
            if not self.datasets:
                raise ValueError("datasets参数不能为空（当use_full=False时）")
            
            adv_data_list = []
            clean_data_list = []
            labels_list = []
            
            for dataset in self.datasets:
                # 对抗流量路径
                if self.defense == 'WT':
                    adv_train = os.path.join(self.data_path, 'splited_data', f'cw100_WT_train_{dataset}.npz')
                    adv_val = os.path.join(self.data_path, 'splited_data', f'cw100_WT_val_{dataset}.npz')
                    adv_test = os.path.join(self.data_path, 'splited_data', f'cw100_WT_test_{dataset}.npz')
                elif self.defense == 'Adv' or self.defense == 'adv':
                    adv_train = os.path.join(self.data_path, 'splited_data', f'cw100_Adv_train_{dataset}.npz')
                    adv_val = os.path.join(self.data_path, 'splited_data', f'cw100_Adv_val_{dataset}.npz')
                    adv_test = os.path.join(self.data_path, 'splited_data', f'cw100_Adv_test_{dataset}.npz')
                elif self.defense == 'mockingbird' or self.defense == 'Mockingbird':
                    adv_train = os.path.join(self.data_path, 'splited_data', f'cw100_mockingbird_train_{dataset}.npz')
                    adv_val = os.path.join(self.data_path, 'splited_data', f'cw100_mockingbird_val_{dataset}.npz')
                    adv_test = os.path.join(self.data_path, 'splited_data', f'cw100_mockingbird_test_{dataset}.npz')
                else:
                    raise ValueError(f"Unknown defense: {self.defense}")
                
                # 加载对抗流量
                train_dict = np.load(adv_train)
                val_dict = np.load(adv_val)
                test_dict = np.load(adv_test)
                
                adv_data_list.append(train_dict['burst_sequences'])
                adv_data_list.append(val_dict['burst_sequences'])
                adv_data_list.append(test_dict['burst_sequences'])
                
                # 加载原始流量和标签
                clean_train = os.path.join(self.data_path, 'splited_data', f'cw100_train_{dataset}.npz')
                clean_val = os.path.join(self.data_path, 'splited_data', f'cw100_val_{dataset}.npz')
                clean_test = os.path.join(self.data_path, 'splited_data', f'cw100_test_{dataset}.npz')
                
                train_dict = np.load(clean_train)
                val_dict = np.load(clean_val)
                test_dict = np.load(clean_test)
                
                clean_data_list.append(train_dict['burst_sequences'])
                clean_data_list.append(val_dict['burst_sequences'])
                clean_data_list.append(test_dict['burst_sequences'])
                
                labels_list.append(train_dict['labels'])
                labels_list.append(val_dict['labels'])
                labels_list.append(test_dict['labels'])
            
            # 合并所有数据
            self.adv_data = np.concatenate(adv_data_list, axis=0)
            self.clean_data = np.concatenate(clean_data_list, axis=0)
            self.labels = np.concatenate(labels_list, axis=0)
            
            if len(self.datasets) == 1:
                dataset_type = f"分类数据集 {self.datasets[0]}"
            else:
                dataset_type = f"分类数据集组合 {'+'.join(self.datasets)}"
        
        # 调整形状
        if len(self.adv_data.shape) == 2:  # (N, 512)
            self.adv_data = self.adv_data.reshape(-1, 1, 1, 512)
        if len(self.clean_data.shape) == 2:  # (N, 512)
            self.clean_data = self.clean_data.reshape(-1, 1, 1, 512)
        
        print(f"AdversarialTrafficDataset loaded:")
        print(f"  Type: {dataset_type}")
        print(f"  Defense: {self.defense}")
        print(f"  Adversarial shape: {self.adv_data.shape}")
        print(f"  Clean shape: {self.clean_data.shape}")
        print(f"  Labels shape: {self.labels.shape}")
    
    def __len__(self):
        return len(self.adv_data)
    
    def __getitem__(self, idx):
        """
        Returns:
            adv_data: (1, 1, 512) 对抗流量
            clean_data: (1, 1, 512) 原始流量
            label: 类别标签
        """
        adv = torch.FloatTensor(self.adv_data[idx])  # (1, 1, 512)
        clean = torch.FloatTensor(self.clean_data[idx])  # (1, 1, 512)
        label = torch.LongTensor([self.labels[idx]])[0]
        
        return adv, clean, label


def get_clean_dataloader(dataset='A', batch_size=256, shuffle=True, 
                         num_workers=4, data_path='../../processed_data', use_full=False):
    """
    获取纯净流量数据加载器
    
    Args:
        dataset: 数据集名称（A/B/C）
        batch_size: 批大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        data_path: 数据路径
        use_full: 是否使用完整数据集
        
    Returns:
        dataloader: 数据加载器
    """
    dataset_obj = CleanTrafficDataset(dataset=dataset, data_path=data_path, use_full=use_full)
    
    dataloader = DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def get_adversarial_dataloader(datasets=None, defense='WT', batch_size=32, 
                                shuffle=True, num_workers=4, 
                                data_path='../../processed_data', use_full=False):
    """
    获取对抗流量数据加载器
    
    Args:
        datasets: 数据集名称列表（['A'], ['A', 'B'], ['A', 'B', 'C']等）或None（使用完整数据集）
        defense: 防御方法（WT/Adv）
        batch_size: 批大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        data_path: 数据路径
        use_full: 是否使用完整数据集
        
    Returns:
        dataloader: 数据加载器
    """
    dataset_obj = AdversarialTrafficDataset(
        datasets=datasets,
        defense=defense,
        data_path=data_path,
        use_full=use_full
    )
    
    dataloader = DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


# ========== 测试代码 ==========
if __name__ == '__main__':
    print("测试数据加载器")
    print("=" * 80)
    
    # 测试纯净流量数据集
    print("\n测试CleanTrafficDataset:")
    print("-" * 80)
    clean_loader = get_clean_dataloader(dataset='A', batch_size=256)
    
    for data, labels in clean_loader:
        print(f"Data shape: {data.shape}")
        print(f"Labels shape: {labels.shape}")
        break
    
    # 测试对抗流量数据集
    print("\n测试AdversarialTrafficDataset:")
    print("-" * 80)
    adv_loader = get_adversarial_dataloader(dataset='A', defense='WT', batch_size=32)
    
    for adv_data, clean_data, labels in adv_loader:
        print(f"Adversarial data shape: {adv_data.shape}")
        print(f"Clean data shape: {clean_data.shape}")
        print(f"Labels shape: {labels.shape}")
        break
    
    print("\n✓ All tests passed!")
