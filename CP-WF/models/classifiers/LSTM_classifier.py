import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        """
        LSTM分类器模型定义

        参数:
            input_size (int): 输入特征的维度
            hidden_size (int): LSTM隐藏层的维度
            num_layers (int): LSTM层的数量
            num_classes (int): 分类类别的数量
            dropout (float): Dropout比率
        """
        super(LSTMClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # 全连接层
        self.fc = nn.Linear(hidden_size, num_classes)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入数据, 形状为 [batch_size, sequence_length, input_size]

        返回:
            模型输出，形状为 [batch_size, num_classes]
        """
        # 对于单个时间步的数据，我们需要将其重塑为序列形式
        # 假设我们将512个特征视为一个长度为512的序列，每个时间步的特征维度为1
        batch_size = x.size(0)
        x = x.view(batch_size, 512, 1)

        # 初始化隐藏状态和细胞状态
        device = x.device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        # LSTM前向传播
        x.requires_grad_(True)  # 强制输入需要梯度
        out, _ = self.lstm(x, (h0, c0))

        # 只使用最后一个时间步的输出
        out = out[:, -1, :]

        # 应用dropout
        out = self.dropout(out)

        # 全连接层
        out = self.fc(out)
       
        return out