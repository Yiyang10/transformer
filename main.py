import torch
import torch.nn as nn
from dataset import SlidingWindowDataset
from model import TransformerModel  
from model import LSTMModel
from train import train_model
import pickle
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np


# def get_data_loaders_by_segment(
#     pickle_path, 
#     sequence_length=32, 
#     batch_size=32, 
#     train_ratio=0.8, 
#     shuffle=True, 
#     random_seed=42
# ):
#     with open(pickle_path, "rb") as f:
#         big_df = pickle.load(f)
#     if not isinstance(big_df, pd.DataFrame):
#         raise ValueError("The loaded pickle does not contain a DataFrame.")

#     segments = big_df["segment_name"].unique()
#     np.random.seed(random_seed)
#     np.random.shuffle(segments)
#     train_count = int(len(segments) * train_ratio)
#     train_segments = segments[:train_count]
#     test_segments = segments[train_count:]

#     train_df = big_df[big_df["segment_name"].isin(train_segments)].reset_index(drop=True)
#     test_df = big_df[big_df["segment_name"].isin(test_segments)].reset_index(drop=True)

#     train_dataset = SlidingWindowDataset(train_df, window_size=sequence_length)
#     test_dataset = SlidingWindowDataset(test_df, window_size=sequence_length)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, test_loader
def get_data_loaders_by_segment(
    train_pickle_path,
    test_pickle_path,
    sequence_length=32, 
    batch_size=32, 
    shuffle=True
):
    # 加载训练数据
    with open(train_pickle_path, "rb") as f:
        train_df = pickle.load(f)
    train_dataset = SlidingWindowDataset(train_df, window_size=sequence_length, return_indices=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # 加载测试数据
    with open(test_pickle_path, "rb") as f:
        test_df = pickle.load(f)
    test_dataset = SlidingWindowDataset(test_df, window_size=sequence_length, return_indices=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def main():
    # ========== 超参数设置 ==========
    #pickle_file = "/Users/john/Desktop/data/pickle/combined.pickle"  # 你的大表 pickle 文件
    train_pickle_file = "/Users/john/Desktop/data/pickle/combined.pickle"
    test_pickle_file = "/Users/john/Desktop/data/pickle/3GC_test.pickle"
    sequence_length = 32
    batch_size = 32

    # 假设 big_df 的特征列有 35 个 (最后两列是 attack & segment_name)
    input_dim = 35         
    hidden_dim = 64
    num_layers = 2
    dropout = 0.1

    # Transformer 相关
    d_model = 64
    nhead = 4
    dim_feedforward = 128

    learning_rate = 1e-3
    num_epochs = 12

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========== 数据加载 ========== 
    # train_loader, test_loader = get_data_loaders_by_segment(
    #     pickle_path=pickle_file,
    #     sequence_length=sequence_length,
    #     batch_size=batch_size
    # )
    train_loader, test_loader = get_data_loaders_by_segment(
    train_pickle_path=train_pickle_file,
    test_pickle_path=test_pickle_file,
    sequence_length=sequence_length,
    batch_size=batch_size
    )

    # ========== 选择并初始化模型 ========== 
    # 例：使用 LSTMModel (batch_size, seq_len, input_dim) -> (batch_size, seq_len, 1)
    model = LSTMModel(input_dim, hidden_dim, num_layers, dropout).to(device)

    # 或者：使用 TransformerModel
    # model = TransformerModel(input_dim, d_model, nhead, num_layers, dim_feedforward, dropout).to(device)

    # ========== 定义损失函数和优化器 ========== 
    # 由于逐帧二分类 (0/1)，并且模型输出是 logits(未过 sigmoid)，可用 BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ========== 开始训练 ========== 
    train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs)


if __name__ == "__main__":
    main()