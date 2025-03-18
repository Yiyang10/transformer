import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import random
import bisect
class SlidingWindowDataset(Dataset):
    """
    懒加载版本：不预先把每个滑窗的 (x, y) 存起来，而是在 __getitem__ 时现用现取。
    同时返回 (seg_idx, window_start) 用于后续对同一帧合并预测。
    """

    def __init__(self, df, window_size=32, return_indices=False):
        self.window_size = window_size
        self.return_indices = return_indices
        # 列名
        all_cols = df.columns.tolist()
        self.feature_cols = all_cols[:-2]  # 前 (M-2) 列
        self.label_col = all_cols[-2]      # 倒数第二列 -> 'attack'
        self.segment_col = all_cols[-1]    # 最后一列 -> 'segment_name'

        # 按 segment 分组
        groups = df.groupby(self.segment_col, sort=False)

        self.segments = []         # 存 (feat_array, label_array, seg_idx)
        self.cumulative_counts = [0]  # 用于前缀和定位

        for seg_idx, (seg_name, seg_df) in enumerate(groups):
            seg_df = seg_df.reset_index(drop=True)
            length = len(seg_df)
            if length < window_size:
                continue

            # 转成 numpy，后面切片更快
            feat_array = seg_df[self.feature_cols].values  # shape: [length, num_features]
            label_array = seg_df[self.label_col].values    # shape: [length]

            num_windows = length - window_size + 1

            # 记录该 segment
            self.segments.append((feat_array, label_array, seg_idx))
            self.cumulative_counts.append(self.cumulative_counts[-1] + num_windows)

    def __len__(self):
        # 全部 segment 的窗口数之和
        return self.cumulative_counts[-1]

    def __getitem__(self, idx):
        """
        idx: 全局第 idx 个滑窗 (0-based)
        返回:
            x_tensor: [window_size, num_features]
            y_tensor: [window_size]
            seg_idx_val: 整型，表示这是第几个 segment
            offset_within_seg: 在该 segment 内的滑窗起始位置
        """
        # 1) 确定该 idx 属于哪个 segment
        seg_idx = bisect.bisect_left(self.cumulative_counts, idx+1) - 1
        feat_array, label_array, seg_idx_val = self.segments[seg_idx]

        # 2) 计算在 segment 内的偏移
        offset_within_seg = idx - self.cumulative_counts[seg_idx]

        # 3) 切片
        start = offset_within_seg
        end = offset_within_seg + self.window_size
        x = feat_array[start:end]  # shape: [window_size, num_features]
        y = label_array[start:end] # shape: [window_size]

        # 4) 转成 torch.tensor
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)  

        if self.return_indices:
            return x_tensor, y_tensor, seg_idx_val, offset_within_seg
        else:
            return x_tensor, y_tensor

# class SlidingWindowDataset(Dataset):
#     """
#     针对一个大表（含多个 segment），以窗口方式遍历数据：
#       - 每个窗口大小为 window_size
#       - 不跨 segment 拼接
#       - 每个滑窗返回 (x, y)，其中：
#           x.shape = [window_size, num_features]
#           y.shape = [window_size] (逐帧标签)
#     """

#     def __init__(self, df, window_size=32):
#         """
#         参数:
#         -------
#         df : pd.DataFrame
#             最后一列必须是 'segment_name'，倒数第二列是 'attack' (0/1 标签)。
#             其余列全是特征。
#         window_size : int
#             滑窗长度
#         """
#         self.window_size = window_size

#         # 1) 准备特征列与标签列的名称
#         all_cols = df.columns.tolist()
#         self.feature_cols = all_cols[:-2]  # 前 (M-2) 列
#         self.label_col = all_cols[-2]      # 倒数第二列 -> "attack"
#         self.segment_col = all_cols[-1]    # 最后一列 -> "segment_name"

#         # 2) 依照 segment_name 分组
#         groups = df.groupby(self.segment_col, sort=False)

#         # 用来存所有 (x, y) 样本
#         self.samples = []

#         # 3) 在每个 segment 内做滑窗
#         for seg_name, seg_df in groups:
#             seg_df = seg_df.reset_index(drop=True)
#             length = len(seg_df)

#             if length < window_size:
#                 # 如果该段不足一个窗口长度，直接跳过
#                 continue

#             # 逐帧滑动: [i, i+1, ..., i+window_size-1]
#             for i in range(length - window_size + 1):
#                 window_data = seg_df.iloc[i : i + window_size]

#                 # 特征 [window_size, num_features]
#                 x = window_data[self.feature_cols].values
#                 # 标签 [window_size]
#                 y = window_data[self.label_col].values

#                 self.samples.append((x, y))

    # def __len__(self):
    #     # 返回一个固定的样本数，可以根据实际情况调整
    #     return self.num_samples

    # def __getitem__(self, idx):
    #     # 随机选择一个 segment
    #     seg_df = random.choice(self.segments)
    #     # 计算该 segment 内可以滑动的起始位置范围
    #     max_start = len(seg_df) - self.window_size
    #     # 随机选取一个合法的起始位置
    #     start_idx = random.randint(0, max_start)
    #     window_data = seg_df.iloc[start_idx : start_idx + self.window_size]

    #     # 提取特征与标签
    #     x = window_data[self.feature_cols].values  # shape: [window_size, num_features]
    #     y = window_data[self.label_col].values       # shape: [window_size]

    #     # 转成 tensor（注意标签需要 float 类型以适配 BCEWithLogitsLoss）
    #     x_tensor = torch.tensor(x, dtype=torch.float32)
    #     y_tensor = torch.tensor(y, dtype=torch.float32)
    #     return x_tensor, y_tensor