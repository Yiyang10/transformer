import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

def calc_metrics(labels, preds, threshold=0.5):
    """
    labels: numpy 数组，形状 [N, seq_len]，真实标签（0/1）
    preds:  numpy 数组，形状 [N, seq_len]，模型输出的 logits
    """
    labels_flat = labels.flatten()
    preds_sigmoid = 1 / (1 + np.exp(-preds))  # sigmoid 转换
    preds_binary = (preds_sigmoid >= threshold).astype(np.float32)
    preds_flat = preds_binary.flatten()
    
    precision = precision_score(labels_flat, preds_flat, zero_division=0)
    recall = recall_score(labels_flat, preds_flat, zero_division=0)
    f1 = f1_score(labels_flat, preds_flat, zero_division=0)
    return precision, recall, f1

def evaluate_model(model, test_loader, device, threshold=0.5, save_csv_path=None):
    model.eval()
    from collections import defaultdict

    # 存储所有预测：predictions_dict[(seg_idx, frame_offset)] = list_of_pred
    predictions_dict = defaultdict(list)
    # 存储每帧的真实标签：labels_dict[(seg_idx, frame_offset)] = label_value
    labels_dict = {}
    with torch.no_grad():
        for val_samples, val_labels, seg_idx_batch, offset_batch in test_loader:
            val_samples = val_samples.to(device)
            outputs = model(val_samples)  # -> [batch_size, window_size, 1]
            outputs = outputs.squeeze(-1).cpu().numpy()   # [batch_size, window_size]
            val_labels = val_labels.numpy()               # [batch_size, window_size]
            seg_idx_batch = seg_idx_batch.numpy()         # [batch_size]
            offset_batch = offset_batch.numpy()           # [batch_size]

            batch_size, seq_len = outputs.shape

            # 把每个窗口的预测分配到对应 (seg_idx, frame_offset)
            for b in range(batch_size):
                seg_idx_val = seg_idx_batch[b]
                window_start = offset_batch[b]
                for frame_i in range(seq_len):
                    frame_offset = window_start + frame_i
                    key = (seg_idx_val, frame_offset)

                    # 记录预测
                    predictions_dict[key].append(outputs[b, frame_i])
                    # 记录标签
                    labels_dict[key] = val_labels[b, frame_i]

    # 计算每帧的平均预测
    final_keys = []
    final_preds = []
    final_labels = []

    # 为了让结果可控，可按 key 排序后再组装
    for key in sorted(predictions_dict.keys()):
        avg_pred = np.mean(predictions_dict[key])  # 对同一帧的多个预测求均值
        label_val = labels_dict[key]

        final_keys.append(key)
        final_preds.append(avg_pred)
        final_labels.append(label_val)

    final_preds = np.array(final_preds)    # [num_frames]
    final_labels = np.array(final_labels)  # [num_frames]

    # 转成概率 & 二值
    preds_sigmoid = 1 / (1 + np.exp(-final_preds))
    preds_binary = (preds_sigmoid >= threshold).astype(np.float32)

    # ======== 计算评估指标 ========
    precision = precision_score(final_labels, preds_binary, zero_division=0)
    recall = recall_score(final_labels, preds_binary, zero_division=0)
    f1 = f1_score(final_labels, preds_binary, zero_division=0)
    print(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")

    # ======== 若需要保存 CSV ========
    if save_csv_path:
        import pandas as pd
        seg_idxs = [k[0] for k in final_keys]
        frame_offsets = [k[1] for k in final_keys]

        df_result = pd.DataFrame({
            "seg_idx": seg_idxs,
            "frame_offset": frame_offsets,
            "label": final_labels,
            "logits": final_preds,
            "prob": preds_sigmoid,
            "prediction": preds_binary
        })
        df_result.to_csv(save_csv_path, index=False)
        print(f"评估结果（含平均后预测）已保存到: {save_csv_path}")
