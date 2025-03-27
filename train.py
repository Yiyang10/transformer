import torch
import numpy as np
from test import evaluate_model  # 调用 test.py 中的评估函数
import wandb

def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=12, threshold=0.5):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_samples, batch_labels in train_loader:
            batch_samples = batch_samples.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_samples)  # [batch_size, seq_len, 1]
            outputs = outputs.squeeze(-1)   # [batch_size, seq_len]
            
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        wandb.log({"Train Loss": avg_loss, "epoch": epoch + 1})
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}")
        
        # 每个 epoch 结束后评估测试集
        evaluate_model(model, test_loader, device, threshold, save_csv_path="transformer_outputs.csv", epoch=epoch)
