import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(d_model, 1)
        # 若需要 Sigmoid，可在此解注释
        # self.sigmoid = nn.Sigmoid()

    def forward(self, src):
        # src: [batch_size, seq_len, input_dim]
        src = self.input_linear(src) 
        src = src.transpose(0, 1)       # [seq_len, batch_size, d_model]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.transpose(0, 1)  # [batch_size, seq_len, d_model]
        output = self.decoder(output)    # [batch_size, seq_len, 1]
        # output = self.sigmoid(output)
        return output

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        
        # batch_first=True => 输入输出均为 [batch_size, seq_len, feature_dim]
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout,
            bidirectional=False  # 如果想用双向LSTM，可改为True
        )
        
        # 把 LSTM 输出的 hidden 映射到 1 维
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        lstm_out, (h_n, c_n) = self.lstm(x)  
        # lstm_out shape: [batch_size, seq_len, hidden_dim]
        
        # 对时间序列上每一步做线性映射 => [batch_size, seq_len, 1]
        out = self.fc(lstm_out)
        return out  # logits