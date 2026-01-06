import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Позиционное кодирование для Transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerPredictor(nn.Module):
    """Transformer модель для прогноза сжатых представлений"""
    def __init__(self, input_size=256, d_model=512, nhead=8, num_layers=4, 
                 output_size=256, pred_len=30, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.output_size = output_size
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_size * pred_len)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        # Input projection and positional encoding
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        # Transformer encoder
        encoded = self.transformer_encoder(x)
        
        # Use last output for prediction
        last_output = encoded[:, -1, :]
        
        # Output projection
        output = self.output_projection(self.dropout(last_output))
        output = output.view(batch_size, self.pred_len, self.output_size)
        
        return output


class TransformerPredictorFactory:
    """Фабрика для создания Transformer моделей"""
    
    @staticmethod
    def create_default(input_size=256, pred_len=30):
        """Создание Transformer модели с параметрами по умолчанию"""
        return TransformerPredictor(
            input_size=input_size,
            d_model=512,
            nhead=8,
            num_layers=4,
            output_size=input_size,
            pred_len=pred_len,
            dropout=0.1
        )
    
    @staticmethod
    def create_custom(input_size, d_model, nhead, num_layers, output_size, pred_len, dropout):
        """Создание Transformer модели с кастомными параметрами"""
        return TransformerPredictor(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            output_size=output_size,
            pred_len=pred_len,
            dropout=dropout
        )