import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    """LSTM модель для прогноза сжатых представлений"""
    def __init__(self, input_size=256, hidden_size=512, num_layers=2, 
                 output_size=256, pred_len=30, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.output_size = output_size
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size * pred_len)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Берем последний выход
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Полносвязный слой
        output = self.fc(self.dropout(last_output))
        output = output.view(batch_size, self.pred_len, self.output_size)
        
        return output


class LSTMPredictorFactory:
    """Фабрика для создания LSTM моделей"""
    
    @staticmethod
    def create_default(input_size=256, pred_len=30):
        """Создание LSTM модели с параметрами по умолчанию"""
        return LSTMPredictor(
            input_size=input_size,
            hidden_size=512,
            num_layers=2,
            output_size=input_size,
            pred_len=pred_len,
            dropout=0.2
        )
    
    @staticmethod
    def create_custom(input_size, hidden_size, num_layers, output_size, pred_len, dropout):
        """Создание LSTM модели с кастомными параметрами"""
        return LSTMPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            pred_len=pred_len,
            dropout=dropout
        )