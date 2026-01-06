#!/usr/bin/env python
"""Тестирование всех моделей"""

import torch
import torch.nn as nn
from model_factory import OceanForecastingModelFactory

def test_all_models():
    """Тестирование всех доступных моделей"""
    print("=" * 80)
    print("Тестирование моделей прогноза океана")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    
    # Получаем список доступных моделей
    available_models = OceanForecastingModelFactory.get_available_models()
    print(f"\nДоступно моделей: {len(available_models)}")
    
    # ========== Этап 1: Контрастное обучение ==========
    print("\n" + "=" * 80)
    print("ЭТАП 1: Контрастное обучение")
    print("=" * 80)
    
    # MoCo ResNet50
    print("\n1. MoCo ResNet50:")
    try:
        # Используем адаптированную версию
        moco_resnet = OceanForecastingModelFactory.create_moco_resnet50_adapted()
        moco_resnet = moco_resnet.to(device)
        OceanForecastingModelFactory.print_model_summary(moco_resnet, (7, 349, 661))
        
        # Тест прямого прохода
        dummy_input1 = torch.randn(2, 7, 349, 661).to(device)
        dummy_input2 = torch.randn(2, 7, 349, 661).to(device)
        loss = moco_resnet(dummy_input1, dummy_input2, m=0.99)
        print(f"  ✓ Прямой проход работает, loss: {loss.item():.6f}")
        
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    
    # MoCo ViT-Small
    print("\n2. MoCo ViT-Small:")
    try:
        moco_vit = OceanForecastingModelFactory.create_moco_vit_small()
        moco_vit = moco_vit.to(device)
        OceanForecastingModelFactory.print_model_summary(moco_vit, (7, 661, 661))
    except Exception as e:
        print(f"  ✗ Ошибка создания модели: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== Этап 2: Прогнозирование ==========
    print("\n" + "=" * 80)
    print("ЭТАП 2: Прогнозирование сжатых представлений")
    print("=" * 80)
    
    # LSTM Predictor
    print("\n1. LSTM Predictor:")
    lstm = OceanForecastingModelFactory.create_lstm_predictor()
    lstm = lstm.to(device)
    OceanForecastingModelFactory.print_model_summary(lstm, (10, 256))  # seq_len=10, input_size=256
    
    # Transformer Predictor
    print("\n2. Transformer Predictor:")
    transformer = OceanForecastingModelFactory.create_transformer_predictor()
    transformer = transformer.to(device)
    OceanForecastingModelFactory.print_model_summary(transformer, (10, 256))
    
    # ========== Этап 3: Декодирование ==========
    print("\n" + "=" * 80)
    print("ЭТАП 3: Декодирование")
    print("=" * 80)
    
    # ResNet Decoder (требуется энкодер)
    print("\n1. ResNet Decoder (с фиктивным энкодером):")
    
    # Создаем фиктивный энкодер
    class DummyEncoder(nn.Module):
        def __init__(self, output_dim=256):
            super().__init__()
            self.output_dim = output_dim
            self.conv = nn.Conv2d(7, 64, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, output_dim)
            
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    dummy_encoder = DummyEncoder().to(device)
    resnet_decoder = OceanForecastingModelFactory.create_resnet_decoder(dummy_encoder)
    resnet_decoder = resnet_decoder.to(device)
    OceanForecastingModelFactory.print_model_summary(resnet_decoder, (7, 349, 661))
    
    # ViT Decoder (требуется энкодер)
    print("\n2. ViT Decoder (с фиктивным энкодером):")
    
    class DummyViTEncoder(nn.Module):
        def __init__(self, output_dim=256):
            super().__init__()
            self.output_dim = output_dim
            self.fc = nn.Linear(7 * 661 * 661, output_dim)
            
        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    dummy_vit_encoder = DummyViTEncoder().to(device)
    vit_decoder = OceanForecastingModelFactory.create_vit_decoder(dummy_vit_encoder)
    vit_decoder = vit_decoder.to(device)
    OceanForecastingModelFactory.print_model_summary(vit_decoder, (7, 349, 661))
    
    # ========== Тестирование прямого прохода ==========
    print("\n" + "=" * 80)
    print("Тестирование прямого прохода")
    print("=" * 80)
    
    try:
        # Тест MoCo ResNet
        print("\nТест MoCo ResNet50...")
        dummy_input1 = torch.randn(2, 7, 349, 661).to(device)
        dummy_input2 = torch.randn(2, 7, 349, 661).to(device)
        loss = moco_resnet(dummy_input1, dummy_input2, m=0.99)
        print(f"  Loss: {loss.item():.6f}")
        print("  ✓ Прямой проход работает")
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        
    try:
        # Тест MoCo ViT
        print("\nТест MoCo ViT-Small...")
        dummy_input1 = torch.randn(2, 7, 661, 661).to(device)
        dummy_input2 = torch.randn(2, 7, 661, 661).to(device)
        loss = moco_vit(dummy_input1, dummy_input2, m=0.99)
        print(f"  Loss: {loss.item():.6f}")
        print("  ✓ Прямой проход работает")
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
    
    try:
        # Тест LSTM
        print("\nТест LSTM Predictor...")
        dummy_input = torch.randn(4, 10, 256).to(device)  # batch=4, seq_len=10, features=256
        output = lstm(dummy_input)
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print("  ✓ Прямой проход работает")
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        
        
    try:
        # Тест Transformer
        print("\nТест Transformer Predictor...")
        dummy_input = torch.randn(4, 10, 256).to(device)
        output = transformer(dummy_input)
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print("  ✓ Прямой проход работает")
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
    
    try:
        # Тест ResNet Decoder
        print("\nТест ResNet Decoder...")
        dummy_input = torch.randn(2, 7, 349, 661).to(device)
        output = resnet_decoder(dummy_input)
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print("  ✓ Прямой проход работает")
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
    
    try:
        # Тест ViT Decoder
        print("\nТест ViT Decoder...")
        dummy_input = torch.randn(2, 7, 349, 661).to(device)
        output = vit_decoder(dummy_input)
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print("  ✓ Прямой проход работает")
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
    
    
    print("\n" + "=" * 80)
    print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ!")
    print("=" * 80)

if __name__ == "__main__":
    test_all_models()