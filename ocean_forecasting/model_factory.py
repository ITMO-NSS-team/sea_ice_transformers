import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from models.contrastive.moco_resnet import MoCoResNetFactory, MoCo_ResNet
from models.contrastive.moco_vit import MoCoViTFactory, MoCo_ViT
from models.forecasting.lstm_predictor import LSTMPredictorFactory, LSTMPredictor
from models.forecasting.transformer_predictor import TransformerPredictorFactory, TransformerPredictor
from models.decoding.resnet_decoder import ResNetDecoderFactory, FrozenResNetEncoderDecoder
from models.decoding.vit_decoder import ViTDecoderFactory, FrozenViTEncoderDecoder


class OceanForecastingModelFactory:
    """Фабрика для создания всех моделей прогноза океана"""
    
    # Реестр всех моделей
    MODEL_REGISTRY = {
        # Этап 1: Контрастное обучение
        'moco_resnet50': 'create_moco_resnet50',
        'moco_resnet101': 'create_moco_resnet101',
        'moco_vit_small': 'create_moco_vit_small',
        'moco_vit_base': 'create_moco_vit_base',
        
        # Этап 2: Прогнозирование
        'lstm_predictor': 'create_lstm_predictor',
        'transformer_predictor': 'create_transformer_predictor',
        
        # Этап 3: Декодирование
        'resnet_decoder': 'create_resnet_decoder',
        'vit_decoder': 'create_vit_decoder',
    }
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> nn.Module:
        """
        Создание модели по типу
        
        Args:
            model_type: тип модели из реестра
            **kwargs: параметры для создания модели
            
        Returns:
            Инициализированная модель PyTorch
        """
        if model_type not in cls.MODEL_REGISTRY:
            available = list(cls.MODEL_REGISTRY.keys())
            raise ValueError(f"Model type '{model_type}' not found. Available: {available}")
        
        creator_method = getattr(cls, cls.MODEL_REGISTRY[model_type])
        return creator_method(**kwargs)
    
    # ========== Этап 1: Контрастное обучение ==========
    
    @staticmethod
    def create_moco_resnet50(
        input_channels: int = 7,
        dim: int = 256,
        mlp_dim: int = 4096,
        T: float = 1.0
    ) -> MoCo_ResNet:
        """Создание MoCo ResNet50"""
        return MoCoResNetFactory.create_resnet50(
            input_channels=input_channels,
            dim=dim,
            mlp_dim=mlp_dim,
            T=T
        )
    
    @staticmethod
    def create_moco_resnet101(
        input_channels: int = 7,
        dim: int = 256,
        mlp_dim: int = 4096,
        T: float = 1.0
    ) -> MoCo_ResNet:
        """Создание MoCo ResNet101"""
        return MoCoResNetFactory.create_resnet101(
            input_channels=input_channels,
            dim=dim,
            mlp_dim=mlp_dim,
            T=T
        )
    
    @staticmethod
    def create_moco_vit_small(
        img_size: int = 661,
        in_chans: int = 7,
        dim: int = 256,
        mlp_dim: int = 4096,
        T: float = 1.0
    ) -> MoCo_ViT:
        """Создание MoCo ViT-Small"""
        return MoCoViTFactory.create_vit_small(
            img_size=img_size,
            in_chans=in_chans,
            dim=dim,
            mlp_dim=mlp_dim,
            T=T
        )
    
    @staticmethod
    def create_moco_vit_base(
        img_size: int = 661,
        in_chans: int = 7,
        dim: int = 256,
        mlp_dim: int = 4096,
        T: float = 1.0
    ) -> MoCo_ViT:
        """Создание MoCo ViT-Base"""
        return MoCoViTFactory.create_vit_base(
            img_size=img_size,
            in_chans=in_chans,
            dim=dim,
            mlp_dim=mlp_dim,
            T=T
        )
    
    # ========== Этап 2: Прогнозирование ==========
    
    @staticmethod
    def create_lstm_predictor(
        input_size: int = 256,
        pred_len: int = 30,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2
    ) -> LSTMPredictor:
        """Создание LSTM модели для прогноза"""
        return LSTMPredictorFactory.create_custom(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=input_size,
            pred_len=pred_len,
            dropout=dropout
        )
    
    @staticmethod
    def create_transformer_predictor(
        input_size: int = 256,
        pred_len: int = 30,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ) -> TransformerPredictor:
        """Создание Transformer модели для прогноза"""
        return TransformerPredictorFactory.create_custom(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            output_size=input_size,
            pred_len=pred_len,
            dropout=dropout
        )
    
    # ========== Этап 3: Декодирование ==========
    
    @staticmethod
    def create_resnet_decoder(
        encoder: nn.Module,
        in_channels: int = 7,
        height: int = 349,
        width: int = 661,
        latent_dim: int = 256,
        base_channels: int = 256
    ) -> FrozenResNetEncoderDecoder:
        """Создание декодера для ResNet"""
        return ResNetDecoderFactory.create_default(
            encoder=encoder,
            in_channels=in_channels,
            height=height,
            width=width
        )
    
    @staticmethod
    def create_vit_decoder(
        encoder: nn.Module,
        in_channels: int = 7,
        height: int = 349,
        width: int = 661,
        encoder_img_size: int = 661,
        latent_dim: int = 256,
        base_channels: int = 256
    ) -> FrozenViTEncoderDecoder:
        """Создание декодера для ViT"""
        return ViTDecoderFactory.create_default(
            encoder=encoder,
            in_channels=in_channels,
            height=height,
            width=width,
            encoder_img_size=encoder_img_size
        )
    
    # ========== Утилиты ==========
    
    @staticmethod
    def get_available_models() -> Dict[str, str]:
        """Получение списка доступных моделей с описанием"""
        return {
            'moco_resnet50': 'MoCo с ResNet50 для контрастного обучения',
            'moco_resnet101': 'MoCo с ResNet101 для контрастного обучения',
            'moco_vit_small': 'MoCo с ViT-Small для контрастного обучения',
            'moco_vit_base': 'MoCo с ViT-Base для контрастного обучения',
            'lstm_predictor': 'LSTM для прогноза сжатых представлений',
            'transformer_predictor': 'Transformer для прогноза сжатых представлений',
            'resnet_decoder': 'Декодер для восстановления из ResNet представлений',
            'vit_decoder': 'Декодер для восстановления из ViT представлений',
        }
    
    @staticmethod
    def print_model_summary(model: nn.Module, input_shape: tuple = None):
        """Печать информации о модели"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model: {model.__class__.__name__}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        
        if input_shape:
            try:
                device = next(model.parameters()).device
                
                # Для MoCo моделей нужны два входа
                if model.__class__.__name__ in ['MoCo_ResNet', 'MoCo_ViT']:
                    dummy_input1 = torch.randn(1, *input_shape).to(device)
                    dummy_input2 = torch.randn(1, *input_shape).to(device)
                    loss = model(dummy_input1, dummy_input2, m=0.99)
                    print(f"Input shape: {input_shape} (x2 для MoCo)")
                    print(f"Output: loss={loss.item():.6f}")
                else:
                    # Для остальных моделей
                    dummy_input = torch.randn(1, *input_shape).to(device)
                    output = model(dummy_input)
                    print(f"Input shape: {input_shape}")
                    print(f"Output shape: {output.shape}")
            except Exception as e:
                print(f"Error testing model: {str(e)[:100]}...")
            
            
    @staticmethod
    def create_moco_resnet50_adapted(
        input_channels: int = 7,
        dim: int = 256,
        mlp_dim: int = 4096,
        T: float = 1.0
    ):
        """Создание MoCo ResNet50 с адаптацией под вашу архитектуру"""
        import torchvision.models as torchvision_models
        
        # Создаем модель напрямую, как в вашем коде
        model = MoCo_ResNet(
            base_encoder=lambda num_classes: torchvision_models.resnet50(weights=None),
            dim=dim,
            mlp_dim=mlp_dim,
            T=T
        )
        
        # Модифицируем conv1 для многоканальных данных
        model.base_encoder.conv1 = nn.Conv2d(
            input_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        model.momentum_encoder.conv1 = nn.Conv2d(
            input_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Инициализация весов нового conv1
        nn.init.kaiming_normal_(model.base_encoder.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(model.momentum_encoder.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # Заменяем BatchNorm на Identity
        def replace_batchnorm_with_identity(model):
            for name, module in model.named_children():
                if isinstance(module, nn.BatchNorm2d):
                    setattr(model, name, nn.Identity())
                else:
                    replace_batchnorm_with_identity(module)
        
        replace_batchnorm_with_identity(model.base_encoder)
        replace_batchnorm_with_identity(model.momentum_encoder)
        
        print(f"  Создана MoCo ResNet50 с Identity вместо BatchNorm")
        print(f"  Входные каналы: {input_channels}")
        
        return model