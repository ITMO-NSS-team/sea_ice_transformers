import torch
import torch.nn as nn
from functools import partial

class MoCo_ResNet(nn.Module):
    """MoCo с ResNet энкодером для контрастного обучения"""
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        super(MoCo_ResNet, self).__init__()
        self.T = T
        
        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)
        
        self._build_projector_and_predictor_mlps(dim, mlp_dim)
        
        # initialize momentum encoder
        self._init_momentum_encoder()
        
        print(f"MoCo ResNet initialized: dim={dim}, mlp_dim={mlp_dim}, T={T}")

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        # Универсальное определение hidden_dim - АДАПТИРОВАНО ПОД ВАШ КОД
        if hasattr(self.base_encoder, 'fc'):
            # Для ResNet - проверяем, не Identity ли это
            if isinstance(self.base_encoder.fc, nn.Identity):
                # Берем размер из предыдущего слоя (avgpool)
                if hasattr(self.base_encoder, 'avgpool'):
                    # Эмпирический размер для ResNet после avgpool
                    hidden_dim = 2048  # для ResNet50/101
                else:
                    # Если нет avgpool, используем фиксированный размер
                    hidden_dim = 512
            else:
                # Обычный Linear слой
                hidden_dim = self.base_encoder.fc.weight.shape[1]
        elif hasattr(self.base_encoder, 'head'):
            # Для ViT
            if isinstance(self.base_encoder.head, nn.Identity):
                hidden_dim = self.base_encoder.embed_dim
            else:
                hidden_dim = self.base_encoder.head.weight.shape[0]
        else:
            # Пробуем найти последний слой другим способом
            for name, module in reversed(list(self.base_encoder.named_modules())):
                if isinstance(module, nn.Linear):
                    hidden_dim = module.weight.shape[1]
                    break
            else:
                raise AttributeError("Не могу найти слой для projector")
        
        print(f"  Определен hidden_dim: {hidden_dim}")
        
        # Projector для base encoder
        projector = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, dim)
        )
        
        # Заменяем слои
        if hasattr(self.base_encoder, 'fc'):
            self.base_encoder.fc = projector
        elif hasattr(self.base_encoder, 'head'):
            self.base_encoder.head = projector
        else:
            # Заменяем последний найденный Linear слой
            for name, module in reversed(list(self.base_encoder.named_children())):
                if isinstance(module, nn.Linear):
                    setattr(self.base_encoder, name, projector)
                    break
        
        # Projector для momentum encoder - АНАЛОГИЧНО
        if hasattr(self.momentum_encoder, 'fc'):
            if isinstance(self.momentum_encoder.fc, nn.Identity):
                momentum_hidden_dim = hidden_dim  # Используем тот же размер
            else:
                momentum_hidden_dim = self.momentum_encoder.fc.weight.shape[1]
                
            self.momentum_encoder.fc = nn.Sequential(
                nn.Linear(momentum_hidden_dim, mlp_dim),
                nn.BatchNorm1d(mlp_dim),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_dim, dim)
            )
        elif hasattr(self.momentum_encoder, 'head'):
            self.momentum_encoder.head = nn.Sequential(
                nn.Linear(hidden_dim, mlp_dim),
                nn.BatchNorm1d(mlp_dim),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_dim, dim)
            )
    
    def _init_momentum_encoder(self):
        """Инициализация momentum encoder - АДАПТИРОВАННАЯ ВЕРСИЯ"""
        # Копируем веса, где возможно
        for param_b, param_m in zip(self.base_encoder.parameters(), 
                                   self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False
        
        # Также заменяем BatchNorm на Identity в momentum_encoder
        for name, module in self.momentum_encoder.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # Находим родительский модуль и заменяем
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                if parent_name:
                    parent = self.momentum_encoder
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                    setattr(parent, name.split('.')[-1], nn.Identity())
                else:
                    setattr(self.momentum_encoder, name, nn.Identity())

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # Нормализация с защитой от NaN
        q = nn.functional.normalize(q, dim=1, eps=1e-8)
        k = nn.functional.normalize(k, dim=1, eps=1e-8)
        
        # Логиты
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        
        # Метки
        N = logits.shape[0]
        labels = torch.arange(N, dtype=torch.long).to(q.device)
        
        # Потеря
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        # Проверка на NaN
        if torch.isnan(loss):
            print("WARNING: NaN loss detected!")
            print(f"q stats: mean={q.mean().item():.6f}, std={q.std().item():.6f}")
            print(f"k stats: mean={k.mean().item():.6f}, std={k.std().item():.6f}")
            
        return loss

    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """
        
        # Проверка входных данных
        if torch.any(torch.isnan(x1)) or torch.any(torch.isnan(x2)):
            print("WARNING: NaN in input data!")
            x1 = torch.nan_to_num(x1, nan=0.0)
            x2 = torch.nan_to_num(x2, nan=0.0)
        
        # update momentum encoder
        self._update_momentum_encoder(m)
        
        # compute queries
        q = self.base_encoder(x1)
        
        # compute keys - no gradient
        with torch.no_grad():
            k = self.momentum_encoder(x2)

        return self.contrastive_loss(q, k)


class MoCoResNetFactory:
    """Фабрика для создания MoCo ResNet моделей"""
    
    @staticmethod
    def create_resnet50(input_channels=7, dim=256, mlp_dim=4096, T=1.0):
        """Создание MoCo ResNet50 модели для многоканальных данных"""
        import torchvision.models as torchvision_models
        
        def base_encoder_resnet50(num_classes=mlp_dim):
            model = torchvision_models.resnet50(weights=None)
            
            # Модификация для многоканальных входных данных
            model.conv1 = nn.Conv2d(
                input_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )
            nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
            
            # Устанавливаем Identity вместо fc - КАК У ВАС В КОДЕ
            model.fc = nn.Identity()
            
            return model
        
        model = MoCo_ResNet(
            base_encoder=base_encoder_resnet50,
            dim=dim,
            mlp_dim=mlp_dim,
            T=T
        )
        
        # Дополнительная инициализация как в вашем коде
        for name, module in model.base_encoder.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                setattr(model.base_encoder, name, nn.Identity())
        
        for name, module in model.momentum_encoder.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                setattr(model.momentum_encoder, name, nn.Identity())
        
        return model
    
    @staticmethod
    def create_resnet101(input_channels=7, dim=256, mlp_dim=4096, T=1.0):
        """Создание MoCo ResNet101 модели"""
        import torchvision.models as torchvision_models
        
        def base_encoder_resnet101(num_classes=mlp_dim):
            model = torchvision_models.resnet101(weights=None)
            
            # Модификация для многоканальных входных данных
            model.conv1 = nn.Conv2d(
                input_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )
            nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
            
            model.fc = nn.Identity()
            return model
        
        return MoCo_ResNet(
            base_encoder=base_encoder_resnet101,
            dim=dim,
            mlp_dim=mlp_dim,
            T=T
        )