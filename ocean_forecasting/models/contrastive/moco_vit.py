import torch
import torch.nn as nn
import math
from functools import partial, reduce
from operator import mul

class VisionTransformerMoCo(nn.Module):
    """Vision Transformer с фиксированным позиционным кодированием для MoCo"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                 qkv_bias=True, norm_layer=None, stop_grad_conv1=False):
        super().__init__()
        
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        # Patch embedding
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            in_chans=in_chans, embed_dim=embed_dim
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding (фиксированное 2D sin-cos)
        self.pos_embed = self.build_2d_sincos_position_embedding(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, norm_layer=norm_layer
            )
            for _ in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()  # Временная замена
        
        # Инициализация весов
        self._init_weights(stop_grad_conv1)
        
    def build_2d_sincos_position_embedding(self, img_size=224, patch_size=16, 
                                          embed_dim=768, temperature=10000.):
        """Создание 2D sin-cos позиционного кодирования"""
        h, w = img_size // patch_size, img_size // patch_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='xy')
        
        assert embed_dim % 4 == 0, 'Embed dim must be divisible by 4'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)
        
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([
            torch.sin(out_w), torch.cos(out_w),
            torch.sin(out_h), torch.cos(out_h)
        ], dim=1)[None, :, :]
        
        # Добавляем CLS token
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        pos_embed.requires_grad = False
        return pos_embed
    
    def _init_weights(self, stop_grad_conv1):
        """Инициализация весов ViT"""
        # Инициализация CLS token
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Инициализация patch embedding
        if hasattr(self.patch_embed, 'proj'):
            nn.init.xavier_uniform_(self.patch_embed.proj.weight)
            if self.patch_embed.proj.bias is not None:
                nn.init.zeros_(self.patch_embed.proj.bias)
            
            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False
        
        # Инициализация блоков трансформера
        for block in self.blocks:
            nn.init.xavier_uniform_(block.attn.qkv.weight)
            nn.init.zeros_(block.attn.qkv.bias)
            nn.init.xavier_uniform_(block.mlp.fc1.weight)
            nn.init.zeros_(block.mlp.fc1.bias)
            nn.init.xavier_uniform_(block.mlp.fc2.weight)
            nn.init.zeros_(block.mlp.fc2.bias)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Добавляем CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Добавляем позиционное кодирование
        x = x + self.pos_embed
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        # Нормализация
        x = self.norm(x)
        
        # Возвращаем только CLS token
        return x[:, 0]


class PatchEmbed(nn.Module):
    """Разбиение изображения на патчи"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Block(nn.Module):
    """Блок трансформера"""
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4,
                 qkv_bias=True, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm
            
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Attention(nn.Module):
    """Многоголовое внимание"""
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """Многослойный перцептрон"""
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MoCo_ViT(nn.Module):
    """MoCo с Vision Transformer энкодером"""
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        super(MoCo_ViT, self).__init__()
        self.T = T
        
        # build encoders
        self.base_encoder = base_encoder
        self.momentum_encoder = type(base_encoder)(
            img_size=base_encoder.patch_embed.img_size[0],
            patch_size=base_encoder.patch_embed.patch_size[0],
            in_chans=base_encoder.in_chans,
            embed_dim=base_encoder.embed_dim,
            depth=len(base_encoder.blocks),
            num_heads=base_encoder.blocks[0].attn.num_heads,
            mlp_ratio=4,
            qkv_bias=True
        )
        
        self._build_projector_and_predictor_mlps(dim, mlp_dim)
        
        # initialize momentum encoder
        self._init_momentum_encoder()
        
        print(f"MoCo ViT initialized: dim={dim}, mlp_dim={mlp_dim}, T={T}")
    
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        """Build MLPs for projector and predictor - АДАПТИРОВАННАЯ ВЕРСИЯ"""
        # Используем embed_dim для ViT
        hidden_dim = self.base_encoder.embed_dim
        
        # Projector для base encoder
        self.base_encoder.head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.Identity(),  # Заменяем BatchNorm на Identity как у вас
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, dim)
        )
        
        # Projector для momentum encoder
        self.momentum_encoder.head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.Identity(),  # Заменяем BatchNorm на Identity
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, dim)
        )
    
    def _init_momentum_encoder(self):
        """Инициализация momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), 
                                   self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False
    
    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), 
                                   self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)
    
    def contrastive_loss(self, q, k):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]
        labels = torch.arange(N, dtype=torch.long).to(q.device)
        
        return nn.CrossEntropyLoss()(logits, labels)
    
    def forward(self, x1, x2, m):
        # update momentum encoder
        self._update_momentum_encoder(m)
        
        # compute queries
        q = self.base_encoder(x1)
        
        # compute keys
        with torch.no_grad():
            k = self.momentum_encoder(x2)
        
        return self.contrastive_loss(q, k)


class MoCoViTFactory:
    """Фабрика для создания MoCo ViT моделей"""
    
    @staticmethod
    def create_vit_small(img_size=224, in_chans=7, dim=256, mlp_dim=4096, T=1.0):
        """Создание MoCo ViT-Small модели"""
        base_encoder = VisionTransformerMoCo(
            img_size=img_size,
            patch_size=16,
            in_chans=in_chans,
            embed_dim=384,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True
        )
        
        return MoCo_ViT(
            base_encoder=base_encoder,
            dim=dim,
            mlp_dim=mlp_dim,
            T=T
        )
    
    @staticmethod
    def create_vit_base(img_size=224, in_chans=7, dim=256, mlp_dim=4096, T=1.0):
        """Создание MoCo ViT-Base модели"""
        base_encoder = VisionTransformerMoCo(
            img_size=img_size,
            patch_size=16,
            in_chans=in_chans,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True
        )
        
        return MoCo_ViT(
            base_encoder=base_encoder,
            dim=dim,
            mlp_dim=mlp_dim,
            T=T
        )