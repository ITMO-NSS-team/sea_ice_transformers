import torch
import torch.nn as nn
import torch.nn.functional as F

class FrozenViTEncoderDecoder(nn.Module):
    """
    Декодер для восстановления полей океана из сжатых представлений ViT
    """
    def __init__(
        self,
        encoder,
        in_channels=7,
        height=349,
        width=661,
        encoder_img_size=661,
        latent_dim=256,
        base_channels=256,
    ):
        super().__init__()
        
        self.encoder = encoder
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.encoder_img_size = encoder_img_size
        self.base_channels = base_channels
        
        # Проверка размеров
        if encoder_img_size < height or encoder_img_size < width:
            raise ValueError("encoder_img_size должен быть не меньше height и width")
        
        # Замораживаем энкодер
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        
        # Узнаём размер выхода энкодера
        with torch.no_grad():
            enc_device = next(self.encoder.parameters()).device
            dummy = torch.zeros(1, in_channels, encoder_img_size, encoder_img_size, device=enc_device)
            enc_out = self.encoder(dummy)
            enc_dim = enc_out.shape[1]
        
        # Линейная проекция в латент
        if latent_dim is not None and latent_dim > 0 and latent_dim != enc_dim:
            self.to_latent = nn.Linear(enc_dim, latent_dim)
            latent_actual = latent_dim
        else:
            self.to_latent = nn.Identity()
            latent_actual = enc_dim
        
        # Расчёт размеров для декодера
        self.num_upsamples = 4
        factor = 2 ** self.num_upsamples
        
        H0 = height // factor
        W0 = width // factor
        
        self.H0 = H0
        self.W0 = W0
        
        # Остаток для output_padding
        rem_h = height - H0 * factor
        rem_w = width - W0 * factor
        
        # Разложение остатка по слоям
        def decompose_remainder(rem, n_bits):
            bits = []
            for i in range(n_bits - 1, -1, -1):
                bit = (rem >> i) & 1
                bits.append(bit)
            return bits
        
        ops_h = decompose_remainder(rem_h, self.num_upsamples)
        ops_w = decompose_remainder(rem_w, self.num_upsamples)
        
        # Линейный слой для создания feature map
        self.fc = nn.Linear(latent_actual, base_channels * H0 * W0)
        
        # Слои декодера
        def up_block(in_ch, out_ch, layer_idx):
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_ch, out_ch,
                    kernel_size=4, stride=2, padding=1,
                    output_padding=(ops_h[layer_idx], ops_w[layer_idx])
                ),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        
        self.decoder = nn.Sequential(
            up_block(base_channels, base_channels // 2, 0),
            up_block(base_channels // 2, base_channels // 4, 1),
            up_block(base_channels // 4, base_channels // 8, 2),
            up_block(base_channels // 8, base_channels // 8, 3),
            nn.Conv2d(base_channels // 8, in_channels, kernel_size=3, padding=1),
        )
    
    def _pad_for_vit(self, x):
        """Паддинг для ViT энкодера"""
        _, _, H, W = x.shape
        
        pad_h_total = self.encoder_img_size - H
        pad_w_total = self.encoder_img_size - W
        
        if pad_h_total == 0 and pad_w_total == 0:
            return x
        
        pad_top = pad_h_total // 2
        pad_bottom = pad_h_total - pad_top
        pad_left = pad_w_total // 2
        pad_right = pad_w_total - pad_left
        
        x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return x_padded
    
    def forward(self, x):
        # Паддинг для ViT
        x_vit = self._pad_for_vit(x)
        
        # Фиксированный энкодер
        with torch.no_grad():
            z_enc = self.encoder(x_vit)
        
        # В латентное пространство
        z = self.to_latent(z_enc)
        
        # В feature map
        feat = self.fc(z)
        feat = feat.view(z.shape[0], self.base_channels, self.H0, self.W0)
        
        # Декодирование
        out = self.decoder(feat)
        
        return out


class ViTDecoderFactory:
    """Фабрика для создания декодеров ViT"""
    
    @staticmethod
    def create_default(encoder, in_channels=7, height=349, width=661, encoder_img_size=661):
        """Создание декодера с параметрами по умолчанию"""
        return FrozenViTEncoderDecoder(
            encoder=encoder,
            in_channels=in_channels,
            height=height,
            width=width,
            encoder_img_size=encoder_img_size,
            latent_dim=256,
            base_channels=256
        )