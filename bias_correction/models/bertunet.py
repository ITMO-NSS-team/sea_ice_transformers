import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.models.bert.configuration_bert import BertConfig
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence


from .unet_parts import DoubleConv, Down, Up, OutConv

class BERTUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, h=210, w=280, chan_factor=2):
        super(BERTUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        print('Model:', n_channels, '->', n_classes)

        self.unet_mode = 'train'  # Режим для U-net
        self.transformer_mode = 'train'  # Режим для трансформера
        self.chan_factor = chan_factor
        self.factor = 2 if bilinear else 1
        self.h = h
        self.w = w
        self.latent_dim = 1024 // self.chan_factor // self.factor
        self.patch_encoding = nn.Embedding((self.h // 16) * (self.w // 16), self.latent_dim)
        self.pos_encoding = nn.Embedding(10, self.latent_dim)
        
        self.causal_bro = BertEncoder(BertConfig(
            hidden_size=self.latent_dim,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=self.latent_dim * 4
        ))

        self.inc = (DoubleConv(n_channels, 64 // chan_factor))
        self.down1 = (Down(64 // chan_factor, 128 // chan_factor))
        self.down2 = (Down(128 // chan_factor, 256 // chan_factor))
        self.down3 = (Down(256 // chan_factor, 512 // chan_factor))
        self.down4 = (Down(512 // chan_factor, 1024 // chan_factor // self.factor))
        self.up1 = (Up(1024 // chan_factor, 512 // chan_factor // self.factor, bilinear))
        self.up2 = (Up(512 // chan_factor, 256 // chan_factor // self.factor, bilinear))
        self.up3 = (Up(256 // chan_factor, 128 // chan_factor // self.factor, bilinear))
        self.up4 = (Up(128 // chan_factor, 64 // chan_factor))
        self.outc = (OutConv(64 // chan_factor, n_classes))

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = x5.view(orig_shape[0], orig_shape[1], self.latent_dim, (self.h // 16) * (self.w // 16))

        x5 = x5.permute(1, 0, 3, 2)

        x5_patch = self.patch_encoding(torch.arange(0, x5.shape[2], device=x5.device))
        x5_pos = self.pos_encoding(torch.arange(0, x5.shape[1], device=x5.device))
        x5 = x5 + x5_patch.view(1, 1, (self.h // 16) * (self.w // 16), self.latent_dim) + x5_pos.view(1, orig_shape[0], 1, self.latent_dim)
        x5 = x5.reshape(x5.shape[0], -1, self.latent_dim)
        x5 = self.causal_bro(x5).last_hidden_state
        x5 = x5.reshape(x5.shape[0] * orig_shape[0], (self.h // 16), (self.w // 16), self.latent_dim)
        x5 = x5.permute(0, 3, 1, 2)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x).view(orig_shape[0], orig_shape[1], 3, orig_shape[3], orig_shape[4])
        return logits
    
    def freeze_backbone(self):
        """
        Замораживает параметры U-net части модели, оставляя трансформер для обучения.
        """
        for name, param in self.named_parameters():
            if not name.startswith("causal_bro") and not "encoding" in name:  # Только трансформер остается обучаемым
                param.requires_grad = False
        print("U-net backbone заморожен. Параметры трансформера остаются обучаемыми.")

    def set_mode(self, unet_mode='eval', transformer_mode='train'):
        """
        Устанавливает режимы для U-net и трансформера.

        Args:
            unet_mode (str): Режим для U-net ('train' или 'eval').
            transformer_mode (str): Режим для трансформера ('train' или 'eval').
        """
        self.unet_mode = unet_mode  # Режим для U-net
        self.transformer_mode = transformer_mode  # Режим для трансформера
        # Определяем, какие слои относятся к U-net и трансформеру
        for name, module in self.named_modules():
            if name.startswith("causal_bro") or "encoding" in name:
                module.train() if transformer_mode == 'train' else module.eval()
            else:
                module.train() if unet_mode == 'train' else module.eval()

        print(f"U-net переведен в режим {unet_mode}. Transformer переведен в режим {transformer_mode}.")

    def train(self, mode: bool = True):
        for name, module in self.named_modules():
            if name == '':
                module.training = mode
                continue
            if name.startswith("causal_bro") or "encoding" in name:
                module.train(self.transformer_mode == 'train' and mode)
            else:
                module.train(self.unet_mode == 'train' and mode)
        return self


class S2SBERTUnet(nn.Module):
    def __init__(self, n_channels, n_classes, max_len=24, chan_factor=2, bilinear=True):
        super(S2SBERTUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.max_len = max_len
        print('Model:', n_channels, '->', n_classes)

        # self.spatial_map_encode = nn.Parameter(128, 105, 140)

        self.emb_size = 512 // chan_factor
        factor = 2 if bilinear else 1
        # self.spatial_map_encode = nn.Parameter(torch.empty([1, 128, 52, 70]))
        # self.spatial_map_decode = nn.Parameter(torch.empty([1, 128, 52, 70]))

        # nn.init.normal_(self.spatial_map_encode)
        # nn.init.normal_(self.spatial_map_decode)

        self.patch_encoding = nn.Embedding(13 * 17, 512 // 2)
        self.pos_encoding = nn.Embedding(self.max_len, 512 // 2)

        self.causal_bro = BertEncoder(BertConfig(
            hidden_size=512 // 2,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=512 * 4 // 2
        ))

        self.inc = (DoubleConv(n_channels, 64 // chan_factor))
        self.down1 = (Down(64 // chan_factor, 128 // chan_factor))
        self.down2 = (Down(128 // chan_factor, 256 // chan_factor))
        self.down3 = (Down(256 // chan_factor, 512 // chan_factor))
        self.down4 = (Down(512 // chan_factor, 1024 // chan_factor // factor))
        self.up1 = (Up(1024 // chan_factor, 512 // chan_factor // factor, bilinear))
        self.up2 = (Up(512 // chan_factor, 256 // chan_factor // factor, bilinear))
        self.up3 = (Up(256 // chan_factor, 128 // chan_factor // factor, bilinear))
        self.up4 = (Up(128 // chan_factor, 64 // chan_factor))
        self.outc = (OutConv(64 // chan_factor, n_classes))

    def forward(self, packed_x):
        assert type(packed_x) == torch.nn.utils.rnn.PackedSequence
        x = packed_x.data
        # print(x.shape, 'input data shape')
        # x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x3 = x3 + self.spatial_map_encode
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x5.shape, 'x5')
        packed_x = PackedSequence(x5, packed_x.batch_sizes, packed_x.sorted_indices, packed_x.unsorted_indices)
        pad_x5, lengths = pad_packed_sequence(packed_x, batch_first=False, total_length=self.max_len)
        # print(pad_x5.device, lengths.device, 'x5 and len devices')
        # print(x5.shape, pad_x5.shape, 'shape after padding')
        # print(lengths, 'lengths')
        batch_size = len(lengths)
        attention_mask = torch.arange(self.max_len, device=pad_x5.device).expand(batch_size, self.max_len) < lengths.unsqueeze(1).to(pad_x5.device)
        # import matplotlib.pyplot as plt
        # plt.imshow(attention_mask)
        attention_mask = attention_mask.unsqueeze(-1).unsqueeze(-1)  # [bs, sl, 1, 1]
        attention_mask = attention_mask.expand(-1, -1, 13, 17)        # [bs, sl, h, w]
        attention_mask = attention_mask.reshape(batch_size, self.max_len * 13 * 17) * 1.     # [bs, sl*h*w]

        # print(pad_x5.shape, attention_mask.shape, 'pad_x5, attention_mask')

        pad_x5 = pad_x5.flatten(-2, -1)

        pad_x5 = pad_x5.permute(1, 0, 3, 2)
        # print(pad_x5.shape, 'pad x5 shape before applying encoding')

        x5_patch = self.patch_encoding(torch.arange(0, pad_x5.shape[2], device=pad_x5.device))
        # print(pad_x5.shape[1], self.pos_encoding.weight.shape)
        x5_pos = self.pos_encoding(torch.arange(0, pad_x5.shape[1], device=pad_x5.device))
        # print(pad_x5.shape, x5_patch.shape, x5_pos.shape)
        # return pad_x5, attention_mask
        # print(attention_mask)
        # print((pad_x5.reshape(batch_size, -1, self.emb_size) * attention_mask.unsqueeze(-1) - pad_x5.reshape(batch_size, -1, self.emb_size)).sum())
        # assert torch.allclose(pad_x5.reshape(batch_size, -1, self.emb_size) * attention_mask.unsqueeze(-1), pad_x5.reshape(batch_size, -1, self.emb_size))
        pad_x5 = pad_x5 + x5_patch.view(1, 1, 13 * 17, self.emb_size) + x5_pos.view(1, self.max_len, 1, self.emb_size)
        # print(pad_x5.shape)
        pad_x5 = pad_x5.reshape(batch_size, -1, self.emb_size)
        # print(pad_x5.shape, attention_mask.shape, ' bert input')

        x5 = self.causal_bro(pad_x5, attention_mask=attention_mask[:, None, None]).last_hidden_state
        # print(x5.shape, 'x5 just after bert')
        x5 = x5.reshape(batch_size, self.max_len, 13, 17, self.emb_size)
        x5 = x5.permute(0, 1, 4, 2, 3)
        # print(x5.shape, 'x5 after bert and reshape')
        x5 = pack_padded_sequence(x5, lengths, batch_first=True, enforce_sorted=False).data
        # print(x5.shape, 'x5')
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # print(x.shape)
        logits = self.outc(x)#.view(orig_shape[0], orig_shape[1], 3, orig_shape[3], orig_shape[4])
        # print(logits.shape, 'out')
        logits = PackedSequence(logits, packed_x.batch_sizes, packed_x.sorted_indices, packed_x.unsorted_indices)
        return logits
