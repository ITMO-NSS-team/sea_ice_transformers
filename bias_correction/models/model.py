import functools
from torch import nn
import torch.nn.functional as F
import torch


class Corrector(nn.Module):
    def __init__(self, model, channels=6):
        super().__init__()
        self.channels = channels
        self.unet = model

    def forward(self, x_orig):
        x = x_orig
        unet_out = self.unet(x)
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            o_input = torch.split(x_orig.data, 3, dim=-3)
            res = o_input[0] + unet_out.data
            return torch.nn.utils.rnn.PackedSequence(res, x_orig.batch_sizes, x_orig.sorted_indices, x_orig.unsorted_indices)
        else:
            o_input = torch.split(x_orig, 3, dim=-3)
            # print(unet_out.shape, 'out shape')
            # print(o_input[0].shape)
            return o_input[0] + unet_out.view(x.shape[0], x.shape[1], 3, x.shape[3], x.shape[4])


def i2itos2s(cls):
    class Seq2SeqModel(cls):
        def forward(self, x):
            s = x.shape
            out = super().forward(x.flatten(0, 1))
            return out.unflatten(0, s[:2])
    return Seq2SeqModel