import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchcnnbuilder.preprocess.time_series import multi_output_tensor
from skimage.metrics import structural_similarity as ssim
from timesformer.gen_synth_ts import get_anime_timeseries, get_cycled_data
from TimeSformer.vit import TimeSformer
import torch
import numpy as np


# Script for validation TimeSformer on toy example
# Show predicted gif frames in comparison with real data

def mae(prediction, target):
    return float(np.mean(abs(prediction - target)))


def embed_dim_by_img(img, num_heads, emb_mult):
    emb_dim = img * emb_mult
    head_det = emb_dim % num_heads
    if head_det != 0:
        emb_dim = emb_dim - head_det + num_heads
    return emb_dim


def count_patch_size(imgsize):
    patch = imgsize ** 0.5
    if imgsize % patch == 0:
        return patch
    else:
        while imgsize % patch != 0:
            patch = int(patch) - 1
    return patch


# 10 timesteps 45x45 image
data = get_anime_timeseries()
train_data = get_cycled_data(data, 5)[:, :, :, 0]
test_data = get_cycled_data(data, 4)[:, :, :, 0]
img_sizes = (train_data.shape[1], train_data.shape[2])

test_dataset = multi_output_tensor(data=test_data,
                                   pre_history_len=20,
                                   forecast_len=10,
                                   )
dataloader_test = DataLoader(test_dataset, batch_size=2, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
accumulation_steps = 4
lr_max = 0.0001
lr_min = 0.000001
epochs = 20000
predict_period = 10
in_period = 20
batch_size = 2
num_heads = 12
emb_mult = 5
depth = 11

patch_size1 = count_patch_size(img_sizes[0])
patch_size2 = count_patch_size(img_sizes[1])
patch_size = [patch_size1, patch_size2]
embed_dim = embed_dim_by_img(img_sizes[1], num_heads, emb_mult)
dropout = 0.1
attn_drop_rate = 0.1

model = TimeSformer(batch_size=batch_size, output_size=[img_sizes[0], img_sizes[1]], img_size=img_sizes[0],
                    embed_dim=embed_dim,
                    num_frames=4, attention_type='divided_space_time', pretrained_model=False, in_chans=1,
                    out_chans=predict_period,
                    patch_size=patch_size, num_heads=num_heads, in_periods=in_period, depth=depth,
                    emb_mult=emb_mult,
                    attn_drop_rate=attn_drop_rate, dropout=dropout).to(device)
model.load_state_dict(torch.load('anime_weights_2.pt'))

loss_l1 = torch.nn.L1Loss()
for X, y in dataloader_test:
    X = X[:, np.newaxis, :, :]
    y = y[:, np.newaxis, :, :]
    X = X.to(device)
    y = y.squeeze(1).to(device)
    outputs = model(X)

    outputs = model(X)
    loss = loss_l1(outputs, y)

    prediction = outputs.detach().cpu().numpy()[0]
    real = y.detach().cpu().numpy()[0]

    ssim_list = []
    mae_list = []

    fig, (axs) = plt.subplots(2, 10, figsize=(10, 3))
    for i in range(10):
        ssim_list.append(ssim(prediction[i], real[i], data_range=1))
        mae_list.append(mae(prediction[i], real[i]))

        axs[1, i].imshow(prediction[i], cmap='Greys_r', vmax=1, vmin=0)
        axs[1, i].set_title(F'Frame {i}')
        axs[0, i].imshow(real[i], cmap='Greys_r', vmax=1, vmin=0)
        axs[0, i].set_title(F'Frame {i}')
        axs[0, i].set_xticks([])
        axs[1, i].set_xticks([])
        axs[0, i].set_yticks([])
        axs[1, i].set_yticks([])
    plt.suptitle(f'MAE={round(loss.item(), 3)}, SSIM={round(np.mean(ssim_list), 3)}')

    plt.tight_layout()
    plt.show()

    df = pd.DataFrame()
    df['mae'] = mae_list
    df['ssim'] = ssim_list
    df.to_csv('anime_metrics_dist.csv', index=False)

    break
