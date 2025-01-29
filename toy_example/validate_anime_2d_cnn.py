import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchcnnbuilder.models import ForecasterBase
from torchcnnbuilder.preprocess.time_series import multi_output_tensor

from timesformer.gen_synth_ts import get_anime_timeseries, get_cycled_data
from skimage.metrics import structural_similarity as ssim


# Script to show prediction of 2D CNN with loaded weights as images
# in comparison with real data

def mae(prediction, target):
    return float(np.mean(abs(prediction - target)))


# 10 timesteps 45x45 image
data = get_anime_timeseries()
test_data = get_cycled_data(data, 4)[:, :, :, 0]

test_dataset = multi_output_tensor(data=test_data,
                                   pre_history_len=20,
                                   forecast_len=10,
                                   )

dataloader_test = DataLoader(test_dataset, batch_size=2, shuffle=False)

encoder = ForecasterBase(input_size=(45, 45),
                         n_layers=5,
                         in_time_points=20,
                         out_time_points=10,
                         finish_activation_function=nn.ReLU())
encoder.load_state_dict(torch.load('anime_weights.pt'))

device = 'cuda'
encoder.to(device)
print(encoder)

for X, y in dataloader_test:
    X = X.to(device)
    prediction = encoder(X)
    prediction = prediction.detach().cpu().numpy()[0]
    real = y.numpy()[0]

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
    plt.suptitle(f'MAE={round(mae(prediction, real), 3)}, SSIM={round(np.mean(ssim_list), 3)}')
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame()
    df['mae'] = mae_list
    df['ssim'] = ssim_list
    df.to_csv('anime_metrics_dist_2d_cnn.csv', index=False)

    break
