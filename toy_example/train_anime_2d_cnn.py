import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.data import DataLoader
from torchcnnbuilder.models import ForecasterBase
from torchcnnbuilder.preprocess.time_series import multi_output_tensor

from data.gen_synth_ts import get_anime_timeseries, get_cycled_data

# This script provide training 2D CNN for toy example cycled gif-file forecasting
# Save checkpoints of model weights and process of convergence as images of predict and curves

# 10 timesteps 45x45 image
data = get_anime_timeseries()
train_data = get_cycled_data(data, 5)[:, :, :, 0]
test_data = get_cycled_data(data, 4)[:, :, :, 0]

train_dataset = multi_output_tensor(data=train_data,
                                    pre_history_len=20,
                                    forecast_len=10,
                                    )
test_dataset = multi_output_tensor(data=test_data,
                                   pre_history_len=20,
                                   forecast_len=10,
                                   )

dataloader_train = DataLoader(train_dataset, batch_size=10, shuffle=False)
dataloader_test = DataLoader(test_dataset, batch_size=10, shuffle=False)

encoder = ForecasterBase(input_size=(45, 45),
                         n_layers=5,
                         in_time_points=20,
                         out_time_points=10,
                         finish_activation_function=nn.ReLU())
device = 'cuda'
encoder.to(device)
print(encoder)


optimizer = optim.Adam(encoder.parameters(), lr=0.0001)
criterion = nn.L1Loss()

losses = []
epochs = 100000
best_loss = 999
best_model = None

NAME = f'ANIME({epochs}ep)'
if not os.path.isdir(f'{NAME}'):
    os.mkdir(f'{NAME}')
if not os.path.isdir(f'{NAME}/opt_hist_images'):
    os.mkdir(f'{NAME}/opt_hist_images')

for epoch in range(epochs):
    loss = 0
    for train_features, test_features in dataloader_train:
        train_features = train_features.to(device)
        test_features = test_features.to(device)
        optimizer.zero_grad()
        outputs = encoder(train_features)
        train_loss = criterion(outputs, test_features)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()

    loss = loss / len(dataloader_train)
    if loss < best_loss and loss is not None:
        print('Upd best model')
        best_model = encoder
        best_loss = loss
    losses.append(loss)

    print("epoch : {}/{}, loss = {:.8f}".format(epoch + 1, epochs, loss))


    if epoch % 10000 == 0 or epoch == 5 or epoch==50 or epoch==300 or epoch == 1000 or epoch == 5000:

        for X, y in dataloader_test:
            X = X.to(device)
            prediction = encoder(X)
            prediction = prediction.detach().cpu().numpy()[0]
            real = y.numpy()[0]

            fig, (axs) = plt.subplots(2, 10, figsize=(10, 3))
            for i in range(10):
                axs[1, i].imshow(prediction[i], cmap='Greys_r', vmax=1, vmin=0)
                axs[1, i].set_title(F'Frame {i}')
                axs[0, i].imshow(real[i], cmap='Greys_r', vmax=1, vmin=0)
                axs[0, i].set_title(F'Frame {i}')
                axs[0, i].set_xticks([])
                axs[1, i].set_xticks([])
                axs[0, i].set_yticks([])
                axs[1, i].set_yticks([])
            plt.suptitle(f'Epoch={epoch}, loss={round(loss, 3)}')
            plt.tight_layout()
            plt.savefig(f'{NAME}/opt_hist_images/test_images_{epoch}.png')
            plt.show()

            fig = plt.figure(figsize=(6, 4))
            plt.plot(np.arange(len(losses)), losses)
            plt.title(f'Optimization loss = {round(losses[-1], 3)}')
            plt.ylabel('L1 loss')
            plt.xlabel('Epochs')
            plt.savefig(f'{NAME}/opt_hist_images/opt_hist_{epoch}.png')
            plt.show()
            break

        torch.save(best_model.state_dict(),
                   f'{NAME}/anime_weights.pt')

    torch.save(best_model.state_dict(),
               f'{NAME}/anime_weights.pt')