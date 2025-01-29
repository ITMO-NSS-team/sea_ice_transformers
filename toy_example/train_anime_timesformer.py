import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchcnnbuilder.preprocess.time_series import multi_output_tensor

from timesformer.gen_synth_ts import get_anime_timeseries, get_cycled_data
import os
from TimeSformer.vit import TimeSformer
import torch
from tqdm import tqdm
import numpy as np


# This script provide training TimeSformer for toy example cycled gif-file forecasting
# Save checkpoints of model weights and process of convergence as images of predict and curves
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_size_for_reize(img_size, num_heads):
    if img_size[0] % num_heads != 0:
        img_size_new = [(int(img_size[0] / num_heads) + 1) * num_heads, img_size[1]]
        print("New image size is", img_size_new)
        return img_size_new
    else:
        return img_size


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
accumulation_steps = 4
lr_max = 0.0001
lr_min = 0.000001
epochs = 100000
predict_period = 10  # need for dataloader
in_period = 20  # need for dataloader
batch_size = 2  # need for dataloader
num_heads = 12
emb_mult = 5  # Necessary to control the internal size of the embedding to manage the required model size.
NAME = f'ANIME({epochs}ep_batch{batch_size})'
depth = 11

# 10 timesteps 45x45 image
data = get_anime_timeseries()
train_data = get_cycled_data(data, 5)[:, :, :, 0]
test_data = get_cycled_data(data, 4)[:, :, :, 0]
img_sizes = (train_data.shape[1], train_data.shape[2])

train_dataset = multi_output_tensor(data=train_data,
                                    pre_history_len=20,
                                    forecast_len=10,
                                    )
test_dataset = multi_output_tensor(data=test_data,
                                   pre_history_len=20,
                                   forecast_len=10,
                                   )

dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_len = dataloader_train.__len__()
test_len = dataloader_test.__len__()

if img_sizes[1] > img_sizes[0]:
    img_sizes = (img_sizes[1], img_sizes[0])
else:
    patch_size1 = count_patch_size(img_sizes[0])  # int(img_sizes[0]/(img_sizes[0]*2)**0.5)
    patch_size2 = count_patch_size(img_sizes[1])
    patch_size = [patch_size1, patch_size2]  # int(img_sizes[1]/(img_sizes[1]*2)**0.5)
embed_dim = embed_dim_by_img(img_sizes[1], num_heads, emb_mult)
dropout = 0.1
attn_drop_rate = 0.1

loss_l1 = torch.nn.L1Loss()

model = TimeSformer(batch_size=batch_size, output_size=[img_sizes[0], img_sizes[1]], img_size=img_sizes[0],
                    embed_dim=embed_dim,
                    num_frames=4, attention_type='divided_space_time', pretrained_model=False, in_chans=1,
                    out_chans=predict_period,
                    patch_size=patch_size, num_heads=num_heads, in_periods=in_period, depth=depth,
                    emb_mult=emb_mult,
                    attn_drop_rate=attn_drop_rate, dropout=dropout).to(device)

print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_max)  # Weight decay

'''optimizer_sch = CosineLRScheduler(optimizer, t_initial=120, lr_min=lr_min * 10,
                                  warmup_t=5, cycle_limit=1, warmup_lr_init=lr_min, warmup_prefix=False,
                                  t_in_epochs=True,
                                  noise_range_t=None, noise_pct=0.67, noise_std=1.0,
                                  noise_seed=42, initialize=True)'''

img = 0
step = 0
ep = 0
test_step = 0
current_step = 0
if not os.path.isdir(f'{NAME}'):
    os.mkdir(f'{NAME}')
if not os.path.isdir(f'{NAME}/opt_hist_images'):
    os.mkdir(f'{NAME}/opt_hist_images')

train_losses = []
for epoch in tqdm(range(epochs)):
    ep += 1
    model.train()
    # TRAIN
    train_losses_ep = []
    for X, y in dataloader_train:
        step += 1
        X = X[:, np.newaxis, :, :]
        y = y[:, np.newaxis, :, :]
        X = X.to(device)
        y = y.squeeze(1).to(device)
        outputs = model(X)

        loss = loss_l1(outputs, y)
        train_losses_ep.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_losses.append(np.mean(train_losses_ep))

    #optimizer_sch.step(ep)
    optimizer.zero_grad()
    torch.cuda.empty_cache()

    # TEST
    model.eval()
    with torch.no_grad():
        for X, y in dataloader_test:
            test_step += 1
            X = X[:, np.newaxis, :, :]
            y = y[:, np.newaxis, :, :]
            X = X.to(device)
            y = y.squeeze(1).to(device)
            outputs = model(X)
            loss = loss_l1(outputs, y)
            #test_losses_ep.append(loss.item())
            break

        #test_losses.append(np.mean(test_losses_ep))
        torch.cuda.empty_cache()

    print(f'train_loss = {round(train_losses[-1], 5)}')
    if ep % 10000 == 0 or ep == 5 or ep == 50 or ep == 100 or ep == 300 or ep == 1000 or ep == 5000:

        prediction = outputs.detach().cpu().numpy()[0]
        real = y.detach().cpu().numpy()[0]

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
        plt.suptitle(f'Epoch={epoch}, loss={round(loss.item(), 3)}')
        plt.tight_layout()
        plt.savefig(f'{NAME}/opt_hist_images/test_images_{epoch}.png')
        plt.show()

        fig = plt.figure(figsize=(6, 4))
        plt.plot(np.arange(len(train_losses)), train_losses)
        plt.title(f'Optimization loss = {round(train_losses[-1], 3)}')
        plt.ylabel('L1 loss')
        plt.xlabel('Epochs')
        plt.savefig(f'{NAME}/opt_hist_images/opt_hist_{epoch}.png')
        plt.show()

        torch.save(model.state_dict(),
                   f'{NAME}/anime_weights_{batch_size}.pt')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.plot(np.arange(len(train_losses)), train_losses, label='train_loss')
ax1.legend()
ax2.imshow(outputs.detach().cpu().numpy()[0][0], cmap='Grays')
ax2.set_title('Predicted')
ax3.imshow(y.detach().cpu().numpy()[0][0], cmap='Grays')
ax3.set_title('Real')
plt.suptitle(f'Epoch={epoch}')
plt.show()
plt.savefig(f'{NAME}/opt_history.png')

torch.save(model.state_dict(),
           f'{NAME}/anime_weights_{batch_size}.pt')
df = pd.DataFrame()
df['train'] = train_losses
df.to_csv(f'{NAME}/opt_history.csv', index=False)
