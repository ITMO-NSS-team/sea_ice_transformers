import time
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.transform import resize
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchcnnbuilder.models import ForecasterBase
from torchcnnbuilder.preprocess.time_series import multi_output_tensor
from cnn_forecaster_2d.data_loader import get_timespatial_series

# Script for training 3D CNN for each sea with weights and convergence curve save
# To train for each sea change sea_name: kara, barents, laptev, eastsib, chukchi


pre_history_size = 104
forecast_size = 52

device = "cuda"
default_lr = 1e-3
max_epochs = 1000

sea_name = "kara"
start_date = "19790101"
end_date = "20200101"
kernel = (52, 3, 3)
sea_data, dates = get_timespatial_series(sea_name, start_date, end_date)
sea_data = sea_data[::7]
dates = dates[::7]

print("Data loaded")
dataset = multi_output_tensor(
    data=resize(
        sea_data,
        (sea_data.shape[0], sea_data.shape[1] // 2, sea_data.shape[2] // 2),
        anti_aliasing=False,
    ),
    forecast_len=forecast_size,
    pre_history_len=pre_history_size,
)
dataloader = DataLoader(dataset, batch_size=300, shuffle=False)
print("loader created")

forecaster_params = {
    "input_size": (sea_data.shape[1] // 2, sea_data.shape[2] // 2),
    "n_layers": 2,
    "in_time_points": pre_history_size,
    "out_time_points": forecast_size,
    "convolve_params": {"kernel_size": kernel},
    "transpose_convolve_params": {"kernel_size": kernel},
    "conv_dim": 3,
    "activation_function": nn.ReLU(inplace=True),
    "finish_activation_function": nn.ReLU(inplace=True),
}

model = ForecasterBase(**forecaster_params).to(device)
print(model)
optimizer = optim.Adam(model.parameters(), lr=default_lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.95, patience=10
)

criterion = nn.L1Loss()

loss_history = []

best_val = float("inf")
epochs_no_improve = 0
best_model = None

start = time.time()

for epoch in range(max_epochs):
    model.train()
    loss = 0
    ssim_loss = 0
    for X, Y in dataloader:
        X = X[:, None].to(device)
        Y = Y[:, None].to(device)

        optimizer.zero_grad()

        outputs = model(X)
        train_loss = criterion(outputs, Y)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()

    loss = loss / len(dataloader)
    scheduler.step(loss)
    loss_history.append(loss)

    if loss < best_val:
        best_model = model
        best_val = loss
        print("Upd model")

    print(f"-- epoch : {epoch + 1}/{max_epochs}, {loss=}, lr={scheduler.get_last_lr()}")

    model.eval()

end = time.time() - start

torch.save(
    model.state_dict(),
    f"models/{sea_name}_104_52_(2l_{kernel})({start_date}-{end_date}).pt",
)

plt.plot(list(range(len(loss_history))), loss_history)
plt.grid()
plt.xlabel("Epoch")
plt.ylabel(f"L1Loss")
plt.title(f"Runtime={end}")
plt.savefig(f"models/{sea_name}_104_52_(2l_{kernel})({start_date}-{end_date}).png")
plt.show()
