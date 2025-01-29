import time
from skimage.transform import resize

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchcnnbuilder.preprocess.time_series import multi_output_tensor
from torchcnnbuilder.models import ForecasterBase

from cnn_forecaster_2d.data_loader import get_timespatial_series


# This script generate 2D CNN with 5 layers and train it with saving weights of model
# To train model for each sea parameter sea_name should be changed
# Seas names: kara, barents, laptev, eastsib, chukchi


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Calculating on device: {device}')

sea_name = 'kara'
start_date = '19790101'
end_date = '20200101'
sea_data, dates = get_timespatial_series(sea_name, start_date, end_date)
sea_data = sea_data[::7]
dates = dates[::7]

pre_history_size = 104
forecast_size = 52

dataset = multi_output_tensor(data=resize(sea_data,
                                          (sea_data.shape[0], sea_data.shape[1] // 2, sea_data.shape[2] // 2),
                                          anti_aliasing=False),
                              forecast_len=forecast_size,
                              pre_history_len=pre_history_size)
dataloader = DataLoader(dataset, batch_size=200, shuffle=False)
print('Loader created')

encoder = ForecasterBase(input_size=(sea_data.shape[1] // 2, sea_data.shape[2] // 2),
                         n_layers=5,
                         in_time_points=pre_history_size,
                         out_time_points=forecast_size)
encoder.to(device)
print(encoder)

optimizer = optim.Adam(encoder.parameters(), lr=0.001)
criterion = nn.L1Loss()

losses = []
start = time.time()
epochs = 1000
best_loss = 999
best_model = None
for epoch in range(epochs):
    loss = 0
    for train_features, test_features in dataloader:
        train_features = train_features.to(device)
        test_features = test_features.to(device)
        optimizer.zero_grad()
        outputs = encoder(train_features)
        train_loss = criterion(outputs, test_features)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()

    loss = loss / len(dataloader)
    if loss is None:
        break
    if loss < best_loss and loss is not None:
        print('Upd best model')
        best_model = encoder
        best_loss = loss
    losses.append(loss)

    print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))

end = time.time() - start
print(f'Runtime seconds: {end}')
torch.save(encoder.state_dict(), f"models/{sea_name}_{pre_history_size}_{forecast_size}_l1({start_date}-{end_date}){epochs}.pt")
plt.plot(np.arange(len(losses)), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Runtime={end}')
plt.savefig(f"models/{sea_name}_{pre_history_size}_{forecast_size}_l1({start_date}-{end_date}){epochs}.png")
plt.show()
