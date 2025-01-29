import os
from copy import deepcopy
from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta
from skimage.transform import resize

from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch
from torch import tensor, nn
from torchcnnbuilder.models import ForecasterBase
import matplotlib.pyplot as plt
from cnn_forecaster_2d.visualizator import plot_comparison_map, full_name


# This script load 3D CNN weights and produce validation
# /path_to_data/ should be replaced to directory of real data location
# Maps of comparison of prediction with real data and metrics are saved


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Calculating on device: {device}')


def get_prehistory(start_date, sea_name, pre_history_size, data_freq=7):
    prehistory_dates = pd.date_range(start_date - relativedelta(days=(pre_history_size * data_freq)),
                                     start_date,
                                     freq=f'{data_freq}D')[-pre_history_size:]
    datamodule_path = '/path_to_data//OSISAF'
    files_path = f'{datamodule_path}/{sea_name}'
    matrices = []
    dates = []
    for date in prehistory_dates:
        file_name = date.strftime('osi_%Y%m%d.npy')
        matrix = np.load(f'{files_path}/{file_name}').astype(float)
        matrices.append(matrix)
        dates.append(date.strftime('%Y%m%d'))
    return np.array(matrices), dates


def get_target(start_date, sea_name, forecast_size, data_freq=7):
    forecast_dates = pd.date_range(start_date,
                                   start_date + relativedelta(days=(forecast_size * data_freq)),
                                   freq=f'{data_freq}D')[:forecast_size]
    datamodule_path = '/path_to_data//OSISAF'
    files_path = f'{datamodule_path}/{sea_name}'
    matrices = []
    dates = []
    for date in forecast_dates:
        file_name = date.strftime('osi_%Y%m%d.npy')
        matrix = np.load(f'{files_path}/{file_name}').astype(float)
        matrices.append(matrix)
        dates.append(date.strftime('%Y%m%d'))
    return np.array(matrices), dates


def fix_range(image):
    image[image > 1] = 1
    image[image < 0] = 0
    return image


def fix_border(sea_name, image):
    datamodule_path = '/path_to_data/'
    mask = np.load(f'{datamodule_path}/coastline_masks/{sea_name}_mask.npy')
    mask = np.repeat(np.expand_dims(mask, axis=0), image.shape[0], axis=0)
    image[mask == 0] = 0
    return image


def mae(prediction, target):
    return np.mean(abs(prediction - target))


def binary_accuracy(prediction, target):
    prediction = deepcopy(prediction)
    target = deepcopy(target)

    prediction[prediction < 0.2] = 0
    prediction[prediction >= 0.2] = 1
    target[target < 0.2] = 0
    target[target >= 0.2] = 1

    diff = target-prediction
    errors_num = len(np.where(diff == 0)[0])
    acc = errors_num/prediction.size
    return acc


def calculate_metrics(forecast_start_day, sea_name, plot_metric=False, plot_maps=False):
    forecast_start_day = datetime.strptime(forecast_start_day, '%Y%m%d')
    pre_history_size = 104
    forecast_size = 52

    model_name = f'models/{sea_name}_{pre_history_size}_{forecast_size}_(2l_52_3_3)(19790101-20200101).pt'

    features, _ = get_prehistory(forecast_start_day, sea_name, pre_history_size)
    features = resize(features, (features.shape[0], features.shape[1] // 2, features.shape[2] // 2))
    target, target_dates = get_target(forecast_start_day, sea_name, forecast_size)
    target_dates = [datetime.strptime(d, '%Y%m%d') for d in target_dates]

    forecaster_params = {
        "input_size": (target.shape[1] // 2, target.shape[2] // 2),
        "n_layers": 2,
        "in_time_points": pre_history_size,
        "out_time_points": forecast_size,
        "convolve_params": {"kernel_size": (52, 3, 3)},
        "transpose_convolve_params": {"kernel_size": (52, 3, 3)},
        "conv_dim": 3,
        "activation_function": nn.ReLU(inplace=True),
        "finish_activation_function": nn.ReLU(inplace=True),
    }

    model = ForecasterBase(**forecaster_params).to(device)
    model.load_state_dict(torch.load(model_name))
    model.to(device)
    print(model)

    features = np.expand_dims(features, axis=0)

    prediction = model(tensor(features).float().to(device)).detach().cpu().numpy()[0]
    prediction = resize(prediction, (prediction.shape[0], target.shape[1], target.shape[2]))
    prediction = fix_range(prediction)
    prediction = fix_border(sea_name, prediction)

    l1_list = []
    ssim_list = []
    acc_list = []
    for i in range(prediction.shape[0]):

        matrices_path = '/path_to_save/'
        if not os.path.exists(f'{matrices_path}/matrices/{sea_name}'):
            os.makedirs(f'{matrices_path}/matrices/{sea_name}')
        np.save(f'{matrices_path}/matrices/{sea_name}/{target_dates[i].strftime("%Y%m%d")}.npy', prediction[i])


        l1 = np.round(mae(prediction[i], target[i]), 4)
        l1_list.append(l1)
        ssim_metric = np.round(ssim(prediction[i], target[i], data_range=1), 4)
        ssim_list.append(ssim_metric)
        acc = np.round(binary_accuracy(prediction[i], target[i]), 4)
        acc_list.append(acc)

        title = (f'{full_name(sea_name)} - {target_dates[i].strftime("%Y/%m/%d")},\nMAE={l1}, SSIM={ssim_metric}, '
                 f'accuracy={acc}')
        if plot_maps:
            plot_comparison_map(prediction[i], target[i], sea_name, title, save=True)

    if plot_metric:
        plt.plot(target_dates, l1_list)
        plt.title('MAE')
        plt.show()

        plt.plot(target_dates, ssim_list)
        plt.title('SSIM')
        plt.show()

        plt.plot(target_dates, acc_list)
        plt.title('Accuracy, threshold=0.2')
        plt.show()

    return l1_list, ssim_list, acc_list, target_dates


sea_name = 'chukchi'

full_dates = []
full_l1 = []
full_ssim = []
full_acc = []
years_to_predict = ['20200101', '20210101', '20220101', '20230101']
for d in years_to_predict:
    print(d)
    l1, ssim_val, acc, dates = calculate_metrics(d, sea_name, plot_maps=True, plot_metric=True)
    full_dates.extend(dates)
    full_l1.extend(l1)
    full_ssim.extend(ssim_val)
    full_acc.extend(acc)

df = pd.DataFrame()
df['dates'] = full_dates
df['l1'] = full_l1
df['ssim'] = full_ssim
df['accuracy'] = full_acc
df.to_csv(f'results/{sea_name}_metrics({years_to_predict[0]}-{years_to_predict[- 1]}).csv', index=False)
