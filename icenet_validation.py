import os
from datetime import datetime
import numpy as np
import pandas as pd

from cnn_forecaster_2d.visualizator import get_extent


# Script for calculating binary accuracy for each model
# and IceNet downloaded prediction
def accuracy(prediction, target):
    diff = target - prediction
    errors_num = len(np.where(diff == 0)[0])
    acc = errors_num / prediction.size
    return acc


def get_mean_real(year, month, sea_name):
    osi_path = f'/path_to_data//OSISAF/{sea_name}'
    dates = pd.date_range(datetime(year, month, 1), freq='d', periods=30)
    dates = [d.strftime('osi_%Y%m%d.npy') for d in dates]
    month_arr = []
    for date in dates:
        month_arr.append(np.load(f'{osi_path}/{date}'))
    month_arr = np.array(month_arr)
    month_arr = np.mean(month_arr, axis=0)
    month_arr[month_arr >= 0.2] = 1
    month_arr[month_arr < 0.2] = 0
    return month_arr


def get_icenet_error():
    icenet_path = 'D:/ice_sources/icenet/matrices/20200101'
    df = pd.DataFrame()
    for sea_name in ['kara', 'barents', 'chukchi', 'eastsib', 'laptev']:
        icenet_acc = []
        icenet_dates = []
        for file in os.listdir(icenet_path):
            date = datetime.strptime(file, 'icenet_%Y%m%d.npy')
            icenet_dates.append(date)
            matrix = np.load(f'{icenet_path}/{file}')
            matrix = get_extent(matrix, sea_name)
            real_matrix = get_mean_real(date.year, date.month, sea_name)
            acc = accuracy(matrix, real_matrix)
            icenet_acc.append(acc)
        df[sea_name] = icenet_acc
        df['dates'] = icenet_dates
    df


def get_2d_error():
    sea_df = pd.DataFrame()
    for sea_name in ['kara', 'barents', 'chukchi', 'eastsib', 'laptev']:
        file_path = f'../cnn_forecaster_2d/results/{sea_name}_metrics(20200101-20230101).csv'
        df = pd.read_csv(file_path)
        df['dates'] = pd.to_datetime(df['dates'])
        df = df[df['dates'] < datetime(2020, 7, 1)]
        df = df.set_index(df['dates'])
        df = df.resample('1M').mean()
        sea_df[sea_name] = df['accuracy']
        sea_df['dates'] = df['dates']
    sea_df


def get_3d_error():
    sea_df = pd.DataFrame()
    for sea_name in ['kara', 'barents', 'chukchi', 'eastsib', 'laptev']:
        file_path = f'../cnn_forecaster_3d/results/{sea_name}_metrics(20200101-20230101).csv'
        df = pd.read_csv(file_path)
        df['dates'] = pd.to_datetime(df['dates'])
        df = df[df['dates'] < datetime(2020, 7, 1)]
        df = df.set_index(df['dates'])
        df = df.resample('1M').mean()
        sea_df[sea_name] = df['accuracy']
        sea_df['dates'] = df['dates']
    sea_df


def get_timesformer_error():
    sea_df = pd.DataFrame()
    for sea_name in ['kara', 'barents', 'chukchi', 'eastsib', 'laptev']:
        file_path = f'../transformer/results/{sea_name}_metrics(20200101-20230101).csv'
        df = pd.read_csv(file_path)
        df['dates'] = pd.to_datetime(df['dates'])
        df = df[df['dates'] < datetime(2020, 7, 1)]
        df = df.set_index(df['dates'])
        df = df.resample('1M').mean()
        sea_df[sea_name] = df['accuracy']
        sea_df['dates'] = df['dates']
    sea_df


get_timesformer_error()
