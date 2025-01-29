import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

timesformer_path = 'TimeSformer/matrices'
cnn_3d_path = '3d_conv/matrices'
cnn_2d_path = '2d_conv/matrices'
osisaf_path = 'path_to_files'

sea_name = 'kara'

point_inds = (70, 100)


#### 3D CONV TS EXTRACTION
cnn_3d_dates = []
cnn_3d_values = []
for file in os.listdir(f'{cnn_3d_path}/{sea_name}'):
    date = datetime.strptime(file, '%Y%m%d.npy')
    cnn_3d_dates.append(date)
    matrix = np.load(f'{cnn_3d_path}/{sea_name}/{file}')
    '''plt.imshow(matrix)
        plt.scatter(*point_inds, c='r')
        plt.show()'''
    val = matrix[point_inds[1], point_inds[0]]
    cnn_3d_values.append(val)
print('3D loaded')

#### 2D CONV TS EXTRACTION
cnn_2d_dates = []
cnn_2d_values = []
for file in os.listdir(f'{cnn_2d_path}/{sea_name}'):
    date = datetime.strptime(file, '%Y%m%d.npy')
    cnn_2d_dates.append(date)
    matrix = np.load(f'{cnn_2d_path}/{sea_name}/{file}')
    '''plt.imshow(matrix)
        plt.scatter(*point_inds, c='r')
        plt.show()'''
    val = matrix[point_inds[1], point_inds[0]]
    cnn_2d_values.append(val)
print('2D loaded')

#### TIMESFORMER CONV TS EXTRACTION
timesformer_dates = []
timesformer_values = []
for file in os.listdir(f'{timesformer_path}/{sea_name}'):
    date = datetime.strptime(file, '%Y%m%d.npy')
    timesformer_dates.append(date)
    matrix = np.load(f'{timesformer_path}/{sea_name}/{file}')
    '''plt.imshow(matrix)
        plt.scatter(*point_inds, c='r')
        plt.show()'''
    val = matrix[point_inds[1], point_inds[0]]
    timesformer_values.append(val)
print('timesformer load')

### REAL DATA LOAD
osisaf_values = []
for date in cnn_2d_dates:
    matrix = np.load(f'{osisaf_path}/{sea_name}/{date.strftime("osi_%Y%m%d.npy")}')
    val = matrix[point_inds[1], point_inds[0]]
    osisaf_values.append(val)


plt.rcParams['figure.figsize'] = (12, 3)

plt.plot(timesformer_dates, timesformer_values, label='TimeSformer', c='black')
plt.plot(cnn_2d_dates, cnn_2d_values, label='2D Conv based', c='gray')
plt.plot(cnn_3d_dates, cnn_3d_values, label='3D Conv based', c='gray', linestyle='--')
plt.plot(cnn_2d_dates, osisaf_values, label='Ground truth', c='green')

plt.vlines(datetime(2021, 7, 15), -0.05, 1.05, colors='r', linewidth=2)

plt.legend()
plt.title('Ice concentration time series in point lat=79, lon=76')
plt.ylabel('Ice concentration')
plt.tight_layout()
plt.show()