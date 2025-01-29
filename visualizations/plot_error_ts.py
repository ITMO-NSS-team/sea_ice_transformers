from datetime import datetime

import pandas as pd
import  matplotlib.pyplot as plt


# Visualization of error timeseries for different models


for sea_name in ['kara']:
    f_conv2d_df = pd.read_csv(f'../cnn_forecaster_2d/results/{sea_name}_metrics(20200101-20230101).csv')
    f_conv2d_df['dates'] = pd.to_datetime(f_conv2d_df['dates'])
    f_conv3d_df = pd.read_csv(f'../cnn_forecaster_3d/results/{sea_name}_metrics(20200101-20230101).csv')
    f_conv3d_df['dates'] = pd.to_datetime(f_conv3d_df['dates'])
    f_tran_df = pd.read_csv(f'../timesformer/results/{sea_name}_metrics(20200101-20230101).csv')
    f_tran_df['dates'] = pd.to_datetime(f_conv3d_df['dates'])

    full_dates = f_tran_df['dates']

    plt.rcParams['figure.figsize'] = (7.5, 7.5)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    ax1.plot(full_dates, f_conv2d_df['l1'], label='2D Conv-based model')
    ax1.plot(full_dates, f_conv3d_df['l1'], label='3D Conv-based model')
    ax1.plot(full_dates, f_tran_df['l1'], label='TimeSformer')

    ax1.set_ylabel('MAE value')
    lines = [datetime(2021, 12, 31),
             datetime(2020, 12, 31),
             datetime(2022, 12, 31)]
    ax1.vlines(lines, 0, 0.22, colors='black', linestyles='--', linewidth=1)
    ax1.set_title('MAE per image comparison')

    ax2.plot(full_dates, f_conv2d_df['ssim'], label='2D Conv-based model')
    ax2.plot(full_dates, f_conv3d_df['ssim'], label='3D Conv-based model')
    ax2.plot(full_dates, f_tran_df['ssim'], label='TimeSformer')
    ax2.set_ylabel('SSIM value')
    ax2.vlines(lines, 0.2, 0.81, colors='black', linestyles='--', linewidth=1)
    ax2.set_title('SSIM per image comparison')

    ax3.plot(full_dates, f_conv2d_df['accuracy'], label='2D Conv-based model')
    ax3.plot(full_dates, f_conv3d_df['accuracy'], label='3D Conv-based model')
    ax3.plot(full_dates, f_tran_df['accuracy'], label='TimeSformer')
    ax3.set_ylabel('Accuracy value')
    ax3.vlines(lines, 0.6, 1, colors='black', linestyles='--', linewidth=1)
    ax3.set_title('Accuracy (threshold=0.2) per image comparison')
    plt.tight_layout()
    ax3.legend()
    plt.savefig('metrics_ts.png', dpi=600)
    plt.show()

    