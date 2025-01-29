import os

import pandas as pd
from scipy.stats import mannwhitneyu

# Script calculates pairwise mannwhitneyu test to estimate difference 
# between errors for each prediction step for models 

cnn_anime = pd.read_csv('../toy_example/anime_metrics_dist_2d_cnn.csv')
trans_anime = pd.read_csv('../.toy_example/anime_metrics_dist_timesformer.csv')

print('## Menhera experiment ## ')
print('mae:')
print(mannwhitneyu(cnn_anime['mae'], trans_anime['mae']))
print('ssim:')
print(mannwhitneyu(cnn_anime['ssim'], trans_anime['ssim']))


print('\n\n## SEAS5 comparison ## ')

seas5_mae_list = []
d2_mae_list = []
d3_mae_list = []
trans_mae_list = []
for file in os.listdir('seas5'):
    df = pd.read_csv(f'seas5/{file}', decimal=',')
    seas5_mae_list.extend(df['SEAS5_mae'].values)
    d2_mae_list.extend(df['2D Conv-based_mae'].values)
    d3_mae_list.extend(df['3D Conv-based_mae'].values)
    trans_mae_list.extend(df['TimeSformer_mae'].values)

print('\nseas5_mae_list, d2_mae_list')
print(mannwhitneyu(seas5_mae_list, d2_mae_list))
print('\nseas5_mae_list, d3_mae_list')
print(mannwhitneyu(seas5_mae_list, d3_mae_list))
print('\nseas5_mae_list, trans_mae_list')
print(mannwhitneyu(seas5_mae_list, trans_mae_list))
print('\nd3_mae_list, d2_mae_list')
print(mannwhitneyu(d3_mae_list, d2_mae_list))

seas5_ssim_list = []
d2_ssim_list = []
d3_ssim_list = []
trans_ssim_list = []
for file in os.listdir('seas5'):
    df = pd.read_csv(f'seas5/{file}', decimal=',')
    seas5_ssim_list.extend(df['SEAS5_ssim'].values)
    d2_ssim_list.extend(df['2D Conv-based_ssim'].values)
    d3_ssim_list.extend(df['3D Conv-based_ssim'].values)
    trans_ssim_list.extend(df['TimeSformer_ssim'].values)

print('\n\nseas5_ssim_list, d2_ssim_list')
print(mannwhitneyu(seas5_ssim_list, d2_ssim_list))
print('\nseas5_ssim_list, d3_ssim_list')
print(mannwhitneyu(seas5_ssim_list, d3_ssim_list))
print('\nseas5_ssim_list, trans_ssim_list')
print(mannwhitneyu(seas5_ssim_list, trans_ssim_list))
print('\nd3_ssim_list, d2_ssim_list')
print(mannwhitneyu(d3_ssim_list, d2_ssim_list))


print('\n\n## IceNet comparison ## ')
df = pd.read_csv('aaai_icenet_stattest.csv', decimal=',')
print('\nIceNet, 2D Conv-based')
print(mannwhitneyu(df['IceNet'], df['2D Conv-based']))
print('\nIceNet, 3D Conv-based')
print(mannwhitneyu(df['IceNet'], df['3D Conv-based']))
print('\nIceNet, TimeSformer ')
print(mannwhitneyu(df['IceNet'], df['TimeSformer ']))
print('\n2D Conv-based, 3D Conv-based')
print(mannwhitneyu(df['2D Conv-based'], df['3D Conv-based']))
