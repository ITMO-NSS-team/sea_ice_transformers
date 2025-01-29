import pandas as pd

# This script provide averaging of metrics by quarters

l1_df = pd.DataFrame()
ssim_df = pd.DataFrame()
accuracy_df = pd.DataFrame()

for sea_name in ['kara', 'barents', 'laptev', 'eastsib', 'chukchi']:
    sea_df = pd.read_csv(f'results/{sea_name}_metrics(20200101-20230101).csv')
    l1_df[sea_name] = sea_df['l1']
    ssim_df[sea_name] = sea_df['ssim']
    accuracy_df[sea_name] = sea_df['accuracy']
l1_df['dates'] = pd.to_datetime(sea_df['dates'])
ssim_df['dates'] = pd.to_datetime(sea_df['dates'])
accuracy_df['dates'] = pd.to_datetime(sea_df['dates'])

l1_df = l1_df.resample('Q', on='dates').mean()
ssim_df = ssim_df.resample('Q', on='dates').mean()
accuracy_df = accuracy_df.resample('Q', on='dates').mean()

accuracy_df

