import os
from datetime import datetime
import numpy as np
import pandas as pd

from cnn_forecaster_2d.visualizator import get_extent


# Script for calculating binary accuracy for each model
# and IceNet downloaded prediction
def accuracy(prediction, target):
    """
    Calculates the accuracy of a prediction against a target.

        Args:
            prediction: The predicted values.
            target: The true target values.

        Returns:
            float: The accuracy score, representing the proportion of correct predictions.
    """
    diff = target - prediction
    errors_num = len(np.where(diff == 0)[0])
    acc = errors_num / prediction.size
    return acc


def get_mean_real(year, month, sea_name):
    """
    Calculates the mean real value for a given year, month, and sea name.

        Args:
            year: The year to calculate the mean for.
            month: The month to calculate the mean for.
            sea_name: The name of the sea region.

        Returns:
            numpy.ndarray: A NumPy array representing the mean real values
                           with values binarized (0 or 1) based on a threshold of 0.2.
    """
    osi_path = f"/path_to_data//OSISAF/{sea_name}"
    dates = pd.date_range(datetime(year, month, 1), freq="d", periods=30)
    dates = [d.strftime("osi_%Y%m%d.npy") for d in dates]
    month_arr = []
    for date in dates:
        month_arr.append(np.load(f"{osi_path}/{date}"))
    month_arr = np.array(month_arr)
    month_arr = np.mean(month_arr, axis=0)
    month_arr[month_arr >= 0.2] = 1
    month_arr[month_arr < 0.2] = 0
    return month_arr


def get_icenet_error():
    """
    Calculates the accuracy of icenet matrices against real data for different seas.

        This method iterates through files in a specified directory, loads icenet matrices,
        calculates their accuracy compared to corresponding real data using helper functions,
        and stores the results in a Pandas DataFrame.

        Args:
            None

        Returns:
            pd.DataFrame: A DataFrame containing the accuracy scores for each sea
                          ('kara', 'barents', 'chukchi', 'eastsib', 'laptev') and their corresponding dates.
    """
    icenet_path = "D:/ice_sources/icenet/matrices/20200101"
    df = pd.DataFrame()
    for sea_name in ["kara", "barents", "chukchi", "eastsib", "laptev"]:
        icenet_acc = []
        icenet_dates = []
        for file in os.listdir(icenet_path):
            date = datetime.strptime(file, "icenet_%Y%m%d.npy")
            icenet_dates.append(date)
            matrix = np.load(f"{icenet_path}/{file}")
            matrix = get_extent(matrix, sea_name)
            real_matrix = get_mean_real(date.year, date.month, sea_name)
            acc = accuracy(matrix, real_matrix)
            icenet_acc.append(acc)
        df[sea_name] = icenet_acc
        df["dates"] = icenet_dates
    df


def get_2d_error():
    """
    Calculates and returns a DataFrame containing the monthly mean accuracy for different seas.

        This method reads CSV files containing metrics for each sea (kara, barents, chukchi, eastsib, laptev),
        filters data before July 1st, 2020, resamples to monthly frequency, and calculates the mean accuracy.
        It then combines these accuracies into a single DataFrame with seas as columns and dates as index.

        Returns:
            pd.DataFrame: A DataFrame where each column represents a sea's accuracy and the index is the date.
    """
    sea_df = pd.DataFrame()
    for sea_name in ["kara", "barents", "chukchi", "eastsib", "laptev"]:
        file_path = (
            f"../cnn_forecaster_2d/results/{sea_name}_metrics(20200101-20230101).csv"
        )
        df = pd.read_csv(file_path)
        df["dates"] = pd.to_datetime(df["dates"])
        df = df[df["dates"] < datetime(2020, 7, 1)]
        df = df.set_index(df["dates"])
        df = df.resample("1M").mean()
        sea_df[sea_name] = df["accuracy"]
        sea_df["dates"] = df["dates"]
    sea_df


def get_3d_error():
    """
    Calculates and returns the monthly mean accuracy for each sea region.

        This method reads CSV files containing metrics for different seas, filters
        the data to include only dates before July 1st, 2020, resamples the data
        to a monthly frequency, and calculates the mean accuracy for each sea.
        The results are then combined into a single DataFrame.

        Args:
            None

        Returns:
            pd.DataFrame: A DataFrame containing the monthly mean accuracy for each sea region ('kara', 'barents', 'chukchi', 'eastsib', 'laptev') and dates as index.
    """
    sea_df = pd.DataFrame()
    for sea_name in ["kara", "barents", "chukchi", "eastsib", "laptev"]:
        file_path = (
            f"../cnn_forecaster_3d/results/{sea_name}_metrics(20200101-20230101).csv"
        )
        df = pd.read_csv(file_path)
        df["dates"] = pd.to_datetime(df["dates"])
        df = df[df["dates"] < datetime(2020, 7, 1)]
        df = df.set_index(df["dates"])
        df = df.resample("1M").mean()
        sea_df[sea_name] = df["accuracy"]
        sea_df["dates"] = df["dates"]
    sea_df


def get_timesformer_error():
    """
    Calculates and returns the monthly mean accuracy for TimeFormer across several seas.

        This method reads CSV files containing TimeFormer metrics for different seas,
        filters data before July 1st, 2020, resamples to a monthly frequency,
        and combines the results into a single DataFrame.

        Args:
            None

        Returns:
            pd.DataFrame: A DataFrame with monthly mean accuracy values for each sea
                          ('kara', 'barents', 'chukchi', 'eastsib', 'laptev') and dates as index.
    """
    sea_df = pd.DataFrame()
    for sea_name in ["kara", "barents", "chukchi", "eastsib", "laptev"]:
        file_path = f"../transformer/results/{sea_name}_metrics(20200101-20230101).csv"
        df = pd.read_csv(file_path)
        df["dates"] = pd.to_datetime(df["dates"])
        df = df[df["dates"] < datetime(2020, 7, 1)]
        df = df.set_index(df["dates"])
        df = df.resample("1M").mean()
        sea_df[sea_name] = df["accuracy"]
        sea_df["dates"] = df["dates"]
    sea_df


get_timesformer_error()
