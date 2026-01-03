import os
import time
import torch
import random
import logging
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity

matplotlib.use("agg")


def set_seed(seed):
    """
    Sets the seed for PyTorch, random, and NumPy.

        This ensures reproducibility of results by setting the seed for these
        random number generators. It also sets PyTorch's cudnn backend to deterministic mode.

        Args:
            seed: The integer seed value.

        Returns:
            None
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def visualize(inputs, targets, outputs, epoch, idx, cache_dir):
    """
    Visualizes input, target, and output images for a given batch.

        Args:
            inputs: The input tensor.
            targets: The target tensor.
            outputs: The output tensor.
            epoch: The current epoch number.
            idx: The index of the current batch.
            cache_dir: The directory to save the visualization images.

        Returns:
            None
    """
    _, axarray = plt.subplots(3, targets.shape[1], figsize=(targets.shape[1] * 5, 10))

    for t in range(targets.shape[1]):
        axarray[0][t].imshow(inputs[0, t, 0].detach().cpu().numpy(), cmap="gray")
        axarray[1][t].imshow(targets[0, t, 0].detach().cpu().numpy(), cmap="gray")
        axarray[2][t].imshow(outputs[0, t, 0].detach().cpu().numpy(), cmap="gray")

    plt.savefig(os.path.join(cache_dir, "{:03d}-{:03d}.png".format(epoch, idx)))
    plt.close()


def plot_loss(loss_records, loss_type, epoch, plot_dir, step):
    """
    Plots the loss records and saves the plot to a file.

        Args:
            loss_records: The list of loss values to plot.
            loss_type: A string representing the type of loss being plotted (e.g., 'train', 'val').
            epoch: The current epoch number.
            plot_dir: The directory where the plot should be saved.
            step: The interval at which loss is recorded/plotted.

        Returns:
            None
    """
    plt.plot(range((epoch + 1) // step), loss_records, label=loss_type)
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "{}_loss_records.png".format(loss_type)))
    plt.close()


def MAE(pred, true):
    """
    Calculates the Mean Absolute Error (MAE).

        This method computes the average absolute difference between predicted and
        true values across all dimensions and then sums the result.

        Args:
            pred: Predicted values.
            true: True values.

        Returns:
            float: The MAE value, representing the average magnitude of errors.
    """
    return np.mean(np.abs(pred - true), axis=(0, 1)).sum()


def MSE(pred, true):
    """
    Calculates the Mean Squared Error between predicted and true values.

        Args:
            pred: The predicted values.
            true: The true values.

        Returns:
            float: The calculated Mean Squared Error.
    """
    return np.mean((pred - true) ** 2, axis=(0, 1)).sum()


# cite the 'PSNR' code from E3D-LSTM, Thanks!
# https://github.com/google/e3d_lstm/blob/master/src/trainer.py line 39-40
def PSNR(pred, true):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.

        Args:
            pred: The predicted image data.
            true: The ground truth image data.

        Returns:
            float: The PSNR value in decibels (dB).
    """
    mse = np.mean((np.uint8(pred * 255) - np.uint8(true * 255)) ** 2)
    return 20 * np.log10(255) - 10 * np.log10(mse)


def compute_metrics(predictions, targets):
    """
    Computes the Mean Squared Error (MSE) and Structural Similarity Index (SSIM) between predictions and targets.

        Args:
            predictions: The predicted values.
            targets: The ground truth values.

        Returns:
            tuple[float, float]: A tuple containing the MSE and SSIM scores.  The first element is the MSE,
                                 and the second element is the SSIM.
    """
    targets = targets.permute(0, 1, 3, 4, 2).detach().cpu().numpy()
    predictions = predictions.permute(0, 1, 3, 4, 2).detach().cpu().numpy()

    batch_size = predictions.shape[0]
    Seq_len = predictions.shape[1]

    ssim = 0

    for batch in range(batch_size):
        for frame in range(Seq_len):
            ssim += structural_similarity(
                targets[batch, frame].squeeze(), predictions[batch, frame].squeeze()
            )

    ssim /= batch_size * Seq_len

    mse = MSE(predictions, targets)

    return mse, ssim


def check_dir(path):
    """
    Checks if a directory exists and creates it if it doesn't.

        Args:
            path: The path to the directory to check/create.

        Returns:
            None
    """
    if not os.path.exists(path):
        os.makedirs(path)


def make_dir(args):
    """
    Creates necessary directories for resources.

        This method creates 'cache', 'model', and 'log' directories within the
        resource directory specified in the input arguments, ensuring they exist.

        Args:
            args: An object containing the resource directory path (res_dir).

        Returns:
            tuple: A tuple containing the paths to the cache, model, and log directories,
                   respectively.
    """
    cache_dir = os.path.join(args.res_dir, "cache")
    check_dir(cache_dir)

    model_dir = os.path.join(args.res_dir, "model")
    check_dir(model_dir)

    log_dir = os.path.join(args.res_dir, "log")
    check_dir(log_dir)

    return cache_dir, model_dir, log_dir


def init_logger(log_dir):
    """
    Initializes the logger with file and console handlers.

        Args:
            log_dir: The directory where log files will be stored.

        Returns:
            logging: The configured logging module.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        filename=os.path.join(log_dir, time.strftime("%Y_%m_%d") + ".log"),
        filemode="w",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    return logging
