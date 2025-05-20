from utils import *
import torch.nn as nn
from configs_7days import get_args
from functions import train, test
from read_npy import create_dataloaders
from skimage.transform import resize
from torch.utils.tensorboard import SummaryWriter


def check_size_for_reize(img_size, num_heads):
    """
    Checks if the image size is divisible by the number of heads and adjusts it if necessary.

        Args:
            img_size: The original image size as a tuple (width, height).
            num_heads: The number of attention heads.

        Returns:
            tuple: The adjusted or original image size as a tuple (width, height).
    """
    if img_size[0] % num_heads != 0:
        img_size_new = [(int(img_size[0] / num_heads) + 1) * num_heads, img_size[1]]
        print("New image size is", img_size_new)
        return img_size_new
    else:
        return img_size


def count_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.

        Args:
            model: The PyTorch model to analyze.

        Returns:
            int: The total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def embed_dim_by_img(img, num_heads, emb_mult):
    """
    Calculates the embedding dimension based on image size and other factors.

        Adjusts the embedding dimension to be divisible by the number of heads.

        Args:
            img: The size of the input image.
            num_heads: The number of attention heads.
            emb_mult: A multiplier for the embedding dimension.

        Returns:
            int: The calculated embedding dimension, adjusted to be divisible by num_heads.
    """
    emb_dim = img * emb_mult
    head_det = emb_dim % num_heads
    if head_det != 0:
        emb_dim = emb_dim - head_det + num_heads
    return emb_dim


def count_patch_size(imgsize):
    """
    Calculates the optimal patch size for an image given its total size.

        The method determines the largest integer that evenly divides the image size,
        suitable for creating patches without remainder. It starts with the square root
        of the image size and iteratively decreases until a divisor is found.

        Args:
            imgsize: The total size of the image (e.g., width * height).

        Returns:
            int: The calculated patch size, which is an integer representing the side length
                 of square patches that can perfectly tile the image.
    """
    patch = imgsize**0.5
    if imgsize % patch == 0:

        return patch
    else:
        while imgsize % patch != 0:
            patch = int(patch) - 1
    return patch


# Loss f-n
loss_l1 = torch.nn.L1Loss()

accumulation_steps = 12
writer = SummaryWriter(f"writer/lstm_104_52_predtrain")


def setup(args):
    """
    Sets up the model, data loaders, optimizer, and criterion.

        Args:
            args:  Arguments object containing configurations for training and model
                   parameters (e.g., learning rate, patch size, input image size).

        Returns:
            tuple: A tuple containing the trained model, loss criterion, optimizer,
                   training data loader, and testing data loader.
    """
    path_to_mask = r"D:\Projects\test_cond\AAAI_code\Ice\coastline_masks"
    path_to_sea = r"D:\Projects\test_cond\AAAI_code\Ice"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lr_max = 0.0005
    lr_min = 0.00001
    epochs = 90
    predict_period = 52
    in_period = 86
    batch_size = 1
    num_heads = 8
    emb_mult = 4
    place = "kara"
    from_ymd_train = [1979, 1, 1]
    to_ymd_train = [2012, 1, 1]
    from_ymd_test = [2012, 1, 2]
    to_ymd_test = [2020, 1, 1]
    stride = 7
    mask = np.load(rf"{path_to_mask}\{place}_mask.npy")
    resize_img = [64, 64]
    if resize_img is not None:
        resize_img = check_size_for_reize(resize_img, num_heads=num_heads)
        mask = np.load(rf"{path_to_mask}\{place}_mask.npy")
        mask = resize(mask, (resize_img[0], resize_img[1]), anti_aliasing=False)
    dataloader_train, img_sizes = create_dataloaders(
        path_to_dir=f"{path_to_sea}/{place}",
        batch_size=batch_size,
        in_period=in_period,
        predict_period=predict_period,
        stride=stride,
        test_end=None,
        from_ymd=from_ymd_train,
        to_ymd=to_ymd_train,
        pad=False,
        train_test_split=None,
        resize_img=resize_img,
    )
    dataloader_test, img_sizes = create_dataloaders(
        path_to_dir=f"{path_to_sea}/{place}",
        batch_size=1,
        in_period=in_period,
        predict_period=predict_period,
        stride=stride,
        test_end=None,
        from_ymd=from_ymd_test,
        to_ymd=to_ymd_test,
        pad=False,
        train_test_split=None,
        resize_img=resize_img,
    )
    train_len = dataloader_train.__len__()
    test_len = dataloader_test.__len__()

    if img_sizes[1] > img_sizes[0]:
        img_sizes = (img_sizes[1], img_sizes[0])
    if img_sizes[1] != img_sizes[0]:
        patch_size1 = count_patch_size(
            img_sizes[0]
        )  # int(img_sizes[0]/(img_sizes[0]*2)**0.5)
        patch_size2 = count_patch_size(img_sizes[1])
        patch_size = [
            patch_size1,
            patch_size2,
        ]  # int(img_sizes[1]/(img_sizes[1]*2)**0.5)
    else:
        patch_size = int(img_sizes[0] / (img_sizes[0] * 2) ** 0.5)
    embed_dim = 128

    if args.model == "SwinLSTM-D":
        from SwinLSTM_D import SwinLSTM

        model = SwinLSTM(
            img_size=args.input_img_size,
            patch_size=args.patch_size,
            in_chans=args.input_channels,
            embed_dim=embed_dim,
            depths_downsample=args.depths_down,
            depths_upsample=args.depths_up,
            num_heads=args.heads_number,
            window_size=args.window_size,
        ).to(args.device)
        model.load_state_dict(
            torch.load(
                r"results_12_6\model\trained_model_state_dict_mse_80.86911010742188"
            )
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.MSELoss()

    return model, criterion, optimizer, dataloader_train, dataloader_test


def main(writer, accumulation_steps):
    """
    Trains a model and saves the best performing state dictionary.

        Args:
            writer: The writer object for logging training information.
            accumulation_steps: The number of steps to accumulate gradients before updating weights.

        Returns:
            None
    """
    args = get_args()
    set_seed(args.seed)
    cache_dir, model_dir, log_dir = make_dir(args)
    logger = init_logger(log_dir)

    model, criterion, optimizer, train_loader, valid_loader = setup(args)

    train_losses, valid_losses = [], []

    best_metric = (0, float("inf"), float("inf"))

    for epoch in range(args.epochs):

        start_time = time.time()
        train_loss = train(
            args,
            logger,
            epoch,
            model,
            train_loader,
            criterion,
            optimizer,
            writer,
            accumulation_steps=accumulation_steps,
        )
        train_losses.append(train_loss)
        plot_loss(train_losses, "train", epoch, args.res_dir, 1)

        if (epoch + 1) % args.epoch_valid == 0:

            valid_loss, mse, ssim = test(
                args, logger, epoch, model, valid_loader, criterion, cache_dir, writer
            )

            valid_losses.append(valid_loss)

            plot_loss(valid_losses, "valid", epoch, args.res_dir, args.epoch_valid)

            if mse < best_metric[1]:
                torch.save(
                    model.state_dict(),
                    f"{model_dir}/trained_model_state_dict_mse_{mse}",
                )
                best_metric = (epoch, mse, ssim)

            logger.info(
                f"[Current Best] EP:{best_metric[0]:04d} MSE:{best_metric[1]:.4f} SSIM:{best_metric[2]:.4f}"
            )

        print(f"Time usage per epoch: {time.time() - start_time:.0f}s")


if __name__ == "__main__":
    main(writer=writer, accumulation_steps=accumulation_steps)
