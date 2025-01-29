from utils import *
from configs_7days import get_args
from functions import inference
from read_npy import create_dataloaders
from skimage.transform import resize
from torch.utils.tensorboard import SummaryWriter


def check_size_for_reize(img_size, num_heads):
    if img_size[0] % num_heads != 0:
        img_size_new = [(int(img_size[0] / num_heads) + 1) * num_heads, img_size[1]]
        print("New image size is", img_size_new)
        return img_size_new
    else:
        return img_size


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def embed_dim_by_img(img, num_heads, emb_mult):
    emb_dim = img * emb_mult
    head_det = emb_dim % num_heads
    if head_det != 0:
        emb_dim = emb_dim - head_det + num_heads
    return emb_dim


def count_patch_size(imgsize):
    patch = imgsize ** 0.5
    if imgsize % patch == 0:

        return patch
    else:
        while imgsize % patch != 0:
            patch = int(patch) - 1
    return patch


# Loss f-n
loss_l1 = torch.nn.L1Loss()

place = 'kara'
predict_period = 52
in_period = 52
accumulation_steps = 12
writer = SummaryWriter(f'writer_test/lstm_{in_period}_{predict_period}_predict_7day_52__to52')


def setup(args, predict_period,
          in_period, place):
    path_to_mask = r'D:\Projects\test_cond\AAAI_code\Ice\coastline_masks'
    path_to_sea = r'D:\Projects\test_cond\AAAI_code\Ice'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    lr_max = 0.0005
    lr_min = 0.00001
    epochs = 90
    batch_size = 1
    num_heads = 8
    emb_mult = 4
    place = place
    from_ymd_train = [1979, 1, 1]
    to_ymd_train = [2012, 1, 1]
    from_ymd_test = [2018, 1, 1]  # [2018,12,23]#
    to_ymd_test = [2025, 1, 1]
    stride = 7

    mask = np.load(fr'{path_to_mask}\{place}_mask.npy')
    resize_img = [48, 48]
    if resize_img is not None:
        resize_img = check_size_for_reize(resize_img, num_heads=num_heads)
        mask = np.load(fr'{path_to_mask}\{place}_mask.npy')
        mask = resize(mask, (resize_img[0], resize_img[1]), anti_aliasing=False)
    dataloader_test, img_sizes = create_dataloaders(path_to_dir=fr'D:\Projects\test_cond\AAAI_code\Ice/{place}',
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
                                                    shift=predict_period)
    test_len = dataloader_test.__len__()

    if img_sizes[1] > img_sizes[0]:
        img_sizes = (img_sizes[1], img_sizes[0])
    if img_sizes[1] != img_sizes[0]:
        patch_size1 = count_patch_size(img_sizes[0])
        patch_size2 = count_patch_size(img_sizes[1])
        patch_size = [patch_size1, patch_size2]
    else:
        patch_size = int(img_sizes[0] / (img_sizes[0] * 2) ** 0.5)

    embed_dim = embed_dim_by_img(img_sizes[1], num_heads, emb_mult)

    if args.model == 'SwinLSTM-D':
        from SwinLSTM_D import SwinLSTM
        model = SwinLSTM(img_size=args.input_img_size, patch_size=args.patch_size,
                         in_chans=args.input_channels, embed_dim=embed_dim,
                         depths_downsample=args.depths_down, depths_upsample=args.depths_up,
                         num_heads=args.heads_number, window_size=args.window_size).to(args.device)
        model.load_state_dict(
            torch.load(r'D:\Projects\SwinLSTM\kara_12_6_7days\trained_model_state_dict_mse_43.404335021972656_8_49_13'))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.L1Loss()
    return model, criterion, optimizer, dataloader_test, mask


def main(writer, accumulation_steps, place):
    args = get_args()
    set_seed(args.seed)
    cache_dir, model_dir, log_dir = make_dir(args)
    logger = init_logger(log_dir)

    model, criterion, optimizer, valid_loader, mask = setup(args,
                                                            predict_period,
                                                            in_period, place)

    train_losses, valid_losses = [], []

    best_metric = (0, float('inf'), float('inf'))

    for epoch in range(1):
        start_time = time.time()
        valid_loss, mse, ssim = inference(args, logger, epoch, model, valid_loader, criterion, cache_dir, writer, place,
                                          mask)
        valid_losses.append(valid_loss)
        plot_loss(valid_losses, 'valid', epoch, args.res_dir, args.epoch_valid)
        logger.info(f'[Current Best] EP:{best_metric[0]:04d} MSE:{best_metric[1]:.4f} SSIM:{best_metric[2]:.4f}')

        print(f'Time usage per epoch: {time.time() - start_time:.0f}s')


if __name__ == '__main__':
    main(writer=writer, accumulation_steps=accumulation_steps, place=place)