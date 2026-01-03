from timm.scheduler.cosine_lr import CosineLRScheduler
from dataloader import create_dataloaders
import os
from TimeSformer.vit import TimeSformer
import torch
from tqdm import tqdm
import numpy as np
from skimage.transform import resize


# Script for training TimeSformer for sea ice prediction task with weights saving


def count_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.

        Args:
            model: The PyTorch model to analyze.

        Returns:
            int: The total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
accumulation_steps = 4
lr_max = 0.0005
lr_min = 0.000001
epochs = 140
predict_period = 52  # need for dataloader
in_period = 104  # need for dataloader
batch_size = 2  # need for dataloader
num_heads = 12
emb_mult = 5  # Necessary to control the internal size of the embedding to manage the required model size.
place = "laptev"  # Name of folder with sea
PARALLEL = True
from_ymd_train = [1979, 1, 1]  # Dates
to_ymd_train = [2012, 1, 1]
from_ymd_test = [2012, 1, 2]
to_ymd_test = [2020, 1, 1]
load_predtrain = False  # If you need to continue train or start inference
depth = 11
LOSS = "MAE"
predtreain_path = (
    "model_weights_in_per_104_pred_per_52_bs_1__dates_1979to_2012_stride7_sigmoid_ep80"
)

# [2012,1,1] to [2012,1,1] [2020,1,1]
stride = 7
resize_img = [72, 60]
if resize_img is not None:
    resize_img = check_size_for_reize(resize_img, num_heads=num_heads)
    mask = np.load(rf"coastline_masks/{place}_mask.npy")
    mask = resize(mask, (resize_img[0], resize_img[1]), anti_aliasing=False)
else:
    mask = np.load(rf"coastline_masks/{place}_mask.npy")
dataloader_train, img_sizes = create_dataloaders(
    path_to_dir=f"{place}",
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
    path_to_dir=f"{place}",
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
    patch_size = [patch_size1, patch_size2]  # int(img_sizes[1]/(img_sizes[1]*2)**0.5)
else:
    patch_size = int(img_sizes[0] / (img_sizes[0] * 2) ** 0.5)
embed_dim = embed_dim_by_img(img_sizes[1], num_heads, emb_mult)
dropout = 0.1
attn_drop_rate = 0.1
# Loss f-n
####################
NAME = f"Sea_{place}_pred{load_predtrain}_LOSS_{LOSS}dropout{dropout}_depth_{depth}attn_drop_rate{attn_drop_rate}_num_heads_{num_heads}_emb_dim_{embed_dim}"  # last num_heads=6
####################

if LOSS == "MAE":
    loss_l1 = torch.nn.L1Loss()


def loss_fn(x, y):
    """
    Computes the L1 loss between two tensors.

        Args:
            x: The first tensor.
            y: The second tensor.

        Returns:
            float: The L1 loss value.
    """
    out = loss_l1(x, y)
    return out


# Model
# output_size is needed to specify the size of the picture to input
# (even if the sides are different). img_size It is necessary to specify the larger side or any side.
#  embed_dim can be set at your discretion, but it is better to define it through the functions above.
# Here it is very important to choose the size of the picture, emb_mult and num_heads so that the internal dimensionality converges on the division into heads.

if PARALLEL:
    model = TimeSformer(
        batch_size=batch_size,
        output_size=[img_sizes[0], img_sizes[1]],
        img_size=img_sizes[0],
        embed_dim=embed_dim,
        num_frames=4,
        attention_type="divided_space_time",
        pretrained_model=False,
        in_chans=1,
        out_chans=predict_period,
        patch_size=patch_size,
        num_heads=num_heads,
        in_periods=in_period,
        place=place,
        depth=depth,
        emb_mult=emb_mult,
        attn_drop_rate=attn_drop_rate,
        dropout=dropout,
    )
    if load_predtrain:
        model_dict_pred_train = torch.load(
            predtreain_path, map_location=torch.device("cpu")
        )
        model_dict = model.state_dict()
        dict_matched = [
            i
            for i, k in zip(model_dict_pred_train, model_dict)
            if model_dict_pred_train[i].shape == model_dict[k].shape
        ]
        test_dict = {i: model_dict_pred_train[i] for i in dict_matched}
        model_dict.update(test_dict)
        model.load_state_dict(model_dict)
        pretrained_dict = {
            k: v for k, v in model_dict_pred_train.items() if k in model_dict
        }

    model = torch.nn.DataParallel(model)
    model.to(device)
else:
    model = TimeSformer(
        batch_size=batch_size,
        output_size=[img_sizes[0], img_sizes[1]],
        img_size=img_sizes[0],
        embed_dim=embed_dim,
        num_frames=4,
        attention_type="divided_space_time",
        pretrained_model=False,
        in_chans=1,
        out_chans=predict_period,
        patch_size=patch_size,
        num_heads=num_heads,
        in_periods=in_period,
        place=place,
        depth=depth,
        emb_mult=emb_mult,
        attn_drop_rate=attn_drop_rate,
        dropout=dropout,
    ).to(device)
    if load_predtrain:
        model_dict_pred_train = torch.load(
            predtreain_path, map_location=torch.device(device)
        )
        model_dict = model.state_dict()
        dict_matched = [
            i
            for i, k in zip(model_dict_pred_train, model_dict)
            if model_dict_pred_train[i].shape == model_dict[k].shape
        ]
        test_dict = {i: model_dict_pred_train[i] for i in dict_matched}
        model_dict.update(test_dict)
        model.load_state_dict(model_dict)

weight_decay = 0.001
optimizer = torch.optim.AdamW(
    model.parameters(), lr=lr_max, betas=(0.9, 0.98), eps=1e-9
)  # Weight decay

optimizer_sch = CosineLRScheduler(
    optimizer,
    t_initial=120,
    lr_min=lr_min * 10,
    warmup_t=5,
    cycle_limit=1.0,
    warmup_lr_init=lr_min,
    warmup_prefix=False,
    t_in_epochs=True,
    noise_range_t=None,
    noise_pct=0.67,
    noise_std=1.0,
    noise_seed=42,
    initialize=True,
)

img = 0
step = 0
ep = 0
test_step = 0
current_step = 0
if not os.path.isdir(f"{NAME}"):
    os.mkdir(f"{NAME}")
for epoch in tqdm(range(epochs)):
    ep += 1

    model.train()
    # TRAIN
    for i, batch in enumerate(dataloader_train):
        print(i)
        X, y, x_d, y_d = batch
        step += 1
        # current_step +=1
        X = X.to(device)
        y = y.squeeze(1).to(device)
        outputs = model(X)

        loss = loss_fn(outputs, y) / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    optimizer_sch.step(ep)
    optimizer.zero_grad()
    torch.cuda.empty_cache()

    # TEST
    model.eval()
    with torch.no_grad():
        for X, y, x_d, y_d in dataloader_test:
            test_step += 1
            # current_step +=1
            X = X.to(device)
            y = y.squeeze(1).to(device)
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss_masked = loss_fn(
                outputs * torch.tensor(np.float32(mask)).to(device), y
            )
            imgs = np.absolute(
                outputs.detach().cpu().numpy() - y.detach().cpu().numpy()
            )
        torch.cuda.empty_cache()
    if ep % 30 == 0:
        if PARALLEL:
            torch.save(
                model.module.state_dict(),
                f"{NAME}/Module_model_weights_in_per_{in_period}_pred_per_{predict_period}_bs_{batch_size}__dates_{from_ymd_train[0]}to_{to_ymd_train[0]}_stride{stride}_sigmoid",
            )
        else:
            torch.save(
                model.state_dict(),
                f"{NAME}/model_weights_in_per_{in_period}_pred_per_{predict_period}_bs_{batch_size}__dates_{from_ymd_train[0]}to_{to_ymd_train[0]}_stride{stride}_sigmoid",
            )

if PARALLEL:
    torch.save(
        model.module.state_dict(),
        f"{NAME}/Module_model_weights_in_per_{in_period}_pred_per_{predict_period}_bs_{batch_size}__dates_{from_ymd_train[0]}to_{to_ymd_train[0]}_stride{stride}_sigmoid",
    )
else:
    torch.save(
        model.state_dict(),
        f"{NAME}/model_weights_in_per_{in_period}_pred_per_{predict_period}_bs_{batch_size}__dates_{from_ymd_train[0]}to_{to_ymd_train[0]}_stride{stride}_sigmoid",
    )
