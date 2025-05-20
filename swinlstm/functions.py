import numpy as np
import torch
from torch.cuda import amp
from torch.cuda.amp import autocast as autocast
from utils import compute_metrics, visualize
from pytorch_msssim import SSIM
import pandas as pd

scaler = amp.GradScaler()


def model_forward_single_layer(model, inputs, targets_len, num_layers):
    """
    Performs a forward pass through a single layer of a recurrent model.

        This function iterates through the input sequence and then generates outputs
        for a specified number of target steps, updating the hidden state at each step.

        Args:
            model: The recurrent model to use for the forward pass.
            inputs: The input tensor.
            targets_len: The number of target time steps to generate outputs for.
            num_layers:  The number of layers in the model, used to initialize states.

        Returns:
            list: A list containing the output tensors from each time step
                  (input sequence + target sequence).
    """
    outputs = []
    states = [None] * len(num_layers)

    inputs_len = inputs.shape[1]

    last_input = inputs[:, -1]

    for i in range(inputs_len - 1):
        output, states = model(inputs[:, i], states)
        outputs.append(output)

    for i in range(targets_len):
        output, states = model(last_input, states)
        outputs.append(output)
        last_input = output

    return outputs


def model_forward_multi_layer(model, inputs, targets_len, num_layers):
    """
    Performs a forward pass through a multi-layer model.

        This function iterates through the input sequence and then generates outputs
        for a specified number of target steps, updating internal states at each step.

        Args:
            model: The model to use for the forward pass.
            inputs: The input tensor.
            targets_len: The number of target steps to generate.
            num_layers: A list indicating the number of layers in the model.  This is used to initialize state lists, but not directly used in computation.

        Returns:
            list: A list of output tensors from each forward pass step.
    """
    states_down = [None] * len(num_layers)
    states_up = [None] * len(num_layers)

    outputs = []

    inputs_len = inputs.shape[1]

    last_input = inputs[:, -1]

    for i in range(inputs_len - 1):
        output, states_down, states_up = model(inputs[:, i], states_down, states_up)
        outputs.append(output)

    for i in range(targets_len):
        output, states_down, states_up = model(last_input, states_down, states_up)
        outputs.append(output)
        last_input = output

    return outputs


def train(
    args,
    logger,
    epoch,
    model,
    train_loader,
    criterion,
    optimizer,
    writer,
    accumulation_steps=1,
):
    """
    Trains the model for a single epoch.

        Args:
            args: Arguments object containing training parameters.
            logger: Logger object for recording training progress.
            epoch: The current epoch number.
            model: The model to train.
            train_loader: DataLoader for the training dataset.
            criterion: Loss function.
            optimizer: Optimizer used for updating model weights.
            writer: TensorBoard writer for logging metrics.
            accumulation_steps: Number of batches to accumulate gradients before updating weights (default 1).

        Returns:
            float: The average loss over the entire training dataset for the epoch.
    """
    writer = writer
    model.train()
    num_batches = len(train_loader)
    losses = []

    for batch_idx, batch in enumerate(train_loader):
        inputs, targets, x_d, y_d = batch
        optimizer.zero_grad()
        inputs = inputs.permute(0, 2, 1, 3, 4)
        targets = targets.permute(0, 2, 1, 3, 4)

        inputs, targets = map(lambda x: x.float().to(args.device), [inputs, targets])
        targets_len = targets.shape[1]
        with autocast():
            if args.model == "SwinLSTM-B":
                outputs = model_forward_single_layer(
                    model, inputs, targets_len, args.depths
                )

            if args.model == "SwinLSTM-D":
                outputs = model_forward_multi_layer(
                    model, inputs, targets_len, args.depths_down
                )

            outputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()
            targets_ = torch.cat((inputs[:, 1:], targets), dim=1)
            loss = criterion(outputs, targets_) / accumulation_steps

        scaler.scale(loss).backward()
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        writer.add_scalar(
            "Loss_train",
            loss.item() * accumulation_steps,
            (epoch * train_loader.__len__()) + batch_idx,
        )
        losses.append(loss.item() * accumulation_steps)

        if batch_idx and batch_idx % args.log_train == 0:
            logger.info(
                f"EP:{epoch:04d} BI:{batch_idx:03d}/{num_batches:03d} Loss:{np.mean(losses):.6f}"
            )

    return np.mean(losses)


def test(args, logger, epoch, model, test_loader, criterion, cache_dir, writer):
    """
    Tests the model on the provided test loader.

        Args:
            args: Arguments object containing configuration parameters.
            logger: Logger for recording training information.
            epoch: The current epoch number.
            model: The PyTorch model to be tested.
            test_loader: DataLoader for the test dataset.
            criterion: Loss function used for evaluation.
            cache_dir: Directory for saving visualizations.
            writer: TensorBoard writer for logging metrics.

        Returns:
            tuple: A tuple containing the mean loss, MSE, and SSIM across all batches.
    """
    writer = writer
    model.eval()
    num_batches = len(test_loader)
    losses, mses, ssims = [], [], []

    for batch_idx, batch in enumerate(test_loader):
        inputs, targets, x_d, y_d = batch
        with torch.no_grad():
            inputs, targets = map(
                lambda x: x.float().to(args.device), [inputs, targets]
            )

            inputs = inputs.permute(0, 2, 1, 3, 4)
            targets = targets.permute(0, 2, 1, 3, 4)
            targets_len = targets.shape[1]
            if args.model == "SwinLSTM-B":
                outputs = model_forward_single_layer(
                    model, inputs, targets_len, args.depths
                )

            if args.model == "SwinLSTM-D":
                outputs = model_forward_multi_layer(
                    model, inputs, targets_len, args.depths_down
                )

            outputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()
            targets_ = torch.cat((inputs[:, 1:], targets), dim=1)

            losses.append(criterion(outputs, targets_).item())
            writer.add_scalar(
                "Loss_tesst", losses[-1], (epoch * test_loader.__len__()) + batch_idx
            )
            inputs_len = inputs.shape[1]
            outputs = outputs[:, inputs_len - 1 :]

            mse, ssim = compute_metrics(outputs, targets)
            writer.add_images(
                "Ground_truth",
                np.expand_dims(targets.squeeze(2).detach().cpu().numpy()[0], axis=1),
                0,
            )
            writer.add_images(
                "Predicts",
                np.expand_dims(outputs.squeeze(2).detach().cpu().numpy()[0], axis=1),
                0,
            )
            mses.append(mse)
            ssims.append(ssim)
            writer.add_scalar("mses", mse, (epoch * test_loader.__len__()) + batch_idx)
            writer.add_scalar(
                "ssims", ssim, (epoch * test_loader.__len__()) + batch_idx
            )
            if batch_idx and batch_idx % args.log_valid == 0:
                logger.info(
                    f"EP:{epoch:04d} BI:{batch_idx:03d}/{num_batches:03d} Loss:{np.mean(losses):.6f} MSE:{mse:.4f} SSIM:{ssim:.4f}"
                )
                visualize(inputs, targets, outputs, epoch, batch_idx, cache_dir)

    return np.mean(losses), np.mean(mses), np.mean(ssims)


def inference(
    args, logger, epoch, model, test_loader, criterion, cache_dir, writer, place, mask
):
    """
    Performs inference on the test dataset.

        Args:
            args: Arguments object containing configuration parameters.
            logger: Logger object for recording information.
            epoch: The current training epoch.
            model: The trained model to use for inference.
            test_loader: DataLoader for the test dataset.
            criterion: Loss function used for evaluation.
            cache_dir: Directory for caching visualization results.
            writer: TensorBoard writer for logging metrics.
            place: Identifier for the location or setup being tested.
            mask: Mask tensor used for masked loss calculation.

        Returns:
            tuple: A tuple containing the mean loss, mean MSE, and mean SSIM
                   across the test dataset.
    """
    l1 = torch.nn.L1Loss(reduction="none")
    writer = writer
    model.eval()
    num_batches = len(test_loader)
    losses, mses, ssims = [], [], []
    # tt = 0
    # ttt=1
    dates = []
    ssim_loses = []
    masked_mae_loses = []
    mae_loses = []
    tt = 1
    for batch_idx, batch in enumerate(test_loader):

        inputs, targets, x_d, y_d = batch
        print(y_d)
        [dates.append(time[0]) for time in y_d]
        with torch.no_grad():
            inputs, targets = map(
                lambda x: x.float().to(args.device), [inputs, targets]
            )

            inputs = inputs.permute(0, 2, 1, 3, 4)
            targets = targets.permute(0, 2, 1, 3, 4)
            targets_len = targets.shape[1]
            if args.model == "SwinLSTM-B":
                outputs = model_forward_single_layer(
                    model, inputs, targets_len, args.depths
                )

            if args.model == "SwinLSTM-D":
                outputs = model_forward_multi_layer(
                    model, inputs, targets_len, args.depths_down
                )
            # outputs=outputs[:,-targets_len:,:,:]
            outputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()
            targets_ = torch.cat((inputs[:, 1:], targets), dim=1)
            # loss = criterion(outputs, targets_)
            # losses.append(loss.item())
            # lossses = [[i,n.item()] for i,n in enumerate(loss.mean(dim=-1).mean(dim=-1)[0])]
            # [writer.add_scalar('Loss_test',n,(time+tt)) for time,n in lossses]

            # writer.add_scalar('Loss_tesst',losses[-1],(epoch*test_loader.__len__())+batch_idx)
            inputs_len = inputs.shape[1]
            outputs = outputs[:, inputs_len - 1 :]
            [
                np.save(
                    f"npy_results_12_6_30days_kara_stack/{place}/{place}_{y_d[i][0]}.npy",
                    arr,
                    allow_pickle=False,
                )
                for i, arr in enumerate(
                    np.squeeze(outputs.detach().cpu().numpy()[0], axis=1)
                )
            ]

            mse_l1 = (
                l1(outputs.permute(1, 0, 2, 3, 4), targets.permute(1, 0, 2, 3, 4))
                .mean(dim=-1)
                .mean(dim=-1)[:, 0, 0]
            )
            lossses = [[i, n.item()] for i, n in enumerate(mse_l1)]
            loss_masked = loss_fn(
                outputs * torch.tensor(np.float32(mask)).to("cuda"), targets
            )
            loss_maskeds = [
                [i, n.item()]
                for i, n in enumerate(loss_masked.mean(dim=-1).mean(dim=-1)[0])
            ]
            [masked_mae_loses.append(n) for time, n in loss_maskeds]

            [writer.add_scalar("Loss_test", n, (time + tt)) for time, n in lossses]
            mse, ssim = compute_metrics(outputs, targets)
            writer.add_images(
                "Ground_truth",
                np.expand_dims(targets.squeeze(2).detach().cpu().numpy()[0], axis=1),
                batch_idx,
            )
            writer.add_images(
                "Predicts",
                np.expand_dims(outputs.squeeze(2).detach().cpu().numpy()[0], axis=1),
                batch_idx,
            )

            tt += lossses[-1][0]
            mses.append(mse)
            ssims.append(ssim)
            writer.add_scalar("mae", mse, (epoch * test_loader.__len__()) + batch_idx)
            writer.add_scalar(
                "ssims", ssim, (epoch * test_loader.__len__()) + batch_idx
            )

            [mae_loses.append(n) for time, n in lossses]
            [
                ssim_loses.append(
                    loss_sim(outputs[:, i : i + 1], targets[:, i : i + 1]).item()
                )
                for i in range(outputs.shape[1])
            ]
            # if batch_idx and batch_idx % args.log_valid == 0:
            #     logger.info(
            #         f'EP:{epoch:04d} BI:{batch_idx:03d}/{num_batches:03d} Loss:{np.mean(losses):.6f} MSE:{mse:.4f} SSIM:{ssim:.4f}')

            # visualize(inputs, targets, outputs, epoch, batch_idx, cache_dir)
    data = pd.DataFrame()
    data["mae"] = mae_loses
    data["dates"] = pd.to_datetime(dates, format="%Y%m%d")
    data["ssim"] = ssim_loses
    data["masked_mae"] = masked_mae_loses
    data.to_csv(f"results/104_52_{place}.csv")
    return np.mean(losses), np.mean(mses), np.mean(ssims)


loss_l1 = torch.nn.L1Loss(reduction="none")


def loss_fn(x, y):
    """
    Calculates the combined loss.

        This function computes a loss value by summing L1 loss and a similarity-based term.

        Args:
            x: The first input tensor.
            y: The second input tensor.

        Returns:
            The calculated loss value.
    """
    out = loss_l1(x, y)  # + 0.05*(1-loss_sim(x,y))
    return out


loss_sim = SSIM(data_range=1, size_average=False, channel=1)


def inference_stack(
    args, logger, epoch, model, test_loader, criterion, cache_dir, writer, place, mask
):
    """
    Performs stacked inference on the provided test loader.

        This method iterates through batches of data in the test loader, performs
        stacked predictions using the given model, and saves the results. It also
        calculates and logs various loss metrics (MAE, MSE, SSIM) throughout the
        inference process. Finally it saves a csv file with the losses and dates.

        Args:
            args:  Arguments object containing configuration parameters.
            logger: Logger object for recording information.
            epoch: The current epoch number.
            model: The PyTorch model to use for inference.
            test_loader: DataLoader providing batches of test data.
            criterion: Loss criterion (not directly used in the provided code).
            cache_dir: Directory for caching results.
            writer: TensorBoard writer object for logging.
            place: A string representing the location or context of the inference.
            mask: Mask to apply during loss calculation.

        Returns:
            tuple: A tuple containing the mean losses, MSEs and SSIMs calculated over all batches.
    """
    l1 = torch.nn.L1Loss(reduction="none")
    writer = writer
    model.eval()
    num_batches = len(test_loader)
    losses, mses, ssims = [], [], []
    tt = 0
    ttt = 1
    dates = []
    ssim_loses = []
    masked_mae_loses = []
    mae_loses = []
    for batch_idx, batch in enumerate(test_loader):
        print(tt)
        inputs, targets, x_d, y_d = batch
        inputs, targets = map(lambda x: x.float().to(args.device), [inputs, targets])
        with torch.no_grad():
            test_loader_len = test_loader.__len__()
            [dates.append(time[0]) for time in y_d]
            outputs, ttt, ssim_loses, mae_loses, masked_mae_loses = stack(
                inputs,
                targets[:, :, :6],
                args,
                model,
                writer,
                tt,
                mses,
                ssims,
                epoch,
                losses,
                logger,
                num_batches,
                cache_dir,
                l1,
                test_loader_len,
                ttt,
                ssim_loses,
                mask,
                mae_loses,
                masked_mae_loses,
            )
            [
                np.save(
                    f"npy_results_12_6_30days_kara_stack/{place}/{place}_{y_d[i][0]}.npy",
                    arr,
                    allow_pickle=False,
                )
                for i, arr in enumerate(
                    np.squeeze(outputs.detach().cpu().numpy()[0], axis=1)
                )
            ]
            tt += 1
            new_input = torch.cat(
                (inputs[:, :, :6], outputs.permute(0, 2, 1, 3, 4)), dim=2
            )
            for i in range(8):
                outputs, ttt, ssim_loses, mae_loses, masked_mae_loses = stack(
                    new_input,
                    targets[:, :, (i + 1) * 6 : (i + 1) * 6 + 6],
                    args,
                    model,
                    writer,
                    tt,
                    mses,
                    ssims,
                    epoch,
                    losses,
                    logger,
                    num_batches,
                    cache_dir,
                    l1,
                    test_loader_len,
                    ttt,
                    ssim_loses,
                    mask,
                    mae_loses,
                    masked_mae_loses,
                )
                tt += 1
                new_input = torch.cat(
                    (
                        inputs[:, :, (i + 1) * 6 : (i + 1) * 6 + 6],
                        outputs.permute(0, 2, 1, 3, 4),
                    ),
                    dim=2,
                )
    data = pd.DataFrame()
    data["mae"] = mae_loses
    data["dates"] = pd.to_datetime(dates, format="%Y%m%d")
    data["ssim"] = ssim_loses
    data["masked_mae"] = masked_mae_loses
    data.to_csv(f"results/test7260_{place}.csv")
    return np.mean(losses), np.mean(mses), np.mean(ssims)


def stack(
    inputs,
    targets,
    args,
    model,
    writer,
    batch_idx,
    mses,
    ssims,
    epoch,
    losses,
    logger,
    num_batches,
    cache_dir,
    l1,
    test_loader_len,
    tt,
    ssim_loses,
    mask,
    mae_loses,
    masked_mae_loses,
):
    """
    Stacks tensors and performs forward pass through the model.

        This method permutes input and target tensors, performs a forward pass
        through the specified model (SwinLSTM-B or SwinLSTM-D), calculates losses
        and metrics, logs results, and visualizes predictions.

        Args:
            inputs: Input tensor.
            targets: Target tensor.
            args: Arguments object containing configuration parameters.
            model: The neural network model.
            writer: TensorBoard writer for logging.
            batch_idx: Current batch index.
            mses: List to store Mean Squared Error values.
            ssims: List to store Structural Similarity Index Measure values.
            epoch: Current epoch number.
            losses: List to store loss values.
            logger: Logger object for recording information.
            num_batches: Total number of batches in the dataset.
            cache_dir: Directory for caching visualizations.
            l1: Loss function (likely L1 loss).
            test_loader_len: Length of the test data loader.
            tt: Time tracking variable.
            ssim_loses: List to store SSIM losses.
            mask: Mask tensor for masked loss calculation.
            mae_loses: List to store MAE losses.
            masked_mae_loses: List to store masked MAE losses.

        Returns:
            tuple: A tuple containing the model outputs, updated time tracking variable (ttt),
                   updated ssim_loses list, mae_loses list, and masked_mae_loses list.
    """
    ttt = tt
    inputs = inputs.permute(0, 2, 1, 3, 4)
    targets = targets.permute(0, 2, 1, 3, 4)
    targets_len = targets.shape[1]
    if args.model == "SwinLSTM-B":
        outputs = model_forward_single_layer(model, inputs, targets_len, args.depths)

    if args.model == "SwinLSTM-D":
        outputs = model_forward_multi_layer(
            model, inputs, targets_len, args.depths_down
        )
    # outputs=outputs[:,-targets_len:,:,:]
    outputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()
    targets_ = torch.cat((inputs[:, 1:], targets), dim=1)
    # loss = criterion(outputs, targets_)
    # losses.append(loss.item())
    # lossses = [[i,n.item()] for i,n in enumerate(loss.mean(dim=-1).mean(dim=-1)[0])]
    # [writer.add_scalar('Loss_test',n,(time+tt)) for time,n in lossses]

    # writer.add_scalar('Loss_tesst',losses[-1],(epoch*test_loader.__len__())+batch_idx)
    inputs_len = inputs.shape[1]
    outputs = outputs[:, inputs_len - 1 :]
    mse_l1 = (
        l1(outputs.permute(1, 0, 2, 3, 4), targets.permute(1, 0, 2, 3, 4))
        .mean(dim=-1)
        .mean(dim=-1)[:, 0, 0]
    )
    lossses = [[i, n.item()] for i, n in enumerate(mse_l1)]
    loss_masked = loss_fn(outputs * torch.tensor(np.float32(mask)).to("cuda"), targets)
    loss_maskeds = [
        [i, n.item()] for i, n in enumerate(loss_masked.mean(dim=-1).mean(dim=-1)[0])
    ]
    [masked_mae_loses.append(n) for time, n in loss_maskeds]
    [writer.add_scalar("Loss_test", n, (time + ttt)) for time, n in lossses]
    mse, ssim = compute_metrics(outputs, targets)
    print(batch_idx)
    writer.add_images(
        "Ground_truth",
        np.expand_dims(targets.squeeze(2).detach().cpu().numpy()[0], axis=1),
        batch_idx,
    )
    writer.add_images(
        "Predicts",
        np.expand_dims(outputs.squeeze(2).detach().cpu().numpy()[0], axis=1),
        batch_idx,
    )
    ttt += lossses[-1][0]
    mses.append(mse)
    ssims.append(ssim)
    writer.add_scalar("mae", mse, (epoch * test_loader_len) + batch_idx)
    writer.add_scalar("ssims", ssim, (epoch * test_loader_len) + batch_idx)
    if batch_idx and batch_idx % args.log_valid == 0:
        logger.info(
            f"EP:{epoch:04d} BI:{batch_idx:03d}/{num_batches:03d} Loss:{np.mean(losses):.6f} MSE:{mse:.4f} SSIM:{ssim:.4f}"
        )
        visualize(inputs, targets, outputs, epoch, batch_idx, cache_dir)
    [mae_loses.append(n) for time, n in lossses]
    [
        ssim_loses.append(loss_sim(outputs[:, i : i + 1], targets[:, i : i + 1]).item())
        for i in range(6)
    ]
    return outputs, ttt, ssim_loses, mae_loses, masked_mae_loses
