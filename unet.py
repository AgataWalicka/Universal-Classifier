# Copyright Agata Walicka, Wroclaw University of Environmental and Life Sciences 2023;
# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys

import torch, data, iou
import torch.optim as optim
import sparseconvnet as scn
import time
import matplotlib.pyplot as plt
from itertools import count
import numpy as np
import unet_common as uc


def update_graph(x_new, y_new, fig, axs, lin, num_feat=-1):
    if num_feat != -1:
        for k in range(0, num_feat):
            lin[k].set_xdata(x_new[k, :])
            lin[k].set_ydata(y_new[k, :])
    else:
        lin.set_xdata(x_new)
        lin.set_ydata(y_new)

    axs.relim()
    axs.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    if len(sys.argv) < 2:
        logging.critical("no config file provided, aborting")
        exit(1)

    config = uc.load_config(sys.argv[1])

    report_save_path = config['unet'].get('report_save_path', '')
    if report_save_path != '':
        file_handler = logging.FileHandler(report_save_path)
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

    for h in logging.getLogger().handlers:
        h.addFilter(uc.logging_filter)

    logging.info(f"Starting unet")
    logging.debug(f"configuration: {config}")

    cuda_available = torch.cuda.is_available()
    logging.info(f"CUDA available: {cuda_available}")

    cuda_device = config['unet'].get('cuda_device')

    use_cuda = config['unet'].get('use_cuda', cuda_device is not None)
    if use_cuda:
        if not cuda_available:
            logging.critical("CUDA is required by configuration (use_cuda or cuda_device), but not available. Aborting")
            exit(1)

        if cuda_device is not None:
            torch.cuda.set_device(cuda_device)

        curr_cuda_dev = torch.cuda.current_device()
        logging.info(f"Using CUDA device: {curr_cuda_dev}, properties: {torch.cuda.get_device_properties(curr_cuda_dev)}")
    else:
        logging.info(f"NOT using CUDA!")

    # Options
    restore =           config['unet']['restore']
    restore_from_file = config['unet']['restore_file_path']
    m =                 config['unet']['m']  # 16 or 32
    residual_blocks =   config['unet']['residual_blocks']
    training_epochs =   config['unet']['training_epochs']
    num_classes =       iou.N_CLASSES
    class_weights =     1/np.sqrt(config['unet']['class_shares'])

    assert len(class_weights) == 0 or len(class_weights) == num_classes, 'Number of class weights does not match number of classes!'

    if len(class_weights) == 0:
        logging.info(f"No class weights available. Calculating loss function without weights.")
    else:
        logging.info(f"Applying class weights: {class_weights}")
        class_weights = torch.Tensor(class_weights)
        class_weights = class_weights.cuda()

    exp_name = f"unet_voxelsize_{1 / config['data']['scale']}_m_{m}_rep_{config['data']['val_reps']}_ResidualBlocks_{residual_blocks}_batchsize_{config['data']['batch_size']}_maxepochs_{training_epochs}_"


    logging.info(
        f"voxel size: {1 / config['data']['scale']}, m: {m}, rep: {config['data']['val_reps']}, Residual blocks:{residual_blocks}, batch size: {config['data']['batch_size']}, training epochs: {training_epochs}")
    logging.info(f"classes: {iou.CLASS_LABELS}")


    dataset = data.Dataset(config['data'])

    dataset.list_train_data()
    dataset.list_val_data()

    unet = uc.Model(config, num_classes)
    if use_cuda:
        unet = unet.cuda()

    optimizer = optim.Adam(unet.parameters())
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, training_epochs + 1, last_epoch=-1, verbose=True)

    if restore:
        training_epoch = uc.load_cpk(restore_from_file, unet, optimizer, scheduler)
    else:
        training_epoch = 1

    logging.info(f'#classifer parameters {sum([x.nelement() for x in unet.parameters()])} \n')

    plt.style.use('classic')
    plt.ion()
    figure, ax = plt.subplots(2, 2, figsize=(23, 15))

    lines_loss, = ax[0, 0].plot([], [], '-')
    ax[0, 0].set_autoscaley_on(True)
    ax[0, 0].set_autoscalex_on(True)
    ax[0, 0].grid()
    ax[0, 0].set_title('Loss value for each batch')

    lines_lr, = ax[0, 1].plot([], [], '-')
    ax[0, 1].set_autoscaley_on(True)
    ax[0, 1].set_xlim(0, training_epochs)
    ax[0, 1].grid()
    ax[0, 1].set_title('Learning rate')

    lines_loss2, = ax[1, 0].plot([], [], 'o-')
    ax[1, 0].set_autoscaley_on(True)
    ax[1, 0].set_xlim(0, training_epochs)
    ax[1, 0].grid()
    ax[1, 0].set_title('Loss value for each iteration')

    num_ious = training_epochs
    lines_features = []
    for f in range(0, num_classes):
        lines, = ax[1, 1].plot([], [], '-o', label=iou.CLASS_LABELS[f])
        lines_features.append(lines)
    ax[1, 1].set_autoscaley_on(True)
    ax[1, 1].set_xlim(0, training_epochs + 5)
    ax[1, 1].grid()
    ax[1, 1].legend(loc='lower right')
    ax[1, 1].set_title('IoU values for each iteration')

    xdata_loss = []
    ydata_loss = []

    xdata_lr = []
    ydata_lr = []

    xdata_loss2 = []
    ydata_loss2 = []

    xdata_features = np.zeros((num_classes, num_ious))
    ydata_features = np.zeros((num_classes, num_ious))

    index_loss = count()
    index_lr = count()
    index_loss2 = count()
    index_features = count()

    train_data_loader = dataset.create_train_dataloader()
    val_data_loader = dataset.create_val_dataloader()

    for epoch in range(training_epoch, training_epochs + 1):
        logging.info(f"training epoch {epoch}")

        xdata_lr.append(next(index_lr))
        ydata_lr.append(scheduler.get_last_lr())
        update_graph(xdata_lr, ydata_lr, figure, ax[0, 1], lines_lr)

        unet.train()
        scn.forward_pass_multiplyAdd_count = 0
        scn.forward_pass_hidden_states = 0
        start = time.time()
        train_loss = 0
        for i, batch in enumerate(train_data_loader):
            optimizer.zero_grad()
            if use_cuda:
                batch['x'][1] = batch['x'][1].float().cuda()
                batch['y'] = batch['y'].cuda()
            predictions = unet(batch['x'])

            if len(class_weights) == 0:
                loss = torch.nn.functional.cross_entropy(predictions, batch['y'])
            else:
                loss = torch.nn.functional.cross_entropy(predictions, batch['y'], class_weights)

            train_loss += loss.item()

            xdata_loss.append(next(index_loss))
            ydata_loss.append(loss.item())
            update_graph(xdata_loss, ydata_loss, figure, ax[0, 0], lines_loss)

            logging.info(f"Loss value for batch {i + 1}: {loss.item()}")

            loss.backward()
            optimizer.step()
        scheduler.step()

        xdata_loss2.append(next(index_loss2))
        ydata_loss2.append(train_loss / (i + 1))
        update_graph(xdata_loss2, ydata_loss2, figure, ax[1, 0], lines_loss2)

        logging.info(f"Epoch {epoch}, Train loss {train_loss / (i + 1)}, MegaMulAdd= {scn.forward_pass_multiplyAdd_count / len(dataset.train_names) / 1e6}, MegaHidden= {scn.forward_pass_hidden_states / len(dataset.train_names) / 1e6}, time= {time.time() - start} s")

        uc.save_cpk(config['unet']['cpk_save_path']+exp_name + f"epoch_{epoch}", epoch, unet, optimizer, scheduler)

        if epoch == training_epochs:
            with torch.no_grad():
                unet.eval()

                scn.forward_pass_multiplyAdd_count = 0
                scn.forward_pass_hidden_states = 0
                start = time.time()
                for rep in range(1, 1 + config['data']['val_reps']):
                    confusion_matr = np.zeros((iou.N_CLASSES, iou.N_CLASSES))
                    pts_number = 0
                    for i, batch in enumerate(val_data_loader):

                        if use_cuda:
                            batch['x'][1] = batch['x'][1].float().cuda()

                        predictions = unet(batch['x'])

                        confusion_temp = iou.confusion_matrix(predictions.cpu().max(1)[1].numpy(), batch['y'].numpy())

                        confusion_matr = np.add(confusion_matr, confusion_temp)
                        pts_number = pts_number + len(batch['y'].numpy())


                    logging.info(f"{epoch}, {rep}, Val MegaMulAdd= {scn.forward_pass_multiplyAdd_count / len(dataset.val_names) / 1e6}, MegaHidden= {scn.forward_pass_hidden_states / len(dataset.val_names) / 1e6}, time= {time.time() - start} s")

                    confusion_matr = confusion_matr.astype('int32')

                    m_iou, ious = iou.evaluate(pts_number, confusion_matr)
                    iou.evaluate_oa(pts_number, confusion_matr)
                    iou.evaluate_pr(pts_number, confusion_matr)

                    i_feat = next(index_features)
                    for i in range(0, num_classes):
                        xdata_features[i, i_feat] = epoch
                        ydata_features[i, i_feat] = ious[iou.CLASS_LABELS[i]][0]

                    xdata_features_diag = xdata_features[:, 0:i_feat + 1]
                    ydata_features_diag = ydata_features[:, 0:i_feat + 1]
                    update_graph(xdata_features_diag, ydata_features_diag, figure, ax[1, 1], lines_features, num_classes)

    figure_save_path = config['unet'].get('figure_save_path', '')
    if figure_save_path != '':
        logging.debug(f'Saving figure to: {figure_save_path}')
        figure.savefig(figure_save_path, dpi=300)


if __name__ == '__main__':
    main()
