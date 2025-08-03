# Copyright Agata Walicka, Wroclaw University of Environmental and Life Sciences 2023;
# Copyright 2016-present, Facebook, Inc.
# All rights reserved.

import torch, data, iou
import laspy
import torch.optim as optim
import sparseconvnet as scn
import numpy as np
import unet_common as uc
import logging
import sys
import os


def save_results(res, path):
    with open(path, 'w') as file:
        for row in res:
            file.write(' '.join([str(item) for item in row]))
            file.write('\n')


def save_las(path, data):
    # create header
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="random", type=np.int32))
    header.offsets = np.min(data[:, 0:3], axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    # create las structure
    las = laspy.LasData(header)
    # add coordinates
    las.x = data[:, 0]
    las.y = data[:, 1]
    las.z = data[:, 2]
    # add attributes
    las.classification = data[:, 3]
    las.points['intensity'] = data[:, 4]
    las.write(path)


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

    logging.info(f"Starting unet test")
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

    m =                 config['unet']['m']  # 16 or 32
    residual_blocks =   config['unet']['residual_blocks']
    training_epochs =   config['unet']['training_epochs']
    num_classes =       iou.N_CLASSES
    unet_path =         config['unet_test']['unet_path']
    pc_save_path =      config['unet_test']['pc_save_path']


    logging.info(
        f"voxel size: {1 / config['data']['scale']}, m: {m}, rep: {config['data']['val_reps']}, Residual blocks:{residual_blocks}, batch size: {config['data']['batch_size']}, training epochs: {training_epochs}")
    logging.info(f"classes: {iou.CLASS_LABELS}")

    dataset = data.Dataset(config['data'])

    dataset.list_val_data()

    unet = uc.Model(config, num_classes)
    if use_cuda:
        unet = unet.cuda()

    optimizer = optim.Adam(unet.parameters())
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, training_epochs, last_epoch=-1, verbose=True)

    uc.load_cpk(unet_path, unet, optimizer, scheduler)
    val_data_loader = dataset.create_val_dataloader()

    mean_results_summary = []
    all_results_summary = []
    confusion_matrices = []
    with torch.no_grad():
        unet.eval()
        scn.forward_pass_multiplyAdd_count = 0
        scn.forward_pass_hidden_states = 0

        for rep in range(1, 1 + config['data']['val_reps']):
            for i, batch in enumerate(val_data_loader):

                temp = batch['name']
                logging.info(f'Analyzed file: {temp}')

                if use_cuda:
                    batch['x'][1] = batch['x'][1].float().cuda()
                predictions = unet(batch['x'])
                original_labels = batch['y'].numpy()
                labels = predictions.cpu().max(1)[1].numpy()

                classified = np.concatenate((batch['z'], labels.reshape((-1, 1)),
                                                 np.asarray(original_labels).reshape((-1, 1))), axis=1)

                confusion_matr = iou.confusion_matrix(labels, original_labels)
                m_iou, ious = iou.evaluate(len(original_labels), confusion_matr)
                m_oa, oas = iou.evaluate_oa(len(original_labels), confusion_matr)
                m_pr, prs = iou.evaluate_pr(len(original_labels), confusion_matr)
                fname = os.path.basename(batch['name'])
                mean_results_summary.append([fname, m_iou, m_oa, m_pr])
                all_results_summary.append([fname, ious.copy(), oas.copy(), prs.copy()])
                confusion_matrices.append([fname, confusion_matr.copy().reshape((1, -1)).tolist()])
                save_las(os.path.join(pc_save_path, fname + '.las'), classified)

    save_results(mean_results_summary, os.path.join(pc_save_path, 'mean_results_summary.txt'))
    save_results(all_results_summary, os.path.join(pc_save_path, 'all_results_summary.txt'))
    save_results(confusion_matrices, os.path.join(pc_save_path, 'confusion_matrices.txt'))
    logging.info("Finished")


if __name__ == '__main__':
    main()
