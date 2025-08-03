# Copyright Agata Walicka, Wroclaw University of Environmental and Life Sciences 2023;
# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch, data
import torch.nn as nn
import sparseconvnet as scn
import tomli


class Model(nn.Module):
    def __init__(self, config, num_classes):
        m = config['unet']['m']
        filter_size = config['unet']['filter_size']
        block_reps = config['unet']['block_reps']

        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(data.dimension, config['data']['full_scale'], mode=4)).add(
            scn.SubmanifoldConvolution(data.dimension, data.number_of_features, m, filter_size, False)).add(
            scn.UNet(data.dimension, block_reps, [m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m],
                     config['unet']['residual_blocks'])).add(
            scn.BatchNormReLU(m)).add(
            scn.OutputLayer(data.dimension))
        self.linear = nn.Linear(m, num_classes)

    def forward(self, x):
        x = self.sparseModel(x)
        x = self.linear(x)
        return x


def save_cpk(save_path1, epoch1, model1, optim1, sch1):
    checkpoint = {
        'epoch': epoch1 + 1,
        'model_state': model1.state_dict(),
        'optimizer_state': optim1.state_dict(),
        'scheduler_state': sch1.state_dict()
    }
    torch.save(checkpoint, save_path1)


def load_cpk(load_path, model, optimizer=None, scheduler=None):
    model_device = next(model.parameters()).device
    checkpoint = torch.load(load_path, map_location=model_device)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    return checkpoint['epoch']


def load_config(config_path):
    with open(config_path, 'rb') as file:
        config = tomli.load(file)
        return config


def logging_filter(record):
    if record.name.startswith('matplotlib'):
        return 0

    return 1
