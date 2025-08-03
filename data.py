# Copyright Agata Walicka, Wroclaw University of Environmental and Life Sciences 2023;
# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import las_reader
import torch, numpy as np, glob, math, torch.utils.data, multiprocessing as mp, time

number_of_features = 2
dimension = 3


class Dataset:

    def __init__(self, config):
        self.config = config
        self.train = []
        self.train_names = []
        self.val = []
        self.val_names = []


    def train_merge(self, tbl):
        logging.debug(f"train merge: {tbl}")

        locs = []
        feats = []
        labels = []
        for idx, i in enumerate(tbl):
            a, b, c, shift = self.load_file([self.train_names[i]])
            m = np.eye(3) + np.random.randn(3, 3) * 0.1
            m[0][0] *= np.random.randint(0, 2) * 2 - 1
            m *= self.config['scale']
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
            a = np.matmul(a, m)
            m = a.min(0)
            M = a.max(0)
            q = M - m
            offset = -m + np.clip(self.config['full_scale'] - M + m - 0.001, 0, None) * np.random.rand(3) + np.clip(
                self.config['full_scale'] - M + m + 0.001, None, 0) * np.random.rand(3)
            a += offset
            idxs = (a.min(1) >= 0) * (a.max(1) < self.config['full_scale'])
            a = a[idxs]
            b = b[idxs]
            c = c[idxs]
            f_number = np.size(b, axis=1)
            a = torch.from_numpy(a).long()
            locs.append(torch.cat([a, torch.LongTensor(a.shape[0], 1).fill_(idx)], 1))
            feats.append(torch.from_numpy(b)+ torch.randn(f_number) * 0.1)
            labels.append(torch.from_numpy(c))
        locs = torch.cat(locs, 0)
        feats = torch.cat(feats, 0)
        labels = torch.cat(labels, 0)
        return {'x': [locs, feats], 'y': labels.long(), 'id': tbl}

    def load_file(self, x):
        file_path = x[0]
        logging.info(f"Loading file {file_path}")

        if file_path.endswith('.pth'):
            return torch.load(file_path)
        elif file_path.endswith('.las') or file_path.endswith('.laz'):
            return las_reader.load_las(file_path, self.config['drop_classes'], self.config['merge_classes'], self.config['classification_type'])

        raise ValueError(f"Unsupported file to read: {file_path}")

    def list_train_data(self):
        self.train_names = glob.glob(self.config['train_data_path'])

    def list_val_data(self):
        self.val_names = glob.glob(self.config['validation_data_path'])

    def create_train_dataloader(self):
        train_data_loader = torch.utils.data.DataLoader(
            list(range(len(self.train_names))),
            batch_size=self.config['batch_size'],
            collate_fn=self.train_merge,
            num_workers=mp.cpu_count() // 2,
            shuffle=True,
            drop_last=True,
            worker_init_fn=lambda x: np.random.seed(x + int(time.time()))
        )

        return train_data_loader

    def val_merge(self, tbl):
        logging.debug(f"val merge: {tbl}")

        locs = []
        feats = []
        labels = []
        a_org = []
        for idx, i in enumerate(tbl):
            a, b, c, shift = self.load_file([self.val_names[i]])
            [a_org.append(temp) for temp in a.copy()]
            m = np.eye(3)
            m *= self.config['scale']
            a = np.matmul(a, m) + self.config['full_scale'] / 2
            m = a.min(0)
            M = a.max(0)
            q = M - m
            offset = -m
            a += offset
            idxs = (a.min(1) >= 0) * (a.max(1) < self.config['full_scale'])
            a = a[idxs]
            b = b[idxs]
            c = c[idxs]
            a = torch.from_numpy(a).long()
            locs.append(torch.cat([a, torch.LongTensor(a.shape[0], 1).fill_(idx)], 1))
            feats.append(torch.from_numpy(b))
            labels.append(torch.from_numpy(c))
        locs = torch.cat(locs, 0)
        feats = torch.cat(feats, 0)
        labels = torch.cat(labels, 0)

        result =  {'name': self.val_names[i][self.val_names[i].find('/')+1:self.val_names[i].rfind('.')],
                'x': [locs, feats],
                'y': labels.long(),
                'z': np.array(a_org) + shift,
                'id': tbl}

        return result

    def create_val_dataloader(self):

        val_data_loader = torch.utils.data.DataLoader(
            list(range(len(self.val_names))),
            batch_size=self.config['batch_size'],
            collate_fn=self.val_merge,
            num_workers=mp.cpu_count() // 2,
            shuffle=True,
            worker_init_fn=lambda x: np.random.seed(x + int(time.time()))
        )

        return val_data_loader
