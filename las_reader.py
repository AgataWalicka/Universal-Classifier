# Copyright Agata Walicka, Wroclaw University of Environmental and Life Sciences 2023;
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import laspy
import numpy as np


def remap_labels(original_labels):
    remapper = np.ones(150) * (-100)
    for i, x in enumerate([2, 5, 6, 8]):
        remapper[x] = i
    return remapper[np.array(original_labels)]


def binary_labels_water_ground(original_labels):
    temp1 = np.asarray([original_labels == 2])
    temp2 = np.asarray([original_labels == 9])
    return np.asarray(temp1 != temp2).astype('uint8')[0]


def preselect_classes(labels, classes_to_miss):
    return ~np.in1d(np.asarray(labels), np.asarray(classes_to_miss))


def merge_classes(labels, destination_class, merging_class):
    labels[labels == merging_class] = destination_class
    return labels


def load_las(file_path, classes_to_drop, classes_to_merge, classification_type):
    data = laspy.read(file_path)
    features = np.transpose(np.array([data.points["number_of_returns"], data.points["return_number"]], dtype=float))
    coord = data.xyz
    mean_coord = np.mean(coord, axis=0)
    coord = coord - mean_coord
    labels = data.classification

    if len(classes_to_drop) > 0:
        ind = preselect_classes(labels, classes_to_drop)
        labels = labels[ind]
        coord = coord[ind]
        features = features[ind]
    if classification_type == 'binary':
        labels_corrected = binary_labels_water_ground(labels.copy())
    elif classification_type == 'multiclass':
        if len(classes_to_merge) > 0:
            for i in range(0, len(classes_to_merge)):
                labels = merge_classes(labels.copy(), classes_to_merge[i][0], classes_to_merge[i][1])
        labels_corrected = remap_labels(labels.copy())
    else:
        labels_corrected = labels
        print("Error")
    return coord, features, labels_corrected, mean_coord
