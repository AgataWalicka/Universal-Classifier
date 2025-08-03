# Copyright Agata Walicka, Wroclaw University of Environmental and Life Sciences 2023;
# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import logging

CLASS_LABELS = ['Ground', 'Vegetation', 'Buildings', 'Other']
UNKNOWN_ID = -100
N_CLASSES = len(CLASS_LABELS)


def confusion_matrix(pred_ids, gt_ids):
    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    idxs = gt_ids >= 0
    return np.bincount(pred_ids[idxs] * N_CLASSES + gt_ids[idxs], minlength=N_CLASSES**2).reshape((N_CLASSES, N_CLASSES))


def get_iou(label_id, confusion):
    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false positives
    fp = np.longlong(confusion[label_id, :].sum()) - tp
    # false negatives
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    denom = (tp + fp + fn)
    if denom == 0:
        return 0, tp, denom
    return float(tp) / denom, tp, denom


def evaluate(gt_ids, confusion):
    logging.info(f'evaluating {gt_ids} points...')

    class_ious = {}
    mean_iou = 0
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        class_ious[label_name] = get_iou(i, confusion)
        mean_iou += class_ious[label_name][0] / N_CLASSES

    logging.info('classes          IoU')
    logging.info('----------------------------')

    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        logging.info('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(label_name, class_ious[label_name][0],
                                                               class_ious[label_name][1], class_ious[label_name][2]))
    logging.info(f'mean IOU: {mean_iou}\n')

    return mean_iou, class_ious


def get_oa(label_id, confusion):
    # true positives
    tp = np.longlong(confusion[label_id, label_id])

    # false negatives
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    denom = (tp + fn)
    if denom == 0:
        return 0, tp, denom
    return float(tp) / denom, tp, denom


def evaluate_oa(gt_ids, confusion):
    logging.info(f'evaluating {gt_ids} points...')

    class_oa = {}
    mean_oa = 0
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        class_oa[label_name] = get_oa(i, confusion)
        mean_oa += class_oa[label_name][0] / N_CLASSES

    logging.info('classes          sensitivity')
    logging.info('----------------------------')

    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]

        logging.info('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(label_name, class_oa[label_name][0],
                                                               class_oa[label_name][1], class_oa[label_name][2]))
    logging.info(f'mean sensitivity {mean_oa}\n')
    return mean_oa, class_oa

def get_pr(label_id, confusion):
    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false positives
    fp = np.longlong(confusion[label_id, :].sum()) - tp

    denom = (tp + fp)
    if denom == 0:
        return 0, tp, denom
    return float(tp) / denom, tp, denom


def evaluate_pr(gt_ids, confusion):
    logging.info(f'evaluating {gt_ids} points...')

    class_pr = {}
    mean_pr = 0
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        class_pr[label_name] = get_pr(i, confusion)
        mean_pr += class_pr[label_name][0] / N_CLASSES

    logging.info('classes          precision')
    logging.info('----------------------------')

    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]

        logging.info('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(label_name, class_pr[label_name][0],
                                                               class_pr[label_name][1], class_pr[label_name][2]))
    logging.info(f'mean precision {mean_pr}\n\n')
    return mean_pr, class_pr
