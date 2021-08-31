from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import time
from sklearn.metrics.pairwise import euclidean_distances
from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr
from collections import Counter

from datasets import get_dataset
from models import get_model, make_encoder
from collections import namedtuple
import torch
import itertools


class Config:
    def __init__(self, name, task_id, root):
        self.name = name
        self.task_id = task_id
        self.root = root

    def get(self, key, default):
        if key in ['name', 'task_id', 'root']:
            return getattr(self, key)
        else:
            return default

def dict_to_cfg(d):
    return Config(**d)

pred_dir = 'rsa/'

for target_id in range(50):
    cfg = dict_to_cfg({'name': 'cub_inat2018_no_aug', 'task_id': target_id, 'root': '/data'})
    train_dataset, _ = get_dataset('/data/', cfg)
    num_images_to_use = 500
    data_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=num_images_to_use,
                             num_workers=1, drop_last=False)
    for images, _ in data_loader:
        break
    num_images_to_use = min(num_images_to_use, images.shape[0])
    for source_id in range(50):
        save_dir = "{}/target_{}/source_{}/".format(pred_dir, target_id, source_id)
        rdm_save_path = save_dir + "rdm.npy"
        if os.path.exists(rdm_save_path): continue
        state_dict = torch.load("/data/bw462/task2vec/experts/cub_attributes/class/dataset.task_id={}/expert.pth".format(source_id))

        base_network = get_model('resnet18', pretraining='imagenet',
                              num_classes=state_dict['fc.bias'].shape[0]).cuda()
        base_network.load_state_dict(state_dict)
        base_network.linear = torch.nn.Identity()
        with torch.no_grad(): features = base_network(images.cuda()).detach().cpu().numpy()

        rdm_mat = np.zeros((num_images_to_use, num_images_to_use))
        for rdm_i in range(num_images_to_use):
            for rdm_j in range(num_images_to_use):
                # Just want to fill in strict lower triangle
                if rdm_i <= rdm_j: continue
                rdm_mat[rdm_i][rdm_j] = pearsonr(features[rdm_i], features[rdm_j])[0]
        os.makedirs(save_dir)
        np.save(rdm_save_path, rdm_mat)


