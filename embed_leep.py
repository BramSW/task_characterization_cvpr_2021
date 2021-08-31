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
from collections import Counter

from datasets import get_dataset
from models import get_model, make_encoder
from collections import namedtuple
import torch
import itertools
import random

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

pred_dir = 'leep/'


source_ids = list(range(50))
target_ids = list(range(50))
random.shuffle(source_ids)
random.shuffle(target_ids)

for source_id in source_ids:
    for target_id in target_ids:
        save_dir = "{}/target_{}/source_{}/".format(pred_dir, target_id, source_id)
        save_path = save_dir + "leep.npy"
        if os.path.exists(save_path): continue
        cfg = dict_to_cfg({'name': 'cub_inat2018_no_aug', 'task_id': target_id, 'root': '/data'})
        train_dataset, _ = get_dataset('/data/', cfg)
        if hasattr(train_dataset, 'task_name'):
            print(f"======= Embedding for task: {train_dataset.task_name} =======")
        state_dict = torch.load("/data/bw462/task2vec/experts/cub_attributes/class/dataset.task_id={}/expert.pth".format(source_id))

        base_network = get_model('resnet18', pretraining='imagenet',
                              num_classes=state_dict['fc.bias'].shape[0]).cuda()
        base_network.load_state_dict(state_dict)
        data_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=128,
                                 num_workers=1, drop_last=False)

        n_batches = len(data_loader)
        targets = []
        preds = []
        with torch.no_grad():
            for i, (input, target) in enumerate(itertools.islice(data_loader, 0, n_batches)):
                targets.extend(target.detach().cpu().numpy())
                preds.extend(base_network(input.cuda()).detach().cpu().numpy())
        targets = np.array(targets)
        preds =np.array(preds)
        # Num rows is the real classes (ys), num cols is pred classes (z)
        joint_dist = np.zeros((max(targets)+1, preds.shape[1]))

        for target, pred in zip(targets, preds):
            joint_dist[target] += softmax(pred)
        joint_dist *= 1 / len(preds)
        marginal_dist = joint_dist.sum(axis=0) 
        conditional_dist = joint_dist / marginal_dist
        leep_sum = 0
        for target, pred in zip(targets, preds):
            leep_sum += np.log( np.dot(conditional_dist[target], softmax(pred)))
        leep_sum *= (1 / len(preds))
        print(source_id, target_id, leep_sum)
        os.makedirs(save_dir, exist_ok=True)
        np.save(save_path, leep_sum)


