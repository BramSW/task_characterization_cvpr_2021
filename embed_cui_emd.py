from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import time
from sklearn.metrics.pairwise import euclidean_distances


from datasets import get_dataset
from models import get_model, make_encoder
from collections import namedtuple
import torch
import itertools
import pyemd


def dict_to_cfg(d):
    cfg = namedtuple('cfg', ['name', 'task_id', 'root'])
    return cfg(**d)

feature_dir = 'cui_emd_features/'

for task_id in range(50):
    if os.path.exists(feature_dir + str(task_id) + '_weight.npy'): continue
    cfg = dict_to_cfg({'name': 'cub_inat2018_no_aug', 'task_id': task_id, 'root': '/data'})
    train_dataset, _ = get_dataset('/data/', cfg)
    if hasattr(train_dataset, 'task_name'):
        print(f"======= Embedding for task: {train_dataset.task_name} =======")
    probe_network = make_encoder(get_model('resnet18', pretraining='imagenet',
                              num_classes=train_dataset.num_classes)).cuda()
    data_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=128,
                             num_workers=1, drop_last=False)

    n_batches = len(data_loader)
    targets = []
    features = []
    probe_network.linear = torch.nn.Identity()
    for i, (input, target) in enumerate(itertools.islice(data_loader, 0, n_batches)):
        targets.extend(target.detach().cpu().numpy())
        features.extend(probe_network(input.cuda()).detach().cpu().numpy())
    targets = np.array(targets)
    features =np.array(features)

    sorted_label = sorted(list(set(targets)))
    feature_per_class = np.zeros((len(sorted_label), 512), dtype=np.float32)
    weight = np.zeros((len(sorted_label), ), dtype=np.float32)
    counter = 0
    print('Original feature shape: (%d, %d)' % (features.shape[0], features.shape[1]))
    print('Number of classes: %d' % (len(np.unique(targets))))
    for i in sorted_label:
        idx = [(l==i) for l in targets]
        feature_per_class[counter, :] = np.mean(features[idx, :], axis=0)
        weight[counter] = np.sum(idx)
        counter += 1
    print('Feature per class shape: (%d, %d)' % (feature_per_class.shape[0], 
                                             feature_per_class.shape[1]))

    np.save(feature_dir + str(task_id) + '.npy', feature_per_class)
    np.save(feature_dir + str(task_id) + '_weight.npy', weight)



# Calculate domain similarity by Earth Mover's Distance (EMD).


# Gamma for domain similarity: exp(-gamma x EMD)
gamma = 0.01

source_domains = range(50)
target_domains = range(50)
similarity_matrix = np.zeros((50, 50))

tic = time.time()
for sd in source_domains:
    for td in target_domains:
        print('%s --> %s' % (sd, td))
        f_s = np.load(feature_dir + '%s.npy' %sd)
        f_t = np.load(feature_dir + '%s.npy' %td)
        w_s = np.load(feature_dir + '%s_weight.npy' %sd)
        w_t = np.load(feature_dir + '%s_weight.npy' %td)

        f_s = f_s[idx, :]
        w_s = w_s[idx]

        # Make sure two histograms have the same length and distance matrix is square.
        data = np.float64(np.append(f_s, f_t, axis=0))
        w_1 = np.zeros((len(w_s) + len(w_t),), np.float64)
        w_2 = np.zeros((len(w_s) + len(w_t),), np.float64)
        w_1[:len(w_s)] = w_s / np.sum(w_s)
        w_2[len(w_s):] = w_t / np.sum(w_t)
        D = euclidean_distances(data, data)

        emd = pyemd.emd(np.float64(w_1), np.float64(w_2), np.float64(D))
        domain_similarity = np.exp(-gamma*emd)
        similarity_matrix[sd][td] = domain_similarity
        print('EMD: %.3f    Domain Similarity: %.3f\n' % (emd, domain_similarity))
np.save('cui_similarity_mat.npy', similarity_matrix)
print('Elapsed time: %.3fs' % (time.time() - tic))
np.fill_diagonal(similarity_matrix, np.nan)

def draw_figure_to_plt(distance_matrix, names, label_size=14):
    fig = plt.figure(figsize=(15 / 25. * len(names), 15 / 25. * len(names)))
    ax = plt.gca()

    plt.imshow(distance_matrix, cmap='viridis_r')
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))

    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.tick_params(axis='both', which='major', labelsize=label_size)
    plt.tight_layout()
    plt.savefig('domain_similarity.png')

names = []
for task_id in range(50):
    cfg = dict_to_cfg({'name': 'cub_inat2018', 'task_id': task_id, 'root': '/data'})
    train_dataset, _ = get_dataset('/data/', cfg)
    names.append(train_dataset.task_name)
draw_figure_to_plt(-1 *  similarity_matrix, names)
