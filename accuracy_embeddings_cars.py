#!/usr/bin/env python

# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import os
from copy import deepcopy
import json
import sys
from io import BytesIO
import argparse
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform, pdist
import pickle
import task_similarity
import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

CATEGORIES_JSON_FILE = 'inat2018/categories.json'
ICONS_PATH = './static/iconic_taxa/'

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--root', type=str)
parser.add_argument('--distance', default='cosine', type=str,
                    help='distance to use')
args = parser.parse_args()

imagenet_errs = [21.774839743589745, 31.03966346153846, 41.45633012820513, 25.87139423076923, 32.49198717948718, 18.299278846153847, 6.740785256410256, 3.886217948717949, 30.35857371794872, 10.556891025641026, 0.0, 22.536057692307693, 0.0, 0.671073717948718, 24.28886217948718, 0.0, 9.875801282051283, 7.852564102564102, 0.0, 11.57852564102564, 12.760416666666666, 20.733173076923077, 2.874599358974359, 0.0, 0.0, 11.107772435897436, 9.775641025641026, 13.772035256410257, 0.0, 0.0]


def main():
    distance_matrices = []
    names = []
    root = args.root
    files = glob.glob(os.path.join(root, '*/*/*', 'embedding.p'))
    if not files: files = glob.glob(os.path.join(root, '*/*', 'embedding.p'))
    if not files: files = glob.glob(os.path.join(root, '*', 'embedding.p'))
    if not files: files = glob.glob(os.path.join(root, '*/*/*', 'features.p'))
    if not files: files = glob.glob(os.path.join(root, '*/*', 'features.p'))
    embeddings = [task_similarity.load_embedding(file) for file in files]
    embeddings.sort(key=lambda x: x.meta['dataset']['task_id'])

    for e in embeddings:
        e.task_id = e.meta['dataset']['task_id']
        e.task_name = e.meta['task_name']
        e.dataset = 'CARS'
    task_id_to_name = {e.task_id:e.task_name for e in embeddings}
    distance_matrix = task_similarity.pdist(embeddings, distance=args.distance)
    embeddings = np.array(embeddings)
    np.fill_diagonal(distance_matrix, 0.)
    assert distance_matrix.shape[0]==30, distance_matrix.shape[1]==30
    names = [str(s) for s in list(range(30))]
    names = [n.lower() for n in names]
    error_mat = np.load('npy/car_errs.npy')
    errors = { target_name:{source_name:error_mat[j][i] for j,source_name in enumerate(names)} for i,target_name in enumerate(names[:30])}
    for k,v in errors.items():
        optimal_err = np.inf
        for source_name, source_err in v.items():
            if source_name==k: continue
            else:
                optimal_err = min(optimal_err, source_err)
        errors[k]['optimal'] = optimal_err
    selected_errors = []
    optimal_errors = []
    random_errors = []
    closest_distances = []
    selected_experts = []
    self_errors = []
    for name_i, (distances, name) in enumerate(zip(distance_matrix, names)):
        distances[name_i] = 10e6
        closest_i = np.argmin(distances)
        closest_distances.append((name, distances[closest_i]))
        optimal_error = float(errors[name]['optimal'])
        selected_error = errors[name][names[closest_i]]
        random_error = np.average([errors[name][source] for source in names if source!=name])
        selected_errors.append(selected_error)
        random_errors.append(random_error)
        optimal_errors.append(optimal_error)
        self_errors.append(errors[name][name])
        selected_experts.append(closest_i)
    name_to_imagenet_err = pickle.load(open('pickles/name_to_imagenet_err.pickle', 'rb'))
    name_to_imagenet_err = {k.lower():v for k,v in name_to_imagenet_err.items()}
    err_types = ['t2v', 'opt', 'rand', 'self', 'imagenet']
    for err_list, err_type in zip((selected_errors, optimal_errors, random_errors, self_errors, imagenet_errs), err_types):
        print("Average Relative (per-task) Error")
        print(err_type, average_relative_error_zero_proofed(err_list, optimal_errors))


def average_relative_error_zero_proofed(err_list, opt_list):
    diffs = (np.array(err_list) - np.array(opt_list))
    denoms = np.array(opt_list)
    rel_errs = []
    for diff, denom in zip(diffs, denoms):
        denom += 1
        if denom==0 and diff==0:
            rel_errs.append(0)
        elif denom==0:
            print(denom, diff); asdf
        else:
            rel_errs.append(diff /denom)
    return 100 * (np.average(rel_errs))

if __name__ == '__main__':
    main()
