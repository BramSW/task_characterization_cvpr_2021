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

CUB = 'CUB'
INAT = 'iNat'
CUB_NUM_TASKS = 25
ADDITIONAL_TAXONOMY_DATA = [
    {
        'kingdom': 'Animalia ',
        'supercategory': 'Animalia ',
        'phylum': 'Chordata',
        'class': 'Aves',
        'order': 'Apodiformes',
    }
]


CATEGORIES_JSON_FILE = 'inat2018/categories.json'

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--root', type=str)
parser.add_argument('--data-root', type=str, default='/data')
parser.add_argument('--distance', default='cosine', type=str,
                    help='distance to use')
args = parser.parse_args()

def add_class_information(embeddings):
    # load taxonomy
    with open(os.path.join(args.data_root, CATEGORIES_JSON_FILE), 'r') as f:
        categories = json.load(f)
    categories.extend(ADDITIONAL_TAXONOMY_DATA)

    category_map = {c['order']: c for c in categories}
    category_map.update({c['family']: c for c in categories if 'family' in c})

    for e in embeddings:
        try:
            c = category_map[e.task_name]
        except:
            if 'Passeriformes' in e.task_name and '_' in e.task_name:
                c = category_map[e.task_name.split('_')[1]]
            else:
                raise
        e.meta['order'] = c['order'].lower()
        e.meta['class'] = c['class'].lower()
        e.meta['phylum'] = c['phylum'].lower()
        e.meta['kingdom'] = c['kingdom'].lower()
        e.meta['supercategory'] = c['supercategory'].lower()


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
    assert(len(embeddings)==50)
    for e in embeddings:
        e.task_id = e.meta['dataset']['task_id']
        e.task_name = e.meta['task_name']
        e.dataset = CUB if e.task_id < CUB_NUM_TASKS else INAT
    add_class_information(embeddings)

    task_id_to_name = {e.task_id:e.task_name for e in embeddings}
    distance_matrix = task_similarity.pdist(embeddings, distance=args.distance)
    embeddings = np.array(embeddings)
    np.fill_diagonal(distance_matrix, 0.)
    assert distance_matrix.shape[0]==50, distance_matrix.shape[1]==50
    new_names = [f"[{e.dataset}] {e.task_name} ({e.meta['class']})" if 'order' in e.meta
             else f"[{e.dataset}] {e.task_name} ({e.meta['class']})" for e in embeddings]
    new_names = [s.replace('Passeriformes_', '') for s in new_names]
    names = [n.lower() for n in new_names]
    error_mat = np.load('npy/breast_pattern_errs.npy')
    errors = pickle.load(open('pickles/cub_inat2018_performance_dict.pickle', 'rb'))
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
    imagenet_errors = [name_to_imagenet_err[' '.join(name.split()[:-1])] for name in names]
    err_types = ['t2v', 'opt', 'rand', 'self','imagenet']
    for err_list, err_type in zip((selected_errors, optimal_errors, random_errors, self_errors, imagenet_errors), err_types):
        print("Average Relative (per-task) Error")
        print(err_type, average_relative_error(err_list, optimal_errors))


def average_relative_error(err_list, opt_list):
    return np.average(100 * (np.array(err_list) - np.array(opt_list)) / np.array(opt_list))


if __name__ == '__main__':
    main()
