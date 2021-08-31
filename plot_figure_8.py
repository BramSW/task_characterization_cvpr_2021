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
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams.update({'font.size': 12})
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform, pdist
import pickle
import task_similarity
import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import matplotlib.pyplot as plt

CATEGORIES_JSON_FILE = 'inat2018/categories.json'
ICONS_PATH = './static/iconic_taxa/'

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--roots', nargs='+', type=str)
parser.add_argument('--data-root', default='/data', type=str)
parser.add_argument('--distance', default='cosine', type=str,
                    help='distance to use')
parser.add_argument('--max-tasks', default=120, type=int,
                    help='number of tasks to consider')
args = parser.parse_args()

# Assumes task IDs are mapped as follows:
# CUB: 0-24 (0-9 are orders, 10-24 are Passeriformes families)
# iNat2018: 25-231
# As in cub_inat2018 in datasets.py
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


def invert_icon(img):
    img = (1. - img)
    return img


def get_image(e):
    base = os.path.join(ICONS_PATH, "{}-200px.png")
    possible_names = [base.format(x) for x in
                      [e.meta.get('class'), e.meta.get('phylum'), e.meta.get('kingdom'), 'unknown']]
    for filename in possible_names:
        if os.path.exists(filename):
            img = plt.imread(filename, format='png')
            return invert_icon(img) if e.dataset == CUB else img
    raise FileNotFoundError()


def average_top_k_tax_distance(distance_matrix, from_embeddings, to_embeddings=None, k=2):
    assert k > 0, k

    if to_embeddings is None:
        to_embeddings = from_embeddings

    assert distance_matrix.shape[0] == len(from_embeddings)
    assert distance_matrix.shape[1] == len(to_embeddings)

    tax_distance = []
    for i in range(len(from_embeddings)):
        top_matches = distance_matrix[i].argsort()[:k]
        tax_distance.append(
            np.mean([taxonomy_distance(from_embeddings[i], to_embeddings[j]) for j in top_matches])
        )
    return np.mean(tax_distance)


def plot_changing_k(ax, distance_matrix, from_embeddings, to_embeddings, **kwargs):
    x = [1, 3, 5, 10, 20, 30, 50, 100, 200, len(to_embeddings)]
    x = [v for v in x if v <= len(to_embeddings)]
    y = []
    for k in x:
        y.append(average_top_k_tax_distance(distance_matrix, from_embeddings, to_embeddings, k=k))
    ax.plot(x, y, **kwargs)
    ax.set_xlabel('Size k of neighborhood')
    ax.set_ylabel('Avg. top-k tax. distance')


def sort_distance_matrix(distance_matrix, embeddings, names, method='complete'):
    assert method in ['ward', 'single', 'average', 'complete']
    np.fill_diagonal(distance_matrix, 0.)
    cond_distance_matrix = squareform(distance_matrix, checks=False)
    linkage_matrix = hierarchy.linkage(cond_distance_matrix, method='complete', optimal_ordering=True)
    res_order = hierarchy.leaves_list(linkage_matrix)
    distance_matrix = distance_matrix[res_order][:, res_order]
    embeddings = [embeddings[i] for i in res_order]
    names = [names[i] for i in res_order]
    np.fill_diagonal(distance_matrix, np.nan)
    return distance_matrix, embeddings, names, res_order


def draw_figure_to_plt(distance_matrix, embeddings, names, label_size=14):
    fig = plt.figure(figsize=(15 / 25. * len(embeddings), 15 / 25. * len(embeddings)))
    ax = plt.gca()

    plt.imshow(distance_matrix, cmap='viridis_r')
    ax.set_xticks(np.arange(len(embeddings)))
    ax.set_yticks(np.arange(len(embeddings)))

    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    try:
        for i, e in enumerate(embeddings):
            arr_img = get_image(e)
            imagebox = OffsetImage(arr_img, zoom=0.18)
            imagebox.image.axes = ax
            xy = (i, i)
            ab = AnnotationBbox(imagebox, xy, frameon=False)
            ax.add_artist(ab)
    except FileNotFoundError:
        print("Could not find an icon for a taxonomy entry. Have you downloaded the iconic_taxa directory in ./static?")

    plt.tick_params(axis='both', which='major', labelsize=label_size)
    plt.tight_layout()


def taxonomy_distance(e0, e1):
    for i, k in enumerate(['order', 'class', 'phylum', 'kingdom']):
        if e0.meta[k] == e1.meta[k]:
            return i
    return i + 1


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
    xlim = 1e4
    roots = ['multirun_fixed_pseudolabels_imagenet/variational_resnet18_imagenet',
            'multirun_fixed_pseudolabels_imagenet/variational_resnet18_places365']


    all_embedding_entries = []

    for root in roots:
        os.makedirs('./plots', exist_ok=True)

        files = glob.glob(os.path.join(root, '*/*/*', 'embedding.p'))
        if not files: files = glob.glob(os.path.join(root, '*/*', 'embedding.p'))
        if not files: files = glob.glob(os.path.join(root, '*/*/*', 'features.p'))
        if not files: files = glob.glob(os.path.join(root, '*/*', 'features.p'))
        # get embeddings
        embeddings = [task_similarity.load_embedding(file) for file in files]
        embeddings.sort(key=lambda x: x.meta['dataset']['task_id'])
        embedding_entries = []
        for e in embeddings:
            e.task_id = e.meta['dataset']['task_id']
            e.task_name = e.meta['task_name']
            e.dataset = CUB if e.task_id < CUB_NUM_TASKS else INAT
            embedding_entries.extend(list(e.hessian))
        all_embedding_entries.append(embedding_entries)
    fig, ax = plt.subplots(figsize=(5,2))
    ax.set_yscale('log')

    bins = 20
    ax.hist(all_embedding_entries[0], alpha=0.5, bins=np.linspace(0, xlim, bins), label='ImageNet', color='red')
    ax.hist(all_embedding_entries[1], alpha=0.5, bins=np.linspace(0, xlim, bins), label='Places365', color='green')
    ax.legend(loc='lower center')
    ax.set_xlim(0, xlim)
    ax.set_ylabel('Count')
    ax.set_xlabel('Embedding Element Value')
    fig.suptitle('Values of Entries in PseudoTask Embeddings')
    fig.subplots_adjust(left=0.15, bottom=0.3)
    plt.savefig('paper_figures/embedding_entries.png')

if __name__ == '__main__':
    main()
