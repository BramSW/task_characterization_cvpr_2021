#!/usr/bin/env python3
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

import itertools
import scipy.spatial.distance as distance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import numpy as np
import copy
import pickle

_DISTANCES = {}



def _register_distance(distance_fn):
    _DISTANCES[distance_fn.__name__] = distance_fn
    return distance_fn


def is_excluded(k):
    exclude = ['fc', 'linear']
    return any([e in k for e in exclude])


def load_embedding(filename):
    with open(filename, 'rb') as f:
        e = pickle.load(f)
    return e


def get_trivial_embedding_from(e):
    trivial_embedding = copy.deepcopy(e)
    for l in trivial_embedding['layers']:
        a = np.array(l['filter_logvar'])
        a[:] = l['filter_lambda2']
        l['filter_logvar'] = list(a)
    return trivial_embedding


def binary_entropy(p):
    from scipy.special import xlogy
    return - (xlogy(p, p) + xlogy(1. - p, 1. - p))


def get_layerwise_variance(e, normalized=False):
    var = [np.exp(l['filter_logvar']) for l in e['layers']]
    if normalized:
        var = [v / np.linalg.norm(v) for v in var]
    return var


def get_variance(e, normalized=False):
    # print(e.hessian.min())
    var = 1. / np.array(e.hessian)
    if normalized:
        lambda2 = 1. / np.array(e.scale)
        var = var / lambda2
    return var


def get_variances(*embeddings, normalized=False):
    return [get_variance(e, normalized=normalized) for e in embeddings]


def get_hessian(e, normalized=False):
    hess = np.array(e.hessian)
    if normalized:
        scale = np.array(e.scale)
        hess = hess / scale
    return hess


def get_hessians(*embeddings, normalized=False):
    return [get_hessian(e, normalized=normalized) for e in embeddings]


def get_scaled_hessian(e0, e1, normalized=False):
    # this looks like equation of S3.3
    h0, h1 = get_hessians(e0, e1, normalized=normalized)
    return h0 / (h0 + h1 + 1e-8), h1 / (h0 + h1 + 1e-8)


def get_full_kl(e0, e1):
    var0, var1 = get_variance(e0), get_variance(e1)
    kl0 = .5 * (var0 / var1 - 1 + np.log(var1) - np.log(var0))
    kl1 = .5 * (var1 / var0 - 1 + np.log(var0) - np.log(var1))
    return kl0, kl1


def layerwise_kl(e0, e1):
    layers0, layers1 = get_layerwise_variance(e0), get_layerwise_variance(e1)
    kl0 = []
    for var0, var1 in zip(layers0, layers1):
        kl0.append(np.sum(.5 * (var0 / var1 - 1 + np.log(var1) - np.log(var0))))
    return kl0


def layerwise_cosine(e0, e1):
    layers0, layers1 = get_layerwise_variance(e0, normalized=True), get_layerwise_variance(e1, normalized=True)
    res = []
    for var0, var1 in zip(layers0, layers1):
        res.append(distance.cosine(var0, var1))
    return res

@_register_distance
def layerwise_cosine_test(e0, e1):
    layers0, layers1 = get_layerwise_variance(e0, normalized=True), get_layerwise_variance(e1, normalized=True)
    res = []
    for var0, var1 in zip(layers0, layers1):
        res.append(distance.cosine(var0, var1))
    return res

@_register_distance
def kl(e0, e1):
    var0, var1 = get_variance(e0), get_variance(e1)
    kl0 = .5 * (var0 / var1 - 1 + np.log(var1) - np.log(var0))
    kl1 = .5 * (var1 / var0 - 1 + np.log(var0) - np.log(var1))
    return np.maximum(kl0, kl1).sum()


@_register_distance
def asymmetric_kl(e0, e1):
    var0, var1 = get_variance(e0), get_variance(e1)
    kl0 = .5 * (var0 / var1 - 1 + np.log(var1) - np.log(var0))
    kl1 = .5 * (var1 / var0 - 1 + np.log(var0) - np.log(var1))
    return kl0.sum()


@_register_distance
def jsd(e0, e1):
    var0, var1 = get_variance(e0), get_variance(e1)
    var = .5 * (var0 + var1)
    kl0 = .5 * (var0 / var - 1 + np.log(var) - np.log(var0))
    kl1 = .5 * (var1 / var - 1 + np.log(var) - np.log(var1))
    return (.5 * (kl0 + kl1)).mean()


@_register_distance
def cosine(e0, e1):
    # below yields the Fa/(Fa + Fb)
    if hasattr(e0, 'hessian'): h1, h2 = get_scaled_hessian(e0, e1)
    else: h1, h2 = e0, e1
    return distance.cosine(h1, h2)

@_register_distance
def cosine_norm_bias(e0, e1):
    # THIS WORKS WITH SOFT PSEUDOLABELS VARIATIONAL KL
    # below yields the Fa/(Fa + Fb)
    if hasattr(e0, 'hessian'): h1, h2 = get_scaled_hessian(e0, e1)
    else: h1, h2 = e0, e1
    return distance.cosine(h1, h2) - 1e-7 * np.linalg.norm(e1.hessian)

@_register_distance
def cosine_norm_bias_tmp(e0, e1, alpha):
    # THIS WORKS WITH SOFT PSEUDOLABELS VARIATIONAL KL
    # below yields the Fa/(Fa + Fb)
    if hasattr(e0, 'hessian'): h1, h2 = get_scaled_hessian(e0, e1)
    else: h1, h2 = e0, e1
    return distance.cosine(h1, h2) - alpha * np.linalg.norm(e1.hessian)


@_register_distance
def raw_cosine(e0, e1):
    h0 = e0.hessian
    h1 = e1.hessian
    return distance.cosine(h0, h1)

@_register_distance
def per_element_cosine_tmp(e0, e1):
    # below yields the Fa/(Fa + Fb)
    h0_raw = e0.hessian
    h1_raw = e1.hessian
    denom = abs(h0_raw + h1_raw) + 1e-10
    h0 = h0_raw / denom
    h1 = h1_raw / denom
    return distance.cosine(h0, h1)

@_register_distance
def per_element_l2_norm_cosine(e0, e1):
    # below yields the Fa/(Fa + Fb)
    h0_raw = e0.hessian
    h1_raw = e1.hessian
    denom = (h0_raw**2 + h1_raw**2) + 1e-10
    h0 = h0_raw / denom
    h1 = h1_raw / denom
    return distance.cosine(h0, h1)

@_register_distance
def unit_cosine(e0, e1):
    # below yields the Fa/(Fa + Fb)
    h0 = e0.hessian / np.linalg.norm(e0.hessian)
    h1 = e1.hessian / np.linalg.norm(e1.hessian)
    return distance.cosine(h0, h1)

@_register_distance
def unit_l1_cosine(e0, e1):
    # below yields the Fa/(Fa + Fb)
    h0 = e0.hessian / np.linalg.norm(e0.hessian, ord=1)
    h1 = e1.hessian / np.linalg.norm(e1.hessian, ord=1)
    return distance.cosine(h0, h1)


@_register_distance
def sum_normalized_cosine(e0, e1):
    # below yields the Fa/(Fa + Fb)
    h0 = e0.hessian / sum(e0.hessian)
    h1 = e1.hessian / sum(e1.hessian)
    return distance.cosine(h0, h1)

@_register_distance
def cosine_scaled(e0, e1):
    # below yields the Fa/(Fa + Fb)
    if hasattr(e0, 'hessian'): h1, h2 = get_scaled_hessian(e0, e1, normalized=True)
    else: h1, h2 = e0, e1
    return distance.cosine(h1, h2)


@_register_distance
def asymmetric_cosine(Fa, Fb):
    # Convention is that first entry is row which is target task
    # Thus we're flipped from Eq before 3.4: we compare the *source*
    # (ta in Eq, Fb here) to trivial
    # below yields the Fa/(Fa + Fb)
    if hasattr(Fa, 'hessian'): ha, hb = get_scaled_hessian(Fa, Fb)
    else: raise NotImplementedEror; Fa, Fb = e0, e1
    triv = Fb.scale
    hb_raw = Fb.hessian
    matched_b = hb_raw / (hb_raw + triv + 1e-8)
    matched_triv = triv / (hb_raw + triv + 1e-8)
    term1 = distance.cosine(ha, hb)
    term2 = distance.cosine(matched_b, matched_triv)
    return term1 - 0.15 * term2


@_register_distance
def adjusted_cosine(Fa, Fb):
    # Convention is that first entry is row which is target task
    # Thus we're flipped from Eq before 3.4: we compare the *source*
    # (ta in Eq, Fb here) to trivial
    # below yields the Fa/(Fa + Fb)
    ha = Fa.hessian
    hb = Fb.hessian
    ha = (2/(1e-7)) * ha + Fa.scale
    hb = (2/(1e-7)) * hb + Fb.scale
    ha, hb = ha / (ha + hb + 1e-8), hb / (ha + hb + 1e-8)

    term1 = distance.cosine(ha, hb)
    return term1

@_register_distance
def adjusted_asymmetric_cosine_0_3(Fa, Fb):
    # Convention is that first entry is row which is target task
    # Thus we're flipped from Eq before 3.4: we compare the *source*
    # (ta in Eq, Fb here) to trivial
    # below yields the Fa/(Fa + Fb)
    ha = Fa.hessian
    hb = Fb.hessian
    ha = (2/(1e-7)) * ha + Fa.scale
    hb = (2/(1e-7)) * hb + Fb.scale
    ha, hb = ha / (ha + hb + 1e-8), hb / (ha + hb + 1e-8)

    triv = Fb.scale
    hb_raw = (2/(1e-7)) * Fb.hessian + Fb.scale
    matched_b = hb_raw / (hb_raw + triv + 1e-8)
    matched_triv = triv / (hb_raw + triv + 1e-8)
    term1 = distance.cosine(ha, hb)
    term2 = distance.cosine(matched_b, matched_triv)
    return term1 - 0.3 * term2

@_register_distance
def adjusted_asymmetric_cosine_0_15(Fa, Fb):
    # Convention is that first entry is row which is target task
    # Thus we're flipped from Eq before 3.4: we compare the *source*
    # (ta in Eq, Fb here) to trivial
    # below yields the Fa/(Fa + Fb)
    ha = Fa.hessian
    hb = Fb.hessian
    ha = (2/(1e-7)) * ha + Fa.scale
    hb = (2/(1e-7)) * hb + Fb.scale
    ha, hb = ha / (ha + hb + 1e-8), hb / (ha + hb + 1e-8)

    triv = Fb.scale # * Fb.scale
    hb_raw = (2/(1e-7)) * Fb.hessian + Fb.scale
    matched_b = hb_raw / (hb_raw + triv + 1e-8)
    matched_triv = triv / (hb_raw + triv + 1e-8)
    term1 = distance.cosine(ha, hb)
    term2 = distance.cosine(matched_b, matched_triv)
    return term1 - 0.15 * term2


@_register_distance
def adjusted_asymmetric_cosine_tmp(Fa, Fb, alpha):
    # Convention is that first entry is row which is target task
    # Thus we're flipped from Eq before 3.4: we compare the *source*
    # (ta in Eq, Fb here) to trivial
    # below yields the Fa/(Fa + Fb)
    ha = Fa.hessian
    hb = Fb.hessian
    ha = (2/(1e-7)) * ha + Fa.scale
    hb = (2/(1e-7)) * hb + Fb.scale
    ha, hb = ha / (ha + hb + 1e-8), hb / (ha + hb + 1e-8)

    triv = Fb.scale 
    hb_raw = (2/(1e-7)) * Fb.hessian + Fb.scale
    matched_b = hb_raw / (hb_raw + triv + 1e-8)
    matched_triv = triv / (hb_raw + triv + 1e-8)
    term1 = distance.cosine(ha, hb)
    term2 = distance.cosine(matched_b, matched_triv)
    return term1 - alpha * term2

@_register_distance
def adjusted_asymmetric_cosine_clipped(Fa, Fb):
    # Convention is that first entry is row which is target task
    # Thus we're flipped from Eq before 3.4: we compare the *source*
    # (ta in Eq, Fb here) to trivial
    # below yields the Fa/(Fa + Fb)
    ha = Fa.hessian
    hb = Fb.hessian

    clip_quantile = 0.9
    a_cutoff = np.quantile(ha, clip_quantile)
    b_cutoff = np.quantile(hb, clip_quantile)
    ha = np.clip(ha, ha.min(), a_cutoff)
    hb = np.clip(hb, hb.min(), b_cutoff)

    ha = (2/(1e-7)) * ha + Fa.scale
    hb = (2/(1e-7)) * hb + Fb.scale
    ha, hb = ha / (ha + hb + 1e-8), hb / (ha + hb + 1e-8)

    triv = Fb.scale
    hb_raw = (2/(1e-7)) * Fb.hessian + Fb.scale
    matched_b = hb_raw / (hb_raw + triv + 1e-8)
    matched_triv = triv / (hb_raw + triv + 1e-8)
    term1 = distance.cosine(ha, hb)
    term2 = distance.cosine(matched_b, matched_triv)
    return term1 - 0.15 * term2

@_register_distance
def flipped_asymmetric_cosine(Fb, Fa):
    # Convention is that first entry is row which is target task
    # Thus we're flipped from Eq before 3.4: we compare the *source*
    # (ta in Eq, Fb here) to trivial
    # below yields the Fa/(Fa + Fb)
    if hasattr(Fa, 'hessian'): ha, hb = get_scaled_hessian(Fa, Fb)
    else: raise NotImplementedEror; Fa, Fb = e0, e1
    triv = Fb.scale
    hb_raw = Fb.hessian
    matched_b = hb_raw / (hb_raw + triv + 1e-8)
    matched_triv = triv / (hb_raw + triv + 1e-8)
    term1 = distance.cosine(ha, hb)
    term2 = distance.cosine(matched_b, matched_triv)
    return term1 - 0.15 * term2


@_register_distance
def normalized_cosine(e0, e1):
    h1, h2 = get_variances(e0, e1, normalized=True)
    return distance.cosine(h1, h2)


@_register_distance
def correlation(e0, e1):
    v1, v2 = get_variances(e0, e1, normalized=False)
    return distance.correlation(v1, v2)


@_register_distance
def entropy(e0, e1):
    h1, h2 = get_scaled_hessian(e0, e1)
    return np.log(2) - binary_entropy(h1).mean()

@_register_distance
def pca_cosine(e0, e1):
    similarity = 0
    for i in range(32):
        similarity += e0.components_[i].dot(e1.components_[i])
    return -1 * similarity

@_register_distance
def pca_cosine_normalized(e0, e1):
    similarity = 0
    for i in range(32):
        component_0 = e0.components_[i] / np.linalg.norm(e0.components_[i])
        component_1 = e1.components_[i] / np.linalg.norm(e1.components_[i])
        similarity += component_0.dot(component_1)
    return -1 * similarity

@_register_distance
def average_then_cosine(e0, e1):
    features_0, features_1 = e0.features, e1.features
    return distance.cosine(features_0.mean(axis=0), features_1.mean(axis=0))


@_register_distance
def h_divergence(e0, e1):
    features_0, features_1 = e0.features, e1.features
    clf = make_pipeline(StandardScaler(),LinearSVC())
    X = np.concatenate((features_0, features_1))
    y = np.concatenate((np.zeros(features_0.shape[0]), np.ones(features_1.shape[0])))
    clf.fit(X, y)
    return clf.score(X, y)


import torch
def pytorch_linear_acc(X, y, normalize=False):
    classifier = torch.nn.Linear(512, 2).cuda()
    frac_1 = ( y.shape[0] - y.sum()) / y.sum()
    weight_tensor = torch.Tensor([frac_1])
    one_hot = torch.zeros(y.shape[0], 2)
    one_hot[torch.arange(y.shape[0]), y] = 1
    if normalize:
        X = torch.Tensor(X)
        norm = X.pow(2).sum(1, keepdim=True).pow(1./2)
        X = X.div(norm)
    dataset = torch.utils.data.TensorDataset(torch.Tensor(X), one_hot)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight_tensor).cuda()
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=X.shape[0], num_workers=6, drop_last=False)
    num_epochs = 100
    optimizer = torch.optim.Adam(classifier.parameters())
    epochs_since_best_acc = 0
    best_acc = -1
    epoch_i = 0
    while epochs_since_best_acc < 2:
        for data, target in data_loader:
            data = data.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            output = classifier(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            preds = output.argmax(dim=1)
            true = target.argmax(dim=1)
            acc = float((preds==true).sum()) / preds.size(0)
            if acc > best_acc:
                best_acc = acc
                print("%.2f %.2f %d" %(loss.item(), best_acc, epochs_since_best_acc))
                epochs_since_best_acc = 0
            else:
                epochs_since_best_acc += 1
            epoch_i += 1
    print("-"*10)
    return best_acc


@_register_distance
def h_divergence_gpu(e0, e1):
    features_0, features_1 = e0.features, e1.features
    X = np.concatenate((features_0, features_1))
    y = np.concatenate((np.zeros(features_0.shape[0]), np.ones(features_1.shape[0])))
    return pytorch_linear_acc(X, y)

@_register_distance
def h_divergence_normalized_gpu(e0, e1):
    features_0, features_1 = e0.features, e1.features
    X = np.concatenate((features_0, features_1))
    y = np.concatenate((np.zeros(features_0.shape[0]), np.ones(features_1.shape[0])))
    return pytorch_linear_acc(X, y, normalize=True)


def get_normalized_embeddings(embeddings, normalization=None):
    F = [1. / get_variance(e, normalized=False) if e is not None else None for e in embeddings]
    zero_embedding = np.zeros_like([x for x in F if x is not None][0])
    F = np.array([x if x is not None else zero_embedding for x in F])
    if normalization is None:
        normalization = np.sqrt((F ** 2).mean(axis=0, keepdims=True))
    F /= normalization
    return F, normalization


def pdist(embeddings, distance='cosine'):
    if '_weighted_by_dataset_size' in distance:
        # Idea here is to explicitly bias towards larger datasets instead of more complicated
        # Spefically using the fact that the benefit of datasets generally seems to scale logarithimically
        #
        distance = distance[:-1*len('_weighted_by_dataset_size')]
        import pickle
        dataset_weighting = np.array([count**0.25 for name,count in pickle.load(open('dataset_sizes.pickle', 'rb'))])
    else:
        dataset_weighting = np.ones(len(embeddings))
    if '_tmp_' in distance:
        true_distance,alpha = distance.split('_tmp_')
        true_distance += '_tmp'
        alpha = float(alpha)
        print(true_distance, alpha)
        distance_fn_base = _DISTANCES[true_distance]
        distance_fn = lambda a,b: distance_fn_base(a,b,alpha=alpha)
    else:
        distance_fn = _DISTANCES[distance]
    n = len(embeddings)
    distance_matrix = np.zeros([n, n])
    if 'asymmetric' not in distance:
        for (i, e1), (j, e2) in itertools.combinations(enumerate(embeddings), 2):
            distance_matrix[i, j] = distance_fn(e1, e2) * dataset_weighting[j]
            distance_matrix[j, i] = distance_matrix[i, j] * dataset_weighting[i] / dataset_weighting[j]
    else:
        for (i, e1) in enumerate(embeddings):
            for (j, e2) in enumerate(embeddings):
                distance_matrix[i, j] = distance_fn(e1, e2) * dataset_weighting[j]
    return distance_matrix


def cdist(from_embeddings, to_embeddings, distance='cosine'):
    distance_fn = _DISTANCES[distance]
    distance_matrix = np.zeros([len(from_embeddings), len(to_embeddings)])
    for (i, e1) in enumerate(from_embeddings):
        for (j, e2) in enumerate(to_embeddings):
            if e1 is None or e2 is None:
                continue
            distance_matrix[i, j] = distance_fn(e1, e2)
    return distance_matrix


def plot_distance_matrix(embeddings, labels=None, distance='cosine', savename='tmp.png', fade_diagonal=False, data_save_str='', ensembled_embeddings=False):
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    import pandas as pd
    import matplotlib.pyplot as plt
    if ensembled_embeddings:
        distance_matrices = [pdist(embedding_sample, distance=distance) for embedding_sample in embeddings]
        print(distance_matrices)
        distance_matrix = np.average(distance_matrices, axis=0)
        print(distance_matrix)
    else:
        distance_matrix = pdist(embeddings, distance=distance)
    cond_distance_matrix = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(cond_distance_matrix, method='complete', optimal_ordering=True)
    if fade_diagonal:
        mean_distance = (distance_matrix - np.diag(np.diag(distance_matrix))).mean()
        distance_matrix = distance_matrix - np.diag(np.diag(distance_matrix)) + mean_distance * np.eye(len(embeddings))
    if labels is not None:
        distance_matrix = pd.DataFrame(distance_matrix, index=labels, columns=labels)
    sns.clustermap(distance_matrix, row_linkage=linkage_matrix, col_linkage=linkage_matrix, cmap='viridis_r')
    if data_save_str:
        pickle.dump(distance_matrix, open(data_save_str, 'wb'))
    plt.tight_layout()
    plt.savefig(savename)




