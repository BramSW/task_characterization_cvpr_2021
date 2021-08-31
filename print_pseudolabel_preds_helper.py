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
import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
import logging
import variational
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer
from utils import AverageMeter, get_error, get_device
from pprint import pprint
import pickle

from torchvision.transforms import *
import PIL
import models

imagenet_idx_to_label = pickle.load(open('pickles/imagenet_idx_to_label.pickle', 'rb'))


class ProbeNetwork(ABC, nn.Module):
    """Abstract class that all probe networks should inherit from.

    This is a standard torch.nn.Module but needs to expose a classifier property that returns the final classicifation
    module (e.g., the last fully connected layer).
    """

    @property
    @abstractmethod
    def classifier(self):
        raise NotImplementedError("Override the classifier property to return the submodules of the network that"
                                  " should be interpreted as the classifier")

    @classifier.setter
    @abstractmethod
    def classifier(self, val):
        raise NotImplementedError("Override the classifier setter to set the submodules of the network that"
                                  " should be interpreted as the classifier")


class Task2Vec:

    def __init__(self, model: ProbeNetwork, teacher_model_str, skip_layers=0, max_samples=None, classifier_opts=None,
                 method='montecarlo', method_opts=None, loader_opts=None, bernoulli=False):
        if classifier_opts is None:
            classifier_opts = {}
        if method_opts is None:
            method_opts = {}
        if loader_opts is None:
            loader_opts = {}
        assert method in ('variational', 'montecarlo')
        assert skip_layers >= 0

        self.model = model
        # Fix batch norm running statistics (i.e., put batch_norm layers in eval mode)
        self.model.train()
        self.device = get_device(self.model)
        self.skip_layers = skip_layers
        self.max_samples = max_samples
        self.classifier_opts = classifier_opts
        self.method = method
        self.method_opts = method_opts
        self.loader_opts = loader_opts
        self.bernoulli = True
        self.loss_fn = nn.KLDivLoss(reduction='batchmean') # nn.CrossEntropyLoss() if not self.bernoulli else nn.BCEWithLogitsLoss()
        self.loss_fn = self.loss_fn.to(self.device)
        self.teacher_model = models.get_model(teacher_model_str).cuda()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

    def embed(self, dataset: Dataset, reset_classifier=True):
        self._cache_features(dataset, max_samples=self.max_samples)

    def _cache_features(self, dataset: Dataset, indexes=(-1,), max_samples=None, loader_opts: dict = None):
        logging.info("Caching features...")
        if loader_opts is None:
            loader_opts = {}
        data_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=loader_opts.get('batch_size', 128),
                                 num_workers=loader_opts.get('num_workers', 6), drop_last=False)
        device = next(self.model.parameters()).device

        def _hook(layer, inputs):
            if not hasattr(layer, 'input_features'):
                layer.input_features = []
            layer.input_features.append(inputs[0].data.cpu().clone())

        hooks = [self.model.layers[index].register_forward_pre_hook(_hook)
                 for index in indexes]
        if max_samples is not None:
            n_batches = min(
                math.floor(max_samples / data_loader.batch_size) - 1, len(data_loader))
        else:
            n_batches = len(data_loader)
        targets = []
        for i, (input, real_label) in tqdm(enumerate(itertools.islice(data_loader, 0, n_batches)), total=n_batches,
                                       leave=False,
                                       desc="Caching features"):
            output = self.model(input.to(device))
            pred_label = output.argmax(dim=1)
            pred_text_labels = [imagenet_idx_to_label[idx] for idx in pred_label.detach().cpu().numpy()]
            pprint(list(zip(real_label, pred_text_labels)))
            return

def _get_loader(trainset, testset=None, batch_size=128, num_workers=6, num_samples=10000, drop_last=True):
    if getattr(trainset, 'is_multi_label', False):
        raise ValueError("Multi-label datasets not supported")
    if hasattr(trainset, 'labels'):
        labels = trainset.labels
    elif hasattr(trainset, 'targets'):
        labels = trainset.targets
    else:
        labels = list(trainset.tensors[1].cpu().numpy())
    weights = np.ones(len(labels))
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=num_samples)
    num_workers = num_workers if not isinstance(trainset, torch.utils.data.TensorDataset) else 0
    trainloader = torch.utils.data.DataLoader(trainset, sampler=sampler, batch_size=batch_size,
                                              num_workers=num_workers, drop_last=drop_last)

    if testset is None:
        return trainloader
    else:
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, pin_memory=True, shuffle=False,
                                                 num_workers=num_workers)
        return trainloader, testloader
