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


#!/usr/bin/env python3.6
import pickle

import hydra
import logging

from datasets import get_dataset
from models import get_model

from print_pseudolabel_preds_helper  import Task2Vec
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy
import torch
import os

@hydra.main(config_path="conf/config_print_pseudolabel_preds.yaml")
def main(cfg: DictConfig):
    model_str = '{}_{}'.format(cfg.model.arch, cfg.model.pretraining)
    probe_network = get_model(model_str)
    probe_network.cuda()
    logging.info(cfg.pretty())
    train_dataset, test_dataset = get_dataset(cfg.dataset.root, cfg.dataset)
    if hasattr(train_dataset, 'task_name'):
        print(f"======= Embedding for task: {train_dataset.task_name} =======")
    embedding = Task2Vec(probe_network, model_str, **cfg.task2vec).embed(train_dataset)


if __name__ == "__main__":
    main()
