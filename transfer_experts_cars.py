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
import torch
import os

from datasets import get_dataset
from models import get_model

from sup_training import transfer_model
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf/config_train_experts.yaml")
def main(cfg: DictConfig):
    cfg.dataset.name = 'cars'
    logging.info(cfg.pretty())
    num_to_transfer_to = 30
    train_dataset, test_dataset = get_dataset(cfg.dataset.root, cfg.dataset)
    if hasattr(train_dataset, 'task_name'):
        print(f"======= Transferring task: {train_dataset.task_name} =======")

    base_dir = cfg.cars.base_dir
    sub_dir = cfg.cars.sub_dir_prefix.format(cfg.dataset.task_id, cfg.dataset.task_id)
    trained_sd = torch.load(base_dir + sub_dir + 'expert.pth')
    base_num_classes = train_dataset.num_classes
    for i in range(num_to_transfer_to):
        base_network = get_model(cfg.model.arch, pretraining=cfg.model.pretraining,
                                  num_classes=base_num_classes)
        base_network.load_state_dict(trained_sd)
        base_network = base_network.to(cfg.device)
        cfg.dataset.task_id = i
        train_dataset, test_dataset = get_dataset(cfg.dataset.root, cfg.dataset)
        transfer_error = transfer_model(base_network, train_dataset, test_dataset)
        del base_network
        pickle.dump(transfer_error, open("{}_error.pickle".format(i), "wb"))


if __name__ == "__main__":
    main()
