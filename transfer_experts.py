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
    logging.info(cfg.pretty())
    # By default does classification
    # If want attribute then in commandline specifc that dataset.name is cub_attributes
    # and dataset.attribute is whatever you want (wing color, etc)
    print(cfg.dataset)
    num_to_transfer_to = 50 if cfg.dataset.name=='cub_inat2018' else 25
    # Only have attrs for CUB, but may want to transfer from iNat to
    tmp_dataset_name_holder = None
    tmp_dataset_attr_holder = None
    if cfg.dataset.task_id >= 25:
        tmp_dataset_name_holder = cfg.dataset.name
        tmp_dataset_attr_holder = cfg.dataset.attribute
        cfg.dataset.name = 'cub_inat2018'
        cfg.dataset.attribute = 'class'
    train_dataset, test_dataset = get_dataset(cfg.dataset.root, cfg.dataset)
    if hasattr(train_dataset, 'task_name'):
        print(f"======= Transferring task: {train_dataset.task_name} =======")

    base_dir = '{}/{}_{}/'.format(cfg.cub_inat.base_dir, cfg.dataset.name, cfg.dataset.attribute)
    if cfg.dataset.attribute=='class':
        sub_dir = "{}_dataset.task_id={}/".format(cfg.dataset.task_id, cfg.dataset.task_id)
        if not os.path.exists(base_dir+sub_dir):
            sub_dir = "dataset.task_id={}/".format(cfg.dataset.task_id)
    else:
        sub_dir = "{}_dataset.attribute={},dataset.name={},dataset.task_id={}/".format(cfg.dataset.task_id,
                                                                                        cfg.dataset.attribute,
                                                                                      cfg.dataset.name,
                                                                                      cfg.dataset.task_id)
        if not os.path.exists(base_dir+sub_dir):
            sub_dir = "dataset.attribute={},dataset.name={},dataset.task_id={}/".format(cfg.dataset.attribute,
                                                                                          cfg.dataset.name,
                                                                                          cfg.dataset.task_id)
    trained_sd = torch.load(base_dir + sub_dir + 'expert.pth')
    base_num_classes = train_dataset.num_classes
    if tmp_dataset_name_holder is not None:
        cfg.dataset.name = tmp_dataset_name_holder
        cfg.dataset.attribute = tmp_dataset_attr_holder
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
