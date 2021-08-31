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

from datasets import get_dataset
from models import get_model

from sup_training import transfer_model
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf/config_transfer_from_imagenet.yaml")
def main(cfg: DictConfig):
    logging.info(cfg.pretty())
    # By default does classification
    # If want attribute then in commandline specifc that dataset.name is cub_attributes
    # and dataset.attribute is whatever you want (wing color, etc)
    train_dataset, test_dataset = get_dataset(cfg.dataset.root, cfg.dataset)
    if hasattr(train_dataset, 'task_name'):
        print(f"======= Training task: {train_dataset.task_name} =======")
    base_network = get_model(cfg.model.arch, pretraining=cfg.model.pretraining,
                              num_classes=train_dataset.num_classes)
    base_network = base_network.to(cfg.device)
    transfer_error = transfer_model(base_network, train_dataset, test_dataset)
    meta = OmegaConf.to_container(cfg, resolve=True)
    pickle.dump(transfer_error, open("error.pickle", "wb"))


if __name__ == "__main__":
    main()
