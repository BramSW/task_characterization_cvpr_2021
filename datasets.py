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


import collections
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.vision import VisionDataset
import os
import json
from PIL import Image
import numpy as np

try:
    from IPython import embed
except:
    pass

_DATASETS = {}

Dataset = collections.namedtuple(
    'Dataset', ['trainset', 'testset'])


def _add_dataset(dataset_fn):
    _DATASETS[dataset_fn.__name__] = dataset_fn
    return dataset_fn

def _get_augmented_transforms(augment=True, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    basic_transform = [transforms.ToTensor(), normalize]

    transform_train = []
    if augment:
        transform_train += [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        transform_train += [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    transform_train += basic_transform
    transform_train = transforms.Compose(transform_train)

    transform_test = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    transform_test += basic_transform
    transform_test = transforms.Compose(transform_test)

    return transform_train, transform_test

def _get_transforms(augment=True, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    basic_transform = [transforms.ToTensor(), normalize]

    transform_train = []
    if augment:
        transform_train += [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        transform_train += [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    transform_train += basic_transform
    transform_train = transforms.Compose(transform_train)

    transform_test = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    transform_test += basic_transform
    transform_test = transforms.Compose(transform_test)

    return transform_train, transform_test

def set_metadata(trainset, testset, config, dataset_name):
    trainset.metadata = {
        'dataset': dataset_name,
        'task_id': config.task_id,
        'task_name': trainset.task_name,
    }
    testset.metadata = {
        'dataset': dataset_name,
        'task_id': config.task_id,
        'task_name': testset.task_name,
    }
    return trainset, testset



def load_tasks_map(tasks_map_file):
    assert os.path.exists(tasks_map_file), tasks_map_file
    with open(tasks_map_file, 'r') as f:
        tasks_map = json.load(f)
        tasks_map = {int(k): int(v) for k, v in tasks_map.items()}
    return tasks_map


@_add_dataset
def cub_attributes(root, config):
    """
    This is cub+inat but only works for cub and config needs to have attribute to classify on
    """
    NUM_CUB = 25
    NUM_CUB_ORDERS = 10
    NUM_INAT = 207
    assert 0 <= config.task_id < NUM_CUB + NUM_INAT
    transform_train, transform_test = _get_transforms()
    if 0 <= config.task_id < NUM_CUB:
        # CUB
        from dataset.cub import CUBTasks, CUBDataset
        tasks_map_file = os.path.join(root, 'cub/CUB_200_2011', 'final_tasks_map.json')
        tasks_map = load_tasks_map(tasks_map_file)
        task_id = int(config.task_id)
        train_tasks = CUBTasks(CUBDataset(root, split='train', attribute_to_classify=config.attribute))
        trainset = train_tasks.generate(task_id=task_id,
                                        use_species_names=True,
                                        transform=transform_train,
                                        random_label=config.get('random_label', ''),
                                      attribute_to_classify=config.get('attribute', ''))
        test_tasks = CUBTasks(CUBDataset(root, split='test', attribute_to_classify=config.attribute))
        testset = test_tasks.generate(task_id=task_id,
                                      use_species_names=True,
                                      transform=transform_test,
                                      random_label=config.get('random_label', ''),
                                      attribute_to_classify=config.get('attribute', ''))
    else:
        raise NotImplementedError
    trainset, testset = set_metadata(trainset, testset, config, 'cub_attributes')
    return trainset, testset


@_add_dataset
def cub_inat2018(root, config):
    """This meta-task is the concatenation of CUB-200 (first 25 tasks) and iNat (last 207 tasks).

    - The first 10 tasks are classification of the animal species inside one of 10 orders of birds in CUB-200
        (considering all orders except passeriformes).
    - The next 15 tasks are classification of species inside the 15 families of the order of passerifomes
    - The remaining 207 tasks are classification of the species inside each of 207 families in iNat

    As noted above, for CUB-200 10 taks are classification of species inside an order, rather than inside of a family
    as done in the iNat (recall order > family > species). This is done because CUB-200 has very few images
    in each family of bird (expect for the families of passeriformes). Hence, we go up step in the taxonomy and
    consider classification inside a orders and not families.
    """
    NUM_CUB = 25
    NUM_CUB_ORDERS = 10
    NUM_INAT = 207
    assert 0 <= config.task_id < NUM_CUB + NUM_INAT
    transform_train, transform_test = _get_transforms()
    if 0 <= config.task_id < NUM_CUB:
        # CUB
        from dataset.cub import CUBTasks, CUBDataset
        tasks_map_file = os.path.join(root, 'cub/CUB_200_2011', 'final_tasks_map.json')
        tasks_map = load_tasks_map(tasks_map_file)
        # task_id = tasks_map[config.task_id]
        task_id = int(config.task_id)
        # CUB orders
        train_tasks = CUBTasks(CUBDataset(root, split='train'))
        trainset = train_tasks.generate(task_id=task_id,
                                        use_species_names=True,
                                        transform=transform_train,
                                        random_label=config.get('random_label', ''))
        test_tasks = CUBTasks(CUBDataset(root, split='test'))
        testset = test_tasks.generate(task_id=task_id,
                                      use_species_names=True,
                                      transform=transform_test,
                                      random_label=config.get('random_label', ''))
    else:
        # iNat2018
        from dataset.inat import iNat2018Dataset
        tasks_map_file = os.path.join(root, 'inat2018', 'final_tasks_map.json')
        tasks_map = load_tasks_map(tasks_map_file)
        task_id = tasks_map[config.task_id - NUM_CUB]

        trainset = iNat2018Dataset(root, split='train', transform=transform_train, task_id=task_id, random_label=config.get('random_label', ''))
        testset = iNat2018Dataset(root, split='val', transform=transform_test, task_id=task_id, random_label=config.get('random_label', ''))
    trainset, testset = set_metadata(trainset, testset, config, 'cub_inat2018')
    return trainset, testset

@_add_dataset
def cub_inat2018_no_aug(root, config):
    """This meta-task is the concatenation of CUB-200 (first 25 tasks) and iNat (last 207 tasks).

    - The first 10 tasks are classification of the animal species inside one of 10 orders of birds in CUB-200
        (considering all orders except passeriformes).
    - The next 15 tasks are classification of species inside the 15 families of the order of passerifomes
    - The remaining 207 tasks are classification of the species inside each of 207 families in iNat

    As noted above, for CUB-200 10 taks are classification of species inside an order, rather than inside of a family
    as done in the iNat (recall order > family > species). This is done because CUB-200 has very few images
    in each family of bird (expect for the families of passeriformes). Hence, we go up step in the taxonomy and
    consider classification inside a orders and not families.
    """
    NUM_CUB = 25
    NUM_CUB_ORDERS = 10
    NUM_INAT = 207
    assert 0 <= config.task_id < NUM_CUB + NUM_INAT
    transform_train, transform_test = _get_transforms(augment=False)
    if 0 <= config.task_id < NUM_CUB:
        # CUB
        from dataset.cub import CUBTasks, CUBDataset
        tasks_map_file = os.path.join(root, 'cub/CUB_200_2011', 'final_tasks_map.json')
        tasks_map = load_tasks_map(tasks_map_file)
        # task_id = tasks_map[config.task_id]
        task_id = int(config.task_id)
        # CUB orders
        train_tasks = CUBTasks(CUBDataset(root, split='train'))
        trainset = train_tasks.generate(task_id=task_id,
                                        use_species_names=True,
                                        transform=transform_train)
        test_tasks = CUBTasks(CUBDataset(root, split='test'))
        testset = test_tasks.generate(task_id=task_id,
                                      use_species_names=True,
                                      transform=transform_test)
    else:
        # iNat2018
        from dataset.inat import iNat2018Dataset
        tasks_map_file = os.path.join(root, 'inat2018', 'final_tasks_map.json')
        tasks_map = load_tasks_map(tasks_map_file)
        task_id = tasks_map[config.task_id - NUM_CUB]

        trainset = iNat2018Dataset(root, split='train', transform=transform_train, task_id=task_id)
        testset = iNat2018Dataset(root, split='val', transform=transform_test, task_id=task_id)
    trainset, testset = set_metadata(trainset, testset, config, 'cub_inat2018')
    return trainset, testset


@_add_dataset
def cars(root, config):
    from dataset.cars import CarsDataset, CarsTasks
    transform_train, transform_test = _get_transforms()
    train_tasks = CarsTasks(CarsDataset(root, split='train'))
    trainset = train_tasks.generate(task_id=config.task_id,
                                    transform=transform_train)
    test_tasks = CarsTasks(CarsDataset(root, split='validation'))
    testset = test_tasks.generate(task_id=config.task_id,
                                  transform=transform_test)
    trainset, testset = set_metadata(trainset, testset, config, 'cars')
    return trainset, testset


def get_dataset(root, config=None):
    return _DATASETS[config.name](os.path.expanduser(root), config)
