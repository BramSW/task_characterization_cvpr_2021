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


import torch.utils.model_zoo as model_zoo

import torchvision.models.resnet as resnet
import torch

from embed_orig_task2vec import ProbeNetwork

_MODELS = {}


def _add_model(model_fn):
    _MODELS[model_fn.__name__] = model_fn
    return model_fn


class ResNet(resnet.ResNet, ProbeNetwork):

    def __init__(self, block, layers, num_classes=1000, gaussian_init=False):
        super(ResNet, self).__init__(block, layers, num_classes)
        # Saves the ordered list of layers. We need this to forward from an arbitrary intermediate layer.
        self.layers = [
            self.conv1, self.bn1, self.relu,
            self.maxpool, self.layer1, self.layer2,
            self.layer3, self.layer4, self.avgpool,
            torch.nn.Flatten(), self.fc
        ]
        if gaussian_init:
            sd = self.state_dict()
            for key,tensor in sd.items():
                sd[key] = torch.randn(tensor.shape)
                self.load_state_dict(sd)

    @property
    def classifier(self):
        return self.fc

    def reset_classifier(self):
        self.fc.weight.data = torch.zeros_like(self.fc.weight.data).cuda()
        self.fc.bias.data = torch.zeros_like(self.fc.bias.data).cuda()

    # Modified forward method that allows to start feeding the cached activations from an intermediate
    # layer of the network
    def forward(self, x, start_from=0):
        """Replaces the default forward so that we can forward features starting from any intermediate layer."""
        for layer in self.layers[start_from:]:
            x = layer(x)
        return x

def make_encoder(net):
    return torch.nn.Sequential(*net.layers[:-1])

@_add_model
def resnet18(pretraining='imagenet', num_classes=1000):
    """Constructs a ResNet-18 model.
    Args:
        pretraining (str): 'imagenet' 'places365' ' or 'random' (add moco later)
    """
    model = ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    if pretraining=='imagenet':
        state_dict = model_zoo.load_url(resnet.model_urls['resnet18'])
    elif pretraining=='places365':
        state_dict = torch.load('/data/bw462/torch_model_zoo/checkpoints/resnet18_places365.pth.tar')['state_dict']
        state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
    elif pretraining=='random':
        return model
    else:
        # Don't have the pretraining
        raise NotImplementedError
    state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
    model.load_state_dict(state_dict, strict=False)
    return model

@_add_model
def resnet34(pretraining='imagenet', num_classes=1000):
    """Constructs a ResNet-34 model.
    Args:
        pretraining (str): 'imagenet' 'places365' ' or 'random' (add moco later)
    """
    model = ResNet(resnet.BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    if pretraining=='imagenet':
        state_dict = model_zoo.load_url(resnet.model_urls['resnet34'])
    elif pretraining=='places365':
        raise NotImplementedError
    elif pretraining=='random':
        return model
    else:
        # Don't have the pretraining
        raise NotImplementedError
    state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
    model.load_state_dict(state_dict, strict=False)
    return model

# This and resnet18_places / resnet18_random are for pseudolabeling, different from replacing head
@_add_model
def resnet18_imagenet(pretraining=None, moco=None, gaussian_init=None, num_classes=None):
    model: ProbeNetwork = ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes=1000)
    state_dict = model_zoo.load_url(resnet.model_urls['resnet18'])
    model.load_state_dict(state_dict)
    return model

@_add_model
def resnet18_places365(pretraining=None, moco=None, gaussian_init=None, num_classes=None):
    model: ProbeNetwork = ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes=365)
    state_dict = torch.load('/data/bw462/torch_model_zoo/checkpoints/resnet18_places365.pth.tar')['state_dict']
    state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model

@_add_model
def resnet18_random(pretraining=None, moco=None, gaussian_init=None, num_classes=None):
    model: ProbeNetwork = ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes=1000)
    return model

@_add_model
def resnet50(pretraining='imagenet', num_classes=1000):
    """Constructs a ResNet-50 model.
    Args:
        pretraining (str): 'imagenet' or 'random' or 'moco'
    """
    model = ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    if pretraining=='imagenet':
        state_dict = model_zoo.load_url(resnet.model_urls['resnet50'])
    elif pretraining=='places365':
        raise NotImplementedError
    elif pretraining=='random':
        return model
    elif pretraining=='moco':
        state_dict = torch.load('/home/bw462/t2v/aws-cv-task2vec/moco_v2_800ep_pretrain.pth.tar')['state_dict']
        state_dict = {k.replace('module.','').replace('encoder_q.', ''):v for k,v in state_dict.items()}
    else:
        # Don't have the pretraining
        raise NotImplementedError
    state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
    model.load_state_dict(state_dict, strict=False)
    return model


def get_model(model_name, pretraining=None, num_classes=1000):
    try:
        return _MODELS[model_name](pretraining=pretraining, num_classes=num_classes)
    except KeyError:
        raise ValueError(f"Architecture {model_name} not implemented.")
