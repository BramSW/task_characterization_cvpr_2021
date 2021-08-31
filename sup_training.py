import torch
import numpy as np
from utils import get_error, AverageMeter

def eval_model(model, train_dataset, test_dataset, num_samples=10000):
    train_loader, num_classes = _get_loader(train_dataset, num_samples=num_samples)
    loss_fn = torch.nn.CrossEntropyLoss()
    # Edit the model to match # classes

    test_loader, num_classes = _get_loader(test_dataset, num_samples=num_samples)
    print("Validation")
    metrics = AverageMeter()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            loss = loss_fn(output, target)
            error = get_error(output, target)
            metrics.update(n=data.size(0), loss=loss.item(), error=error)
    mean_error = metrics.avg['error']
    print("Error:", mean_error)
    return mean_error


def transfer_model(model, train_dataset, test_dataset, num_samples=10000):
    train_loader, num_classes = _get_loader(train_dataset, num_samples=num_samples)
    loss_fn = torch.nn.CrossEntropyLoss()
    # Edit the model to match # classes
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes).cuda()
    model.layers[-1] = model.fc
    model.reset_classifier()

    # Initial linear phase
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4, weight_decay=5e-4)
    for epoch_i in range(16):
        print("Linear Fit Epoch: {}/16".format(epoch_i))
        metrics = AverageMeter()
        for data, target in train_loader:
            data = data.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            error = get_error(output, target)
            loss.backward()
            optimizer.step()
            metrics.update(n=data.size(0), loss=loss.item(), error=error)
        print(f"[epoch {epoch_i}]: " + "\t".join(f"{k}: {v}" for k, v in metrics.avg.items()))

    test_loader, num_classes = _get_loader(test_dataset, num_samples=num_samples)
    print("Validation")
    metrics = AverageMeter()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            loss = loss_fn(output, target)
            error = get_error(output, target)
            metrics.update(n=data.size(0), loss=loss.item(), error=error)
    mean_error = metrics.avg['error']
    print("Error:", mean_error)
    return mean_error


def train_model(model, dataset, num_samples=10000):
    loader, num_classes = _get_loader(dataset, num_samples=num_samples)
    loss_fn = torch.nn.CrossEntropyLoss()
    # Edit the model to match # classes
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes).cuda()
    model.layers[-1] = model.fc
    model.reset_classifier()

    # Initial linear phase
    optimizer = torch.optim.Adam(model.classifier.parameters())
    for epoch_i in range(10):
        print("Linear Fit Epoch: {}/10".format(epoch_i))
        metrics = AverageMeter()
        for data, target in loader:
            data = data.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            error = get_error(output, target)
            loss.backward()
            optimizer.step()
            metrics.update(n=data.size(0), loss=loss.item(), error=error)
        print(f"[epoch {epoch_i}]: " + "\t".join(f"{k}: {v}" for k, v in metrics.avg.items()))

    # Full fine-tuning phase
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=5e-4, lr=1e-3)
    for epoch_i in range(60):
        print("Finetuning Epoch: {}/60".format(epoch_i))
        metrics = AverageMeter()
        for data, target in loader:
            data = data.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            error = get_error(output, target)
            loss.backward()
            optimizer.step()
            metrics.update(n=data.size(0), loss=loss.item(), error=error)
        print(f"[epoch {epoch_i}]: " + "\t".join(f"{k}: {v}" for k, v in metrics.avg.items()))
        if epoch_i == 39: optimizer.param_groups[0]['lr'] *= 0.1

    return model


def _get_loader(trainset, testset=None, batch_size=128, num_workers=6, num_samples=10000, drop_last=True):
    if getattr(trainset, 'is_multi_label', False):
        raise ValueError("Multi-label datasets not supported")
    if hasattr(trainset, 'labels'):
        labels = trainset.labels
    elif hasattr(trainset, 'targets'):
        labels = trainset.targets
    else:
        labels = list(trainset.tensors[1].cpu().numpy())
    num_classes = int(max(labels) + 1)
    class_count = np.eye(num_classes)[labels].sum(axis=0)
    weights = 1. / class_count[labels] / num_classes
    weights /= weights.sum()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=num_samples)
    # No need for mutli-threaded loading if everything is already in memory,
    # and would raise an error if TensorDataset is on CUDA
    num_workers = num_workers if not isinstance(trainset, torch.utils.data.TensorDataset) else 0
    trainloader = torch.utils.data.DataLoader(trainset, sampler=sampler, batch_size=batch_size,
                                              num_workers=num_workers, drop_last=drop_last)

    if testset is None:
        return trainloader, num_classes
    else:
        raise NotImplementedError
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, pin_memory=True, shuffle=False,
                                                 num_workers=num_workers)
        return trainloader, testloader

