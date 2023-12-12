import os
import shutil

from torchvision import datasets, transforms
import torchvision

import numpy as np
import pickle

class CIFARDataset(object):
    @staticmethod
    def get_cifar10_transform(name):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        if name == 'AutoAugment':
            policy = transforms.AutoAugmentPolicy.CIFAR10
            augmenter = transforms.AutoAugment(policy)
        elif name == 'RandAugment':
            augmenter = transforms.RandAugment()
        elif name == 'AugMix':
            augmenter = transforms.AugMix()
        else: raise f"Unknown augmentation method: {name}!"

        transform = transforms.Compose([
            augmenter,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        return transform

    @staticmethod
    def get_cifar10_train(path, transform=None, identity_transform=False):
        if transform is None:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2470, 0.2435, 0.2616]
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        if identity_transform:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2470, 0.2435, 0.2616]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
        return trainset

    @staticmethod
    def get_cifar10_test(path):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        testset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)
        return testset

    @staticmethod
    def get_cifar100_train(path, transform=None, identity_transform=False):
        if transform is None:
            mean=[0.507, 0.487, 0.441]
            std=[0.267, 0.256, 0.276]
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        if identity_transform:
            mean=[0.507, 0.487, 0.441]
            std=[0.267, 0.256, 0.276]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        trainset = torchvision.datasets.CIFAR100(root=path, train=True, download=True, transform=transform)
        return trainset

    @staticmethod
    def get_cifar100_test(path):
        mean=[0.507, 0.487, 0.441]
        std=[0.267, 0.256, 0.276]
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        testset = torchvision.datasets.CIFAR100(root=path, train=False, download=True, transform=transform_test)
        return testset

