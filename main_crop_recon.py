import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import os, sys
import argparse
import pickle
from datetime import datetime
import numpy as np
# from torchvision import models
from torchvision.transforms import InterpolationMode
from PIL import Image

from models_cifar import *
from random_select import CoresetSelection
from data import CIFARDataset
from index_data import IndexDataset
import random
def denormalize_cifar100(image_tensor, use_fp16=False):
    '''
    convert floats back to input
    '''
    if use_fp16:
        mean = np.array([0.5070751592371323, 0.48654887331495095, 0.4409178433670343], dtype=np.float16)
        std = np.array([0.2673342858792401, 0.2564384629170883, 0.27615047132568404], dtype=np.float16)
    else:
        mean = np.array([0.5070751592371323, 0.48654887331495095, 0.4409178433670343])
        std = np.array([0.2673342858792401, 0.2564384629170883, 0.27615047132568404])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor        
def save_images(args, images, targets):
    for id in range(images.shape[0]):
        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()

        if not os.path.exists(args.syn_data_path):
            os.mkdir(args.syn_data_path)

        # save into separate folders
        dir_path = '{}/new{:03d}'.format(args.syn_data_path, class_id)
        place_to_store = dir_path +'/class{:03d}_id{:03d}.jpg'.format(class_id,id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10,CIFAR100 Training')
parser.add_argument('--dataset', default='cifar100', type = str,
                        help='whether to store best images')
parser.add_argument('--data_dir', default='/home/st2/Model_Compression', type = str,
                        help='whether to store best images')
parser.add_argument('--batch_size', default=300, type = float,
                        help='whether to store best images')
parser.add_argument('--coreset_ratio', default=1, type = float,
                        help='whether to store best images')
parser.add_argument('--crop_num', default=5, type = int,
                        help='whether to store best images')
parser.add_argument('--save_num', default=4, type = int,
                        help='whether to store best images')
parser.add_argument('--pretrain_path', type=str, default='/home/st2/Model_Compression/SRe2L-main/recover/pretrain_models/renet18-cifar100.pth',
                        help='arch name from pretrained torchvision models')
parser.add_argument('--syn_data_path', type=str, default='/home/st2/Model_Compression/SRe2L-main/RDED/syn_data_n1/',
                        help='arch name from pretrained torchvision models')
args = parser.parse_args()

print(f'Dataset: {args.dataset}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = args.data_dir

print(f'Data dir: {data_dir}')

if args.dataset == 'cifar10':
    trainset = CIFARDataset.get_cifar10_train(data_dir)
elif args.dataset == 'cifar100':
    trainset = CIFARDataset.get_cifar100_train(data_dir)
# sort data with class
sorted_indices = np.argsort(trainset.targets)
trainset.data = trainset.data[sorted_indices]
trainset.targets = np.array(trainset.targets)[sorted_indices]
######################### Coreset Selection #########################
coreset_ratio = args.coreset_ratio
total_numbers = len(trainset)
# 每500个数中随机取300个
batch_size = 500
sample_size = int(coreset_ratio*batch_size)
# 生成所有数的索引
all_indices = list(range(total_numbers))
# 遍历每500个数的组
selected_indices = []
for i in range(0, total_numbers, batch_size):
    # 在当前组中随机选择300个索引
    batch_indices = random.sample(range(i, min(i + batch_size, total_numbers)), sample_size)
    selected_indices.extend(batch_indices)

trainset = torch.utils.data.Subset(trainset, selected_indices)
print(len(trainset))
######################### Coreset Selection end #########################

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=False, num_workers=2)
crop_transforms = transforms.RandomResizedCrop(32, scale=(0.4, 1),
                                        interpolation=InterpolationMode.BILINEAR)
if args.dataset in ['cifar10', 'svhn', 'cinic10']:
    num_classes=10
else:
    num_classes=100

model =  resnet18_cifar(pretrained=True, path = args.pretrain_path)

model = model.to(device)

criterion = nn.CrossEntropyLoss(reduction='none')

model.eval()
with torch.no_grad():
    remain_path = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        patches_class = []
        score_class = []
        print('class ' + str(batch_idx)+ ' start')
        for i in range (args.crop_num):
            transformed_image = crop_transforms(inputs)
            outputs, _ = model(transformed_image)
            patches_class.append(transformed_image)
            score_class.append(-criterion(outputs, targets))
        patches_class = torch.stack(patches_class, dim=0)
        score_class = torch.stack(score_class, dim=0)
        index_1 = score_class.max(0)[1]
        score_class_select_1 = score_class.max(0)[0]
        index_2 = score_class_select_1.topk(50)[1]
        score_class_select_2 = score_class_select_1.topk(50)[0]
        save_patch = patches_class[index_1[index_2], index_2,:,:,:]
        # Resize the last two dimensions to [16, 16]
        # resized_tensor = torch.nn.functional.interpolate(save_patch, size=(16, 16), mode='bilinear', align_corners=False)
        # Check the size after resizing
        # print("Resized Tensor shape:", resized_tensor.shape)
        # Reshape to [50, 4, 3, 16, 16] and then permute dimensions
        # reshaped_tensor = resized_tensor.view(50, 2, 2, 3, 16, 16)

        # 沿着最后两个维度进行拼接
        # concatenated_tensor = reshaped_tensor.permute(0, 3, 1, 4, 2, 5).contiguous().view(50, 3, 32, 32)

        best_inputs = denormalize_cifar100(save_patch)#(concatenated_tensor)
        save_images(args, best_inputs, targets)



            
