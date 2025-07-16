import sys
import time
#time.sleep(10800)
import argparse

parser = argparse.ArgumentParser(description="Training of MLR-2.0")
parser.add_argument("--load_prev", type=bool, default=False, help="Begin training from previous checkpoints")
parser.add_argument("--cuda_device", type=int, default=1, help="Which cuda device to use")
parser.add_argument("--cuda", type=bool, default=True, help="Cuda availability")
parser.add_argument("--folder", type=str, default='test', help="Where to store checkpoints in checkpoints/")
parser.add_argument("--components", nargs='+', type=str, default=['shape', 'color', 'retinal', 'object', 'skip_cropped', 'retinal_object'], help="Which components to train")
parser.add_argument("--z_dim", type=int, default=8, help="Size of the mVAE latent dimension")
parser.add_argument("--train_list", nargs='+', type=str, default=['mVAE', 'label_net', 'SVM'], help="Which models to train")
parser.add_argument("--wandb", type=bool, default=False, help="Track training with wandb")
parser.add_argument("--checkpoint_name", type=str, default='mVAE_checkpoint.pth', help="file name of checkpoint .pth")
parser.add_argument("--start_ep", type=int, default=1, help="what epoch to resume training")
parser.add_argument("--end_ep", type=int, default=300, help="what epoch to train to")
#parser.add_argument("--batch_size", nargs='+', type=int, default=['mVAE', 'label_net', 'SVM'], help="Which components to train")
args = parser.parse_args()

# prerequisites
import torch
import os
from MLR_src.mVAE import load_checkpoint, vae_builder, load_dimensions
from torch.utils.data import DataLoader, ConcatDataset
from MLR_src.dataset_builder import Dataset
from MLR_src.train_mVAE import train_mVAE
from MLR_src.train_labels import train_labelnet
from MLR_src.train_classifiers import train_classifiers
from torchvision import datasets, transforms, utils
import torch.nn as nn

folder_name = args.folder
#torch.set_default_dtype(torch.float64)
checkpoint_folder_path = f'checkpoints/{folder_name}/' # the output folder for the trained model versions

if not os.path.exists('training_samples/'):
    os.mkdir('training_samples/')

if not os.path.exists('checkpoints/'):
    os.mkdir('checkpoints/')

if not os.path.exists(checkpoint_folder_path):
    os.mkdir(checkpoint_folder_path)

if not os.path.exists(f'training_samples/{folder_name}/'):
    os.mkdir(f'training_samples/{folder_name}/')

if args.cuda is True:
    d = args.cuda_device   #which cuda GPU?

load = args.load_prev    #use a previous checkpoint?

print(f'Device: {d}')
print(f'Load: {load}')

if torch.cuda.is_available():
    device = torch.device(f'cuda:{d}')
    torch.cuda.set_device(d)
    print('CUDA')
else:
    device = 'cpu'
    print('CUDA not available')

bs=100   #batch size for training the main VAE
SVM_bs = 25000  #batch size for training the spatial vision transformer
obj_latent_flag = True   #this flag determines whether the VAE has an obj latent space
training_datasets = {'emnist-map': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True},
                     'emnist-skip': {'retina':False, 'colorize':True, 'rotate':True, 'scale':True, 'skip': True},
                     'mnist-map': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True},
                     'mnist-skip': {'retina':False, 'colorize':True, 'rotate':True, 'scale':True, 'skip': True},
                     'quickdraw': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True},
                     'cifar10': {'retina':True, 'rotate':False, 'scale':True},
                     'square': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True},
                     'line': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True},
                     'fashion_mnist': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True}}

training_components = {'shape': [['emnist-map', 'mnist-map'], 3], # shape map, weighted 3 times in training etc
                       'color': [['emnist-map', 'mnist-map'], 3], # color map
                       'object': [['quickdraw'], 3], # map for quickdraw
                       'cropped': [['emnist-map', 'mnist-map'], 3], # shape and color recon
                       'cropped_object': [['quickdraw'], 3], # object and color recon
                       'skip_cropped': [['emnist-skip', 'mnist-skip'], 1], # mnist/emnist skip connection
                       'retinal': [['emnist-map', 'mnist-map'], 1], # retinal, scale, location
                       'retinal_object': [['quickdraw'], 1]} # retinal, scale, location, object

if load is True:
    vae = load_checkpoint(f'{checkpoint_folder_path}/{args.checkpoint_name}', d, obj_latent_flag)
    dimensions = load_dimensions(f'{checkpoint_folder_path}/{args.checkpoint_name}', d)
    print('checkpoint loaded')     

else:
    dimensions = [-1, -1, 128, args.z_dim]
    vae, dimensions = vae_builder(dimensions, obj_latent_flag)

#vae = nn.DataParallel(vae)

dataloaders = {}
SVM_dataloaders = {}
weighted_components = [] #specifies the order/frequency the model latents will be trained

# init dataloaders for mVAE, label training
for component in args.components:
    weight = training_components[component][1]
    components += [component] * weight
    for dataset in training_components[component][0]:
        dataset_name = dataset.split('-')[0]
        dataset_transforms = training_datasets[dataset]
        dataloader = Dataset(dataset_name, dataset_transforms).get_loader(bs//len(training_components[component][0]))
        dataloaders[dataset] = iter(dataloader)

# init dataloaders for SVM training
for component in args.components:
    for dataset in training_components[component][0]:
        dataset_name = dataset.split('-')[0]
        dataset_transforms = training_datasets[dataset]
        SVM_dataloaders[dataset] = Dataset(dataset_name, dataset_transforms).get_loader(SVM_bs)

vae.to(device)

print(dataloaders.keys())

print(f'Training: {args.train_list}')
epoch_count = args.end_ep

#train mVAE
if 'mVAE' in args.train_list:
    print('Training: mVAE')
    train_mVAE(dataloaders, components, vae, epoch_count, folder_name, args.wandb, args.start_ep, dimensions)

#train_labels
if 'label_net' in args.train_list:
    print('Training: label networks')
    train_labelnet(dataloaders, vae, 15, folder_name)

#train_classifiers
if 'SVM' in args.train_list:
    print('Training: classifiers')
    train_classifiers(SVM_dataloaders, vae, folder_name)
