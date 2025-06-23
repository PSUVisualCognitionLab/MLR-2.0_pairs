import sys
import time
#time.sleep(10800)
import argparse

parser = argparse.ArgumentParser(description="Training of MLR-2.0")
parser.add_argument("--load_prev", type=bool, default=False, help="Begin training from previous checkpoints")
parser.add_argument("--cuda_device", type=int, default=1, help="Which cuda device to use")
parser.add_argument("--cuda", type=bool, default=True, help="Cuda availability")
parser.add_argument("--folder", type=str, default='test', help="Where to store checkpoints in checkpoints/")
parser.add_argument("--dataset", type=str, default='mnist', help="Which dataset to train on/")
parser.add_argument("--z_dim", type=int, default=8, help="Size of the mVAE latent dimension")
parser.add_argument("--train_list", nargs='+', type=str, default=['mVAE', 'label_net', 'SVM'], help="Which components to train")
parser.add_argument("--wandb", type=bool, default=False, help="Track training with wandb")
parser.add_argument("--checkpoint_name", type=str, default='mVAE_checkpoint.pth', help="file name of checkpoint .pth")
parser.add_argument("--start_ep", type=int, default=1, help="what epoch to resume training")
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
    d = args.cuda_device

load = args.load_prev
dataset_name = args.dataset

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
SVT_bs = 25000  #batch size for training the spatial vision transformer
obj_dim = True if dataset_name == 'quickdraw' else False
# to resume training an existing model checkpoint, uncomment the following line with the checkpoints filename
if load is True:
    vae = load_checkpoint(f'{checkpoint_folder_path}/{args.checkpoint_name}', d, obj_dim)
    dimensions = load_dimensions(f'{checkpoint_folder_path}/{args.checkpoint_name}', d)
    print('checkpoint loaded')     

else:
    dimensions = [-1, -1, 128, args.z_dim]
    vae, dimensions = vae_builder(dimensions, obj_dim)

#vae = nn.DataParallel(vae)


# training datasets, the return loaders flag is False so the datasets can be concated in the dataloader
mnist_transforms = {'retina':True, 'colorize':True, 'rotate':False, 'scale':True} #set colorize false for cifar

mnist_test_transforms = {'retina':True, 'colorize':True, 'scale':True}
skip_transforms = {'skip':True, 'colorize':True}

#emnist_dataset = Dataset('emnist', mnist_transforms)
mnist_dataset = Dataset('emnist', mnist_test_transforms)
emnist_dataset = Dataset(dataset_name, mnist_transforms) #emnist

#emnist_test_dataset = Dataset('emnist', mnist_test_transforms, train= False)
mnist_test_dataset = Dataset(dataset_name, mnist_test_transforms, train= False)

#blocks
block_dataset = Dataset('square', {'colorize':True, 'retina':True, 'build_retina':False})
block_loader = block_dataset.get_loader(bs)

#blocks, labels = next(iter(block_loader))
#utils.save_image( blocks,
#           'testblock.png',
#          nrow=1, normalize=False)

# mnist size 4 drawings 2 maps
#emnist_skip = Dataset('emnist', skip_transforms)
mnist_skip = Dataset(dataset_name, skip_transforms)

train_loader_noSkip = emnist_dataset.get_loader(bs) # torch.utils.data.DataLoader(dataset=ConcatDataset([mnist_dataset, mnist_dataset, emnist_dataset]), batch_size=bs, shuffle=True,  drop_last= True)
train_loader_SVT = emnist_dataset.get_loader(SVT_bs)
emnist_loader = mnist_dataset.get_loader(bs)
#concat datasets and init dataloaders
#train_loader_noSkip = mnist_dataset.get_loader(bs)
#sample_loader_noSkip = mnist_dataset.get_loader(25)
test_loader_noSkip = torch.utils.data.DataLoader(dataset=ConcatDataset([mnist_dataset, mnist_dataset, emnist_dataset]), batch_size=bs, shuffle=True,  drop_last= True)
#mnist_skip = torch.utils.data.DataLoader(dataset=ConcatDataset([block_dataset, mnist_skip]), batch_size=bs, shuffle=True,  drop_last= True)
mnist_skip = mnist_skip.get_loader(bs)

#add colorsquares dataset to training
vae.to(device)

dataloaders = [train_loader_noSkip, emnist_loader, mnist_skip, test_loader_noSkip, None, block_loader]
label_dataloaders = [emnist_loader, emnist_loader, mnist_skip, test_loader_noSkip, None, block_loader]
SVT_dataloaders = [train_loader_SVT, emnist_loader, mnist_skip, test_loader_noSkip, None, block_loader]

print(f'Training: {args.train_list}')
epoch_count = 180
#train mVAE
if 'mVAE' in args.train_list:
    print('Training: mVAE')
    train_mVAE(dataloaders, vae, epoch_count, folder_name, args.wandb, args.start_ep, dimensions)

#train_labels
if 'label_net' in args.train_list:
    print('Training: label networks')
    
    train_labelnet(label_dataloaders, vae, 15, folder_name)

#train_classifiers
if 'SVM' in args.train_list:
    print('Training: classifiers')
    train_classifiers(SVT_dataloaders, vae, folder_name)
