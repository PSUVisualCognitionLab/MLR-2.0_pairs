# this function  will perform a full stack test of the model, taking in a given object,  classifying it as a kind of object, 
# and then generating that kind of object with the label network and then classifying its own output


import sys
import time


import argparse

parser = argparse.ArgumentParser(description="Training of MLR-2.0")
#parser.add_argument("--load_prev", type=bool, default=False, help="Begin training from previous checkpoints")
parser.add_argument("--cuda", type=bool, default=True, help="Cuda availability")
parser.add_argument("--cuda_device", type=int, default=1, help="Which cuda device to use")
parser.add_argument("--folder", type=str, default='test', help="Where to store checkpoints in checkpoints/")
# VVV defines which components are trained
parser.add_argument("--components", nargs='+', type=str, default=['shape', 'color', 'retinal', 'object', 'skip_cropped', 'cropped', 'retinal_object', 'cropped_object'], help="Which components to train")
#parser.add_argument("--z_dim", type=int, default=8, help="Size of the mVAE latent dimension")
#parser.add_argument("--train_list", nargs='+', type=str, default=['mVAE', 'label_net', 'SVM'], help="Which models to train")
#parser.add_argument("--wandb", type=bool, default=False, help="Track training with wandb")
parser.add_argument("--checkpoint_name", type=str, default='mVAE_checkpoint.pth', help="file name of checkpoint .pth")
#parser.add_argument("--start_ep", type=int, default=1, help="what epoch to resume training")
#parser.add_argument("--end_ep", type=int, default=300, help="what epoch to train to")
#parser.add_argument("--batch_size", nargs='+', type=int, default=['mVAE', 'label_net', 'SVM'], help="Which components to train")
args = parser.parse_args()



# prerequisites
import torch
import os
from MLR_src.mVAE import load_checkpoint, vae_builder, load_dimensions
#from torch.utils.data import DataLoader, ConcatDataset
from MLR_src.dataset_builder import Dataset
from MLR_src.train_mVAE import train_mVAE
from MLR_src.label_network import train_labelnet, load_checkpoint_shapelabels, load_checkpoint_colorlabels
from MLR_src.train_classifiers import train_classifiers, classifier_test
from training_constants import training_datasets, training_components
from itertools import cycle
from torchvision import utils
from torchvision.utils import save_image
import numpy as np

import joblib

folder_name = args.folder
checkpoint_folder_path = f'checkpoints/{folder_name}/' # the output folder for the trained model versions


if args.cuda is True:
    d = args.cuda_device   #which cuda GPU?


print(f'Device: {d}')


if torch.cuda.is_available():
    device = torch.device(f'cuda:{d}')
    torch.cuda.set_device(d)
    print('CUDA')
else:
    device = 'cpu'
    print('CUDA not available')


bs=1000   #batch size for training the main VAE
SVM_bs = 25000  #batch size for training the spatial vision transformer
obj_latent_flag = True   #this flag determines whether the VAE has an obj latent space


vae = load_checkpoint(f'{checkpoint_folder_path}{args.checkpoint_name}', d, obj_latent_flag)
dimensions = load_dimensions(f'{checkpoint_folder_path}/{args.checkpoint_name}', d)
print('VAE checkpoint loaded')     

clf_ess = joblib.load(f'{checkpoint_folder_path}/ess.joblib')
#    pred_sss,  coreport = classifier_test(vae, 'shape', clf_sss, dataloaders['emnist-map'],'emnist','shape', 1)
print('ess Classifier  loaded  (emnist classification)')   #this should be high
    


#vae_shape_labels= VAEshapelabels(xlabel_dim=s_classes, hlabel_dim=20,  zlabel_dim=8)
#vae_object_labels= VAEshapelabels(xlabel_dim=s_classes, hlabel_dim=20,  zlabel_dim=8)
#vae_color_labels= VAEcolorlabels(xlabel_dim=10, hlabel_dim=7,  zlabel_dim=8)

shapenetwork = load_checkpoint_shapelabels(f'{checkpoint_folder_path}label_network_checkpoint.pth')



#vae = nn.DataParallel(vae)

dataloaders = {}
SVM_dataloaders = {}
weighted_components = [] #specifies the order/frequency the model latents will be trained


# init dataloaders for mVAE and label training by making sure the data sets for each model component are added
# model components are the latent spaces, like shape, color, etc   Each component also has a specific list of transforms

for component in args.components:
    weight = training_components[component][1]
    weighted_components += [component] * weight
    for dataset in training_components[component][0]:
        dataset_name = dataset.split('-')[0]
        dataset_transforms = training_datasets[dataset]  #load the transforms for this dataset
        dataloader = cycle(Dataset(dataset_name, dataset_transforms).get_loader(bs//len(training_components[component][0])))
        dataloaders[dataset] = iter(dataloader)

# init dataloaders for SVM training
for component in args.components:
    for dataset in training_components[component][0]:
        dataset_name = dataset.split('-')[0]
        dataset_transforms = training_datasets[dataset]   #load the transforms for this dataset
        SVM_dataloaders[dataset] = cycle(Dataset(dataset_name, dataset_transforms).get_loader(SVM_bs))

vae.to(device)
vae.eval()


####################################################################################################
#Everything is loaded, now time to run some tests
#First is shape classification

#print(dataloaders.keys())
dataloader = dataloaders['emnist-map']
sample, labels = next(dataloader)   #get one batch from the data loader

sample = sample[1].to(device)  # get the cropped versions
shapelabels = labels[0]
colorlabels = labels[1]

passin = 'digit'   #ensure that we are using the shape map not the object map 
whichcomponent = 'shape'

z = vae.activations(sample, False, None, passin)[whichcomponent]  #get the activations from the shape map
pred = clf_ess.predict(z.cpu().detach().numpy())

accurate =  np.sum(pred-shapelabels.numpy() == 0)
print(f'accuracy of the shape classifier is {accurate/len(pred)}')



#output_img = torch.cat([sample.view(sample_size, 3, imgsize, imgsize)[:25], recon.view(sample_size, 3, imgsize, imgsize)[:25], skip.view(sample_size, 3, imgsize, imgsize)[:25],
#                       shape.view(sample_size, 3, imgsize, imgsize)[:25], color.view(sample_size, 3, imgsize, imgsize)[:25]], 0)


#sample_size =50
#imgsize = 28
#output_img = torch.cat([sample.view(sample_size, 3, imgsize, imgsize)])
#utils.save_image(output_img,f'samples.png',
#            nrow=1, normalize=False)
   

#print(sample[1].shape)
#print(labels[1])

#print(len(sample))
