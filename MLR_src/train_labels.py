from MLR_src.label_network import *
import torch
import os

def train_labelnet(dataloaders, vae, epoch_count, checkpoint_folder):
    optimizer = optim.Adam(vae.parameters())
    if not os.path.exists('training_samples/'):
        os.mkdir('training_samples/')
    
    if not os.path.exists(f'training_samples/{checkpoint_folder}/'):
        os.mkdir(f'training_samples/{checkpoint_folder}/')
    
    sample_folder_path = f'training_samples/{checkpoint_folder}/label_net_samples/'
    if not os.path.exists(sample_folder_path):
        os.mkdir(sample_folder_path)

    optimizer_shapelabels= optim.Adam(vae_shape_labels.parameters())
    optimizer_colorlabels= optim.Adam(vae_color_labels.parameters())
    for epoch in range (1,epoch_count):
        train_labels(vae, epoch, dataloaders[0], optimizer_shapelabels, optimizer_colorlabels, sample_folder_path)
        
    checkpoint =  {
            'state_dict_shape_labels': vae_shape_labels.state_dict(),
            'state_dict_color_labels': vae_color_labels.state_dict(),

            'optimizer_shape' : optimizer_shapelabels.state_dict(),
            'optimizer_color': optimizer_colorlabels.state_dict(),

                }
    torch.save(checkpoint, f'checkpoints/{checkpoint_folder}/label_network_checkpoint.pth')