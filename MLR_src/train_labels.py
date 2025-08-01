from MLR_src.label_network import vae_shape_labels, vae_color_labels, vae_object_labels, train_labels, optim
import torch
import os
from training_constants import training_components

def train_labelnet(dataloaders, vae, epoch_count, checkpoint_folder):
    if not os.path.exists('training_samples/'):
        os.mkdir('training_samples/')
    
    if not os.path.exists(f'training_samples/{checkpoint_folder}/'):
        os.mkdir(f'training_samples/{checkpoint_folder}/')
    
    sample_folder_path = f'training_samples/{checkpoint_folder}/label_net_samples/'
    if not os.path.exists(sample_folder_path):
        os.mkdir(sample_folder_path)

    optimizer_shapelabels= optim.Adam(vae_shape_labels.parameters())
    optimizer_colorlabels= optim.Adam(vae_color_labels.parameters())
    optimizer_objectlabels= optim.Adam(vae_object_labels.parameters())

    label_nets = [vae_shape_labels, vae_object_labels, vae_color_labels]
    optimizers = [optimizer_shapelabels, optimizer_objectlabels, optimizer_colorlabels]
    components = ['shape', 'object', 'color']

    for i in range(len(label_nets)):
        label_net, optimizer = label_nets[i], optimizers[i]
        whichcomponent = components[i]
        for epoch in range (1,epoch_count):
            #train_labels(vae, label_net, whichcomponent, epoch, train_loader, optimizer, folder_path):
            whichloader =  training_components[whichcomponent][0][0] 
            train_labels(vae, label_net, whichcomponent, epoch, dataloaders[whichloader], optimizer, sample_folder_path)
            
    checkpoint =  {
            'state_dict_shape_labels': vae_shape_labels.state_dict(),
            'state_dict_color_labels': vae_color_labels.state_dict(),
            'state_dict_object_labels': vae_object_labels.state_dict(),


            'optimizer_shape' : optimizer_shapelabels.state_dict(),
            'optimizer_color': optimizer_colorlabels.state_dict(),
            'optimizer_object': optimizer_objectlabels.state_dict(),

                }
    torch.save(checkpoint, f'checkpoints/{checkpoint_folder}/label_network_checkpoint.pth')