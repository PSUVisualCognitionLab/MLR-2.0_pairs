import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torchvision.utils import save_image
from sklearn.metrics import classification_report, confusion_matrix
from training_constants import training_components, text_to_tensor
import torch.nn.functional as F
import os

from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
from joblib import dump, load
import copy


bs = 100
s_classes = 36
c_classes = 10

def train_labelnet(dataloaders, vae, epoch_count, z_dim, checkpoint_folder):
    if not os.path.exists('training_samples/'):
        os.mkdir('training_samples/')
    
    if not os.path.exists(f'training_samples/{checkpoint_folder}/'):
        os.mkdir(f'training_samples/{checkpoint_folder}/')
    
    sample_folder_path = f'training_samples/{checkpoint_folder}/label_net_samples/'
    if not os.path.exists(sample_folder_path):
        os.mkdir(sample_folder_path)
    
    vae_shape_labels= VAEshapelabels(xlabel_dim=s_classes, hlabel_dim=20,  zlabel_dim=z_dim)
    vae_object_labels= VAEshapelabels(xlabel_dim=s_classes, hlabel_dim=20,  zlabel_dim=z_dim)
    vae_color_labels= VAEcolorlabels(xlabel_dim=10, hlabel_dim=7,  zlabel_dim=z_dim)

    optimizer_shapelabels= optim.Adam(vae_shape_labels.parameters())
    optimizer_colorlabels= optim.Adam(vae_color_labels.parameters())
    optimizer_objectlabels= optim.Adam(vae_object_labels.parameters())

    label_nets = {'shape':[vae_shape_labels, optimizer_shapelabels],
                  'object': [vae_object_labels, optimizer_objectlabels],
                  'color': [vae_color_labels, optimizer_colorlabels]}

    for whichcomponent in label_nets:
        label_net, optimizer = label_nets[whichcomponent]
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

            'z_dim': z_dim
                }
    torch.save(checkpoint, f'checkpoints/{checkpoint_folder}/label_network_checkpoint.pth')

# shape label network
class VAEshapelabels(nn.Module):
    def __init__(self, xlabel_dim, hlabel_dim,  zlabel_dim):
        super(VAEshapelabels, self).__init__()

        # encoder part
        self.fc1label = nn.Linear(xlabel_dim, hlabel_dim)
        self.fc21label= nn.Linear(hlabel_dim,  zlabel_dim) #mu shape
        self.fc22label = nn.Linear(hlabel_dim, zlabel_dim) #log-var shape


    def sampling_labels (self, mu, log_var, n=1):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) * n
        return mu + eps * std

    def forward(self, x_labels, n):
        h = F.relu(self.fc1label(x_labels))
        mu_shape_label = self.fc21label(h)
        log_var_shape_label=self.fc22label(h)
        z_shape_label = self.sampling_labels(mu_shape_label, log_var_shape_label, n)
        return  z_shape_label

# color label network
class VAEcolorlabels(nn.Module):
    def __init__(self, xlabel_dim, hlabel_dim, zlabel_dim):
        super(VAEcolorlabels, self).__init__()

        # encoder part
        self.fc1label = nn.Linear(xlabel_dim, hlabel_dim)
        self.fc21label = nn.Linear(hlabel_dim, zlabel_dim)  # mu color
        self.fc22label = nn.Linear(hlabel_dim, zlabel_dim)  # log-var color

    def sampling_labels(self, mu, log_var, n=1):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) * n
        return mu + eps * std

    def forward(self, x_labels, n=1):
        device = next(self.parameters()).device
        x_labels.to(device)
        h = F.relu(self.fc1label(x_labels))
        mu_shape_label = self.fc21label(h)
        log_var_shape_label = self.fc22label(h)
        z_color_label = self.sampling_labels(mu_shape_label, log_var_shape_label, n)
        return  z_color_label

def load_checkpoint_labels(filepath, label_type, d=0):
    assert label_type in ["shape", "color", "object"], f"label_type: {label_type} is invalid, must be one of: shape, color, object"
    print(f"loading {label_type} label network")

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{d}')
        torch.cuda.set_device(d)
    else:
        device = 'cpu'

    checkpoint = torch.load(filepath)
    z_dim = checkpoint['z_dim']

    label_net_dict = {"shape":VAEshapelabels(xlabel_dim=s_classes, hlabel_dim=20,  zlabel_dim=z_dim), 
                      "object": VAEshapelabels(xlabel_dim=s_classes, hlabel_dim=20,  zlabel_dim=z_dim),
                      "color": VAEcolorlabels(xlabel_dim=10, hlabel_dim=7,  zlabel_dim=z_dim)}
    
    label_net = label_net_dict[label_type]
    label_net.load_state_dict(checkpoint[f'state_dict_{label_type}_labels'])
    for parameter in label_net.parameters():
        parameter.requires_grad = False
    label_net.eval().to(device)
    return label_net

def loss_label(label_act,image_act):

    criterion=nn.MSELoss(reduction='sum')
    e=criterion(label_act,image_act)

    return e

def train_labels(vae, label_net, whichcomponent, epoch, train_loader, optimizer, folder_path):    
    device = next(vae.parameters()).device
    train_loss = 0

    label_net.train().to(device)

    dataiter = train_loader

    max_iter = 100
    for i , j in enumerate(train_loader):
        optimizer.zero_grad()

        image, labels = next(dataiter)
        labels_for_shape=labels[0].clone() # shape or object, depending on dataset
        labels_for_color=labels[1].clone()
              
        image = image[1].cuda() # dataset must be retinal for this(?)

        if whichcomponent == 'color':
            labels_color = labels_for_color  # get the color label from the dataloader
            labels_color = labels_color.to(device)
            color_oneHot = F.one_hot(labels_color, num_classes=10)
            color_oneHot = color_oneHot.float()
            input_one_hot = color_oneHot.to(device)

        else:
            labels_shape = labels_for_shape.to(device)  #get the shape label from the dataloader
            input_oneHot = F.one_hot(labels_shape, num_classes=s_classes) # 36 classes in emnist, 10 classes in f-mnist
            input_oneHot = input_oneHot.float()
            input_one_hot = input_oneHot.to(device)
        
        n = 1 # sampling noise
        z_label = label_net(input_one_hot,n)  #run a label through the model to generate a latent representation
        

        activations = vae.activations(image, False)  #load activity for each of the latent spaces based on the image
        z_actual = activations[whichcomponent]   #extract the activity for that latent representation

        # train shape label net
        
        loss_of_labels = loss_label(z_label, z_actual)   #compute the error
        loss_of_labels.backward(retain_graph = True)    #adjust the weights so that the label network output resembles the latent activation
        train_loss += loss_of_labels.item()      

        optimizer.step()

        if i % 1000 == 0:   #every 1000 samples grab a random sample
            label_net.eval()
            vae.eval()

            if whichcomponent == 'color':   #grab the appropriate decoder from the VAE
                feature_decoder = vae.color_decode_wrapper
            elif whichcomponent == 'shape':
                feature_decoder = vae.decoder_shape
            else:    
                feature_decoder = vae.decoder_object

            with torch.no_grad():   #Use them to reconstruct an object
                feature_recon = feature_decoder(z_actual)         # recon from an actual object
                feature_recon_label = feature_decoder(z_label)    #recon from a label 

                sample_size = 20
                orig_imgs = image[:sample_size]
                feature_recon = feature_recon[:sample_size] 
                feature_recon_label = feature_recon_label[:sample_size]

            output_img = torch.cat(
                    [orig_imgs,
                     feature_recon.view(sample_size, 3, 28, 28),
                     feature_recon_label.view(sample_size, 3, 28, 28)
                     ], 0)
            rows = 3
            #this next bit collapses the long image into a stack of rows so that the text can be added
            #convert the sample_size*rows x 3 x 28 x 28 tensor into a  stack that is now 3 x rows*28 x sample_size*28
            output_img2 = output_img.view(rows,sample_size,3,28,28)
            output_img2 = output_img2.permute(0,2,3,1,4).contiguous().view(rows,3,28,sample_size*28)
            output_img2 = output_img2.permute(1,0,2,3).contiguous().view(3,rows*28,sample_size*28)

            channels, height, width = output_img2.shape
            header_height = 20            
            # Create new tensor with extra height for text
            new_height = height + header_height
            new_tensor = torch.ones(channels, new_height, width) * 0.8  # Light gray background
            new_tensor[:, header_height:, :] = output_img2
            text_tensor = text_to_tensor("Image / recon from encoder / recon from label ",header_height,width)
            new_tensor[:, :header_height, :] = text_tensor

            utils.save_image(new_tensor,
                f'{folder_path}{whichcomponent}{str(epoch).zfill(5)}_{str(i).zfill(5)}.png',
                nrow = 1,
                normalize=False,
            )

        if i > max_iter + 1:
            break
    print(f'====> Epoch: {epoch} Average loss {whichcomponent}: {train_loss}')


    
# not working VVV
def test_outputs(test_loader, n = 0.5):
        vae_shape_labels.eval()
        vae_color_labels.eval()
        vae.eval()

        dataiter = iter(test_loader)
        image, labels = dataiter.next()
        labels_for_shape=labels[0].clone()
        labels_for_color=labels[1].clone()
              
        image = image.cuda()
        labels_shape = labels_for_shape.cuda()
        input_oneHot = F.one_hot(labels_shape, num_classes=s_classes) # 47 classes in emnist, 10 classes in f-mnist
        input_oneHot = input_oneHot.float()
        input_oneHot = input_oneHot.cuda()

        labels_color = labels_for_color  # get the color labels
        labels_color = labels_color.cuda()
        color_oneHot = F.one_hot(labels_color, num_classes=10)
        color_oneHot = color_oneHot.float()
        color_oneHot = color_oneHot.cuda()
        
        n=1
        z_shape_label = vae_shape_labels(input_oneHot,n)
        z_color_label = vae_color_labels(color_oneHot)

        z_shape, z_color, z_location = image_activations(image)

        with torch.no_grad():
                recon_imgs = vae.decoder_cropped(z_shape, z_color,0,0)
                recon_imgs_shape = vae.decoder_shape(z_shape, z_color,0)
                recon_imgs_color = vae.decoder_color(z_shape, z_color,0)

                recon_labels = vae.decoder_cropped(z_shape_label, z_color_label,0,0)
                recon_shapeOnly = vae.decoder_shape(z_shape_label, 0,0)
                recon_colorOnly = vae.decoder_color(0, z_color_label,0)

                sample_size = 20
                orig_imgs = image[:sample_size]
                recon_labels = recon_labels[:sample_size]
                recon_imgs = recon_imgs[:sample_size]
                recon_imgs_shape = recon_imgs_shape[:sample_size]
                recon_imgs_color = recon_imgs_color[:sample_size]
                recon_shapeOnly = recon_shapeOnly[:sample_size]
                recon_colorOnly = recon_colorOnly[:sample_size]

        utils.save_image(
                torch.cat(
                    [orig_imgs,
                     recon_imgs.view(sample_size, 3, 28, 28),
                     recon_imgs_shape.view(sample_size, 3, 28, 28),
                     recon_imgs_color.view(sample_size, 3, 28, 28),
                     recon_labels.view(sample_size, 3, 28, 28),
                     recon_shapeOnly.view(sample_size, 3, 28, 28),
                     recon_colorOnly.view(sample_size, 3, 28, 28)], 0),
                f'sample_training_labels/labeltest_with_{n}.png',
                nrow=sample_size,
                normalize=False,
                range=(-1, 1),
            )

def test_opposite_colors(test_loader, n = 0.5):
        vae_shape_labels.eval()
        vae_color_labels.eval()
        vae.eval()

        dataiter = iter(test_loader)
        image, labels = dataiter.next()
        labels_for_shape=labels[0].clone()
        labels_for_color=labels[1].clone()
              
        image = image.cuda()
        labels_shape = labels_for_shape.cuda()
        input_oneHot = F.one_hot(labels_shape, num_classes=s_classes) # 47 classes in emnist, 10 classes in f-mnist
        input_oneHot = input_oneHot.float()
        input_oneHot = input_oneHot.cuda()

        labels_color = labels_for_color  # get the color labels
        labels_color = labels_color.cuda()
        color_oneHot = F.one_hot(labels_color, num_classes=10)
        color_oneHot = color_oneHot.float()
        color_oneHot = color_oneHot.cuda()
        
        n=1
        z_shape_label = vae_shape_labels(input_oneHot,n)
        z_color_label = vae_color_labels(color_oneHot)

        z_shape, z_color, z_location = image_activations(image)

        with torch.no_grad():
                recon_imgs = vae.decoder_cropped(z_shape, z_color,0,0)
                recon_imgs_shape = vae.decoder_shape(z_shape, z_color,0)
                recon_imgs_color = vae.decoder_color(z_shape, z_color,0)

                recon_labels = vae.decoder_cropped(z_shape_label, z_color_label,0,0)
                recon_shapeOnly = vae.decoder_shape(z_shape_label, 0,0)
                recon_colorOnly = vae.decoder_color(0, z_color_label,0)

                sample_size = 20
                orig_imgs = image[:sample_size]
                recon_labels = recon_labels[:sample_size]
                recon_imgs = recon_imgs[:sample_size]
                recon_imgs_shape = recon_imgs_shape[:sample_size]
                recon_imgs_color = recon_imgs_color[:sample_size]
                recon_shapeOnly = recon_shapeOnly[:sample_size]
                recon_colorOnly = recon_colorOnly[:sample_size]

        utils.save_image(
                torch.cat(
                    [orig_imgs,
                     recon_imgs.view(sample_size, 3, 28, 28),
                     recon_imgs_shape.view(sample_size, 3, 28, 28),
                     recon_imgs_color.view(sample_size, 3, 28, 28),
                     recon_labels.view(sample_size, 3, 28, 28),
                     recon_shapeOnly.view(sample_size, 3, 28, 28),
                     recon_colorOnly.view(sample_size, 3, 28, 28)], 0),
                f'sample_training_labels_red_green/opposite_color_test.png',
                nrow=sample_size,
                normalize=False,
                range=(-1, 1),
            )