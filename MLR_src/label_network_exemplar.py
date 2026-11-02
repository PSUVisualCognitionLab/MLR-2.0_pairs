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
import random


bs = 100
s_classes = 36
c_classes = 10


class ObjectExemplars:
    """Stores exemplar latent activations per object class. 
    Acts like a label net: given a one-hot vector, returns a random exemplar's latent."""
    
    def __init__(self, exemplars_dict, device='cpu'):
        # exemplars_dict: {class_label: tensor of shape (n_exemplars, z_dim)}
        self.exemplars = exemplars_dict
        self.device = device
    
    def __call__(self, one_hot_labels, n=1):
        """Given a batch of one-hot labels, return a random exemplar latent for each."""
        batch_size = one_hot_labels.shape[0]
        class_indices = one_hot_labels.argmax(dim=1)  # convert one-hot to class index
        
        z_dim = list(self.exemplars.values())[0].shape[1]
        output = torch.zeros(batch_size, z_dim, device=self.device)
        
        for i in range(batch_size):
            cls = class_indices[i].item()
            if cls in self.exemplars and len(self.exemplars[cls]) > 0:
                idx = random.randint(0, len(self.exemplars[cls]) - 1)
                output[i] = self.exemplars[cls][idx]
            # else: stays zero (shouldn't happen if data is complete)
        
        return output
    
    def to(self, device):
        self.device = device
        self.exemplars = {k: v.to(device) for k, v in self.exemplars.items()}
        return self
    
    def eval(self):
        return self  # no-op, for compatibility
    
    def train(self):
        return self  # no-op, for compatibility
    
    def state_dict(self):
        return self.exemplars
    
    def load_state_dict(self, state_dict):
        self.exemplars = state_dict


def collect_object_exemplars(vae, dataloader, device, n_per_class=100):
    """Collect latent activations for each object class from the dataloader."""
    vae.eval()
    exemplars = {}  # {class_label: list of latent vectors}
    
    total_needed = n_per_class * 10  # assume 10 object classes, collect enough
    collected = 0
    
    with torch.no_grad():
        while collected < total_needed:
            data, labels = next(iter(dataloader))
            if type(data) == list:
                image = data[1].to(device)
            else:
                image = data.to(device)
            
            object_labels = labels[0]  # class labels
            
            mu_object, log_var_object = vae.encoder_object(image)
            # use mu directly (no sampling noise) for cleaner exemplars
            
            for i in range(len(mu_object)):
                cls = object_labels[i].item()
                if cls not in exemplars:
                    exemplars[cls] = []
                if len(exemplars[cls]) < n_per_class:
                    exemplars[cls].append(mu_object[i].cpu())
                    collected += 1
    
    # stack lists into tensors
    for cls in exemplars:
        exemplars[cls] = torch.stack(exemplars[cls])
    
    print(f"Collected exemplars: {', '.join(f'class {k}: {v.shape[0]}' for k, v in exemplars.items())}")
    return exemplars


def train_labelnet(dataloaders, vae, epoch_count, z_dim, checkpoint_folder, trained_components):
    if not os.path.exists('training_samples/'):
        os.mkdir('training_samples/')
    
    if not os.path.exists(f'training_samples/{checkpoint_folder}/'):
        os.mkdir(f'training_samples/{checkpoint_folder}/')
    
    sample_folder_path = f'training_samples/{checkpoint_folder}/label_net_samples/'
    if not os.path.exists(sample_folder_path):
        os.mkdir(sample_folder_path)
    
    device = next(vae.parameters()).device
    
    vae_shape_labels = VAEshapelabels(xlabel_dim=s_classes, hlabel_dim=20, zlabel_dim=z_dim)
    vae_color_labels = VAEcolorlabels(xlabel_dim=10, hlabel_dim=7, zlabel_dim=z_dim)

    optimizer_shapelabels = optim.Adam(vae_shape_labels.parameters())
    optimizer_colorlabels = optim.Adam(vae_color_labels.parameters())

    # train shape and color label nets normally
    label_nets = {}
    if 'shape' in trained_components:
        label_nets['shape'] = [vae_shape_labels, optimizer_shapelabels]
    if 'color' in trained_components:
        label_nets['color'] = [vae_color_labels, optimizer_colorlabels]

    for whichcomponent in label_nets:
        label_net, optimizer = label_nets[whichcomponent]
        for epoch in range(1, epoch_count):
            whichloader = training_components[whichcomponent][0][0]
            train_labels(vae, label_net, whichcomponent, epoch, dataloaders[whichloader], optimizer, sample_folder_path)

    # collect object exemplars instead of training a network
    vae_object_labels = None
    if 'object' in trained_components:
        print('Collecting object exemplars...')
        whichloader = training_components['object'][0][0]
        exemplars = collect_object_exemplars(vae, dataloaders[whichloader], device, n_per_class=100)
        vae_object_labels = ObjectExemplars(exemplars, device)
        
        # generate sample images to verify
        visualize_exemplars(vae, vae_object_labels, dataloaders[whichloader], device, sample_folder_path)

    checkpoint = {
        'state_dict_shape_labels': vae_shape_labels.state_dict(),
        'state_dict_color_labels': vae_color_labels.state_dict(),
        'state_dict_object_labels': vae_object_labels.state_dict() if vae_object_labels else {},
        'object_label_type': 'exemplar',  # flag so loader knows which type
        'optimizer_shape': optimizer_shapelabels.state_dict(),
        'optimizer_color': optimizer_colorlabels.state_dict(),
        'z_dim': z_dim
    }
    torch.save(checkpoint, f'checkpoints/{checkpoint_folder}/label_network_checkpoint.pth')


def visualize_exemplars(vae, object_exemplars, dataloader, device, folder_path):
    """Generate sample reconstructions from object exemplars to verify quality."""
    vae.eval()
    sample_size = 20
    
    # get real images for comparison
    data, labels = next(iter(dataloader))
    if type(data) == list:
        image = data[1].to(device)
    else:
        image = data.to(device)
    
    object_labels = labels[0]
    
    with torch.no_grad():
        # reconstruct from encoder
        recon_from_encoder = vae.decoder_object(vae.encoder_object(image)[0])
        
        # reconstruct from exemplars
        one_hot = F.one_hot(object_labels, num_classes=10).float().to(device)
        z_exemplar = object_exemplars(one_hot)
        recon_from_exemplar = vae.decoder_object(z_exemplar)
    
    orig_imgs = image[:sample_size]
    recon_enc = recon_from_encoder[:sample_size]
    recon_exm = recon_from_exemplar[:sample_size]
    
    output_img = torch.cat([
        orig_imgs,
        recon_enc.view(sample_size, 3, 28, 28),
        recon_exm.view(sample_size, 3, 28, 28)
    ], 0)
    
    rows = 3
    output_img2 = output_img.view(rows, sample_size, 3, 28, 28)
    output_img2 = output_img2.permute(0, 2, 3, 1, 4).contiguous().view(rows, 3, 28, sample_size * 28)
    output_img2 = output_img2.permute(1, 0, 2, 3).contiguous().view(3, rows * 28, sample_size * 28)
    
    channels, height, width = output_img2.shape
    header_height = 20
    new_height = height + header_height
    new_tensor = torch.ones(channels, new_height, width) * 0.8
    new_tensor[:, header_height:, :] = output_img2.cpu()
    text_tensor = text_to_tensor("Image / recon from encoder / recon from exemplar", header_height, width)
    new_tensor[:, :header_height, :] = text_tensor
    
    utils.save_image(new_tensor, f'{folder_path}object_exemplars_verify.png', nrow=1, normalize=False)
    print(f"Saved exemplar verification to {folder_path}object_exemplars_verify.png")


# shape label network
class VAEshapelabels(nn.Module):
    def __init__(self, xlabel_dim, hlabel_dim, zlabel_dim):
        super(VAEshapelabels, self).__init__()

        # encoder part
        self.fc1label = nn.Linear(xlabel_dim, hlabel_dim)
        self.fc21label = nn.Linear(hlabel_dim, zlabel_dim)  # mu shape
        self.fc22label = nn.Linear(hlabel_dim, zlabel_dim)  # log-var shape

    def sampling_labels(self, mu, log_var, n=1):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) * n
        return mu + eps * std

    def forward(self, x_labels, n):
        h = F.relu(self.fc1label(x_labels))
        mu_shape_label = self.fc21label(h)
        log_var_shape_label = self.fc22label(h)
        z_shape_label = self.sampling_labels(mu_shape_label, log_var_shape_label, n)
        return z_shape_label

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
        return z_color_label


def load_checkpoint_labels(filepath, label_type, d=0):
    assert label_type in ["shape", "color", "object"], f"label_type: {label_type} is invalid, must be one of: shape, color, object"
    print(f"loading {label_type} label network")

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{d}')
        torch.cuda.set_device(d)
    else:
        device = 'cpu'

    checkpoint = torch.load(filepath, map_location=device)
    if 'z_dim' not in checkpoint:
        z_dim = 8
    else:
        z_dim = checkpoint['z_dim']

    if label_type == 'object':
        # check if this checkpoint uses exemplars
        if checkpoint.get('object_label_type') == 'exemplar':
            exemplars = checkpoint['state_dict_object_labels']
            label_net = ObjectExemplars(exemplars, device)
            return label_net
        else:
            # legacy: load as a neural network
            label_net = VAEshapelabels(xlabel_dim=s_classes, hlabel_dim=20, zlabel_dim=z_dim)
            label_net.load_state_dict(checkpoint['state_dict_object_labels'])
            for parameter in label_net.parameters():
                parameter.requires_grad = False
            label_net.eval().to(device)
            return label_net
    
    elif label_type == 'shape':
        label_net = VAEshapelabels(xlabel_dim=s_classes, hlabel_dim=20, zlabel_dim=z_dim)
    elif label_type == 'color':
        label_net = VAEcolorlabels(xlabel_dim=10, hlabel_dim=7, zlabel_dim=z_dim)
    
    label_net.load_state_dict(checkpoint[f'state_dict_{label_type}_labels'])
    for parameter in label_net.parameters():
        parameter.requires_grad = False
    label_net.eval().to(device)
    return label_net


def loss_label(label_act, image_act):
    criterion = nn.MSELoss(reduction='sum')
    e = criterion(label_act, image_act)
    return e


def train_labels(vae, label_net, whichcomponent, epoch, train_loader, optimizer, folder_path):    
    device = next(vae.parameters()).device
    vae.eval()
    train_loss = 0

    label_net.train().to(device)

    dataiter = train_loader

    max_iter = 100
    for i, j in enumerate(train_loader):
        optimizer.zero_grad()

        image, labels = next(dataiter)
        labels_for_shape = labels[0].clone()
        labels_for_color = labels[1].clone()
              
        image = image[1].cuda()

        if whichcomponent == 'color':
            labels_color = labels_for_color.to(device)
            color_oneHot = F.one_hot(labels_color, num_classes=10)
            color_oneHot = color_oneHot.float()
            input_one_hot = color_oneHot.to(device)
        else:
            labels_shape = labels_for_shape.to(device)
            input_oneHot = F.one_hot(labels_shape, num_classes=s_classes)
            input_oneHot = input_oneHot.float()
            input_one_hot = input_oneHot.to(device)
        
        n = 1
        z_label = label_net(input_one_hot, n)

        activations = vae.activations(image, False)
        z_actual = activations[whichcomponent]

        loss_of_labels = loss_label(z_label, z_actual)
        loss_of_labels.backward(retain_graph=True)
        train_loss += loss_of_labels.item()

        optimizer.step()

        if i % max_iter == 0 and i > 0:
            label_net.eval()
            vae.eval()

            if whichcomponent == 'color':
                feature_decoder = vae.color_decode_wrapper
            elif whichcomponent == 'shape':
                feature_decoder = vae.decoder_shape
            else:    
                feature_decoder = vae.decoder_object

            with torch.no_grad():
                if whichcomponent == 'color':
                    feature_recon, _, _, _, _, _, _ = vae(image, 'color', ['color'])
                elif whichcomponent == 'shape':
                    feature_recon, _, _, _, _, _, _ = vae(image, 'shape', ['shape'])
                else:
                    feature_recon, _, _, _, _, _, _ = vae(image, 'object', ['object'])
                feature_recon_label = feature_decoder(z_label)

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
            output_img2 = output_img.view(rows, sample_size, 3, 28, 28)
            output_img2 = output_img2.permute(0, 2, 3, 1, 4).contiguous().view(rows, 3, 28, sample_size * 28)
            output_img2 = output_img2.permute(1, 0, 2, 3).contiguous().view(3, rows * 28, sample_size * 28)

            channels, height, width = output_img2.shape
            header_height = 20
            new_height = height + header_height
            new_tensor = torch.ones(channels, new_height, width) * 0.8
            new_tensor[:, header_height:, :] = output_img2
            text_tensor = text_to_tensor("Image / recon from encoder / recon from label ", header_height, width)
            new_tensor[:, :header_height, :] = text_tensor
            utils.save_image(new_tensor,
                f'{folder_path}{whichcomponent}{str(epoch).zfill(5)}_{str(i).zfill(5)}.png',
                nrow=1,
                normalize=False,
            )

            label_net.train()

        if i > max_iter + 1:
            break
    print(f'====> Epoch: {epoch} Average loss {whichcomponent}: {train_loss}')
