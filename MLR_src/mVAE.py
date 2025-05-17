#MLR 2.0

#The second installment of the MLR model line, written largely by Ian Deal and Brad Wyble
#This original version of this model is published in
#Hedayati, S., O’Donnell, R. E., & Wyble, B. (2022). A model of working memory for latent representations. Nature Human Behaviour, 6(5), 709-719.
#And the code in that work is a variant of
# MNIST VAE from http://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb
# Modified by Brad Wyble, Shekoo Hedayati

#In this version, the model adds to the original MLR model the following features:
#-a large Retina  (100 pixels wide)
#-Convolutional encoder and decoder
#-Location latent space  (in the horizontal diection)
#-improved loss functions for shape and color
#-White is now one of the 10 colors
#-Skip connection trained on bi-color stimuli
#-Label networks  akin to SVRHM paper:
#Hedayati, S., Beaty, R., & Wyble, B. (2021). Seeking the Building Blocks of Visual Imagery and Creativity in a Cognitively Inspired Neural Network. arXiv preprint arXiv:2112.06832.

# prerequisites
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
from torchvision import utils
from torchvision.utils import save_image
from tqdm import tqdm
from torchvision import transforms as torch_transforms

from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION

#torch.set_default_dtype(torch.float64)

# load a saved vae checkpoint
def load_checkpoint(filepath, d=0):
    vae, z = vae_builder()
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{d}')
        torch.cuda.set_device(d)
    else:
        device = 'cpu'
    
    torch_version = torch.__version__
    if torch_version == '2.4.0':
        checkpoint = torch.load(filepath, device, weights_only = True)
    else:
        checkpoint = torch.load(filepath, device)
    vae.load_state_dict(checkpoint['state_dict'], strict=False)
    vae.to(device)
    return vae

# model training data set and dimensions
data_set_flag = 'padded_mnist_3rd' # mnist, cifar10, padded_mnist, padded_cifar10
imgsize = 28
retina_size = 64 # by default should be same size as image
vae_type_flag = 'CNN' # must be CNN or FC
x_dim = retina_size * retina_size * 3
h_dim1 = 256
h_dim2 = 128
z_dim = 12
l_dim = 64*2 # 2dim (2, retina_size) position
zl_dim = z_dim
sc_dim = 10


#CNN VAE
#this model takes in a single cropped image and a location 1-hot vector  (to be replaced by an attentional filter that determines location from a retinal image)
#there are three latent spaces:location, shape and color and 6 loss functions
#loss functions are: shape, color, location, retinal, cropped (shape + color combined), skip

class VAE_CNN(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, l_dim, sc_dim):
        super(VAE_CNN, self).__init__()
        # encoder part
        self.l_dim = l_dim
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.fc2 = nn.Linear(int(imgsize / 4) * int(imgsize / 4)*16, h_dim2)
        self.fc_bn2 = nn.BatchNorm1d(h_dim2)

        # bottle neck part  # Latent vectors mu and sigma
        self.fc31 = nn.Linear(h_dim2, z_dim)  # shape
        self.fc32 = nn.Linear(h_dim2, z_dim)
        self.fc33 = nn.Linear(h_dim2, z_dim)  # color
        self.fc34 = nn.Linear(h_dim2, z_dim)

        # decoder part
        self.fc4s = nn.Linear(z_dim, h_dim2)  # shape
        self.fc4c = nn.Linear(z_dim, h_dim2)  # color
        self.fc4l = nn.Linear(zl_dim, l_dim)  # location
        self.fc4sc = nn.Linear(z_dim, sc_dim)  # scale

        self.fc5 = nn.Linear(h_dim2, int(imgsize/4) * int(imgsize/4) * 16)
        self.fc8 = nn.Linear(16*28*28,16*28*28) #skip conection

        self.conv5 = nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(3)

        self.skip_bn = nn.BatchNorm2d(16)

        self.localization = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(4,16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4,32),#BatchNorm2d(32),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4,16),#BatchNorm2d(16),
            nn.ReLU(),)

        self.regressor = nn.Sequential(
            nn.Linear(int(retina_size / 4) * int(retina_size / 4)*16, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )
  
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.1)
        self.sparse_relu = nn.Threshold(threshold=0.5, value=0)

        # map scalars
        self.shape_scale = 1 #1.9
        self.color_scale = 1 #2

    def construct_theta(self, z_where):
        # Take a batch of three-vectors, and massages them into a batch of
        # 2x3 matrices with elements like so:
        # [s,x,y] -> [[s,0,x],
        #             [0,s,y]]
        n = z_where.size(0)
        out = torch.zeros(n, 2, 3).to(z_where.device)
        out[:,0,0] = z_where[:,0]
        #out[:,0,1] = z_where[:,3]
        out[:,0,2] = z_where[:,1]
        #out[:,1,0] = z_where[:,3]
        out[:,1,1] = z_where[:,0]
        out[:,1,2] = z_where[:,2]
        out = out.view(n, 2, 3)
       
        return out

    def invert_theta(self, z_where):
        # Take a batch of z_where vectors, and compute their "inverse".
        # That is, for each row compute:
        # [s,x,y] -> [1/s,-x/s,-y/s]
        # These are the parameters required to perform the inverse of the
        # spatial transform performed in the generative model.
        n = z_where.size(0)
        out = torch.cat((z_where.new_ones(n, 1), -z_where[:, 1:]), 1)
        # Divide all entries by the scale.
        out = out / z_where[:, 0:1].to(z_where.device)
        return out

    def stn_encode(self, x):
        B = x.shape[0]
        # x is [B, 3, 64, 64]
        z=self.localization(x)
        theta = self.regressor(z.view(-1, int(retina_size / 4) * int(retina_size / 4)*16))  # [B, 2, 3]
        theta = theta.view(-1, 3).to(x.device)
        grid = F.affine_grid(self.construct_theta(theta), (B,3,64,64), align_corners=False)  # same 64×64 resolution
        x_transformed = F.grid_sample(x, grid, align_corners=False)

        # crop by slicing out the 28×28 region centered by the stn
        crop = x_transformed[:, :, 18:46, 18:46]
        return crop, theta

    def stn_decode(self, crop, theta):
        B = crop.shape[0]

        # put the 28×28 crop back into a 64×64 retina
        canvas = torch.zeros(B, 3, 64, 64, device=crop.device)
        canvas[:, :, 18:46, 18:46] = crop  # place the 28×28 patch back
        #canvas[:, :, 18:46, 18:46] = torch.rot90(crop, k=2, dims=(-2, -1))
        theta = self.invert_theta(theta)
        theta = self.construct_theta(theta).to(crop.device)
        
        # reconstruct full retina
        grid = F.affine_grid(theta, (B, 3, 64, 64), align_corners=False)
        retina = F.grid_sample(canvas, grid, align_corners=False)

        return retina
    
    def encoder(self, x, hskip = None):
        if hskip is not None: # for reprocessing l1 through bottleneck
            h = hskip.view(-1, 16, imgsize, imgsize)
            h = self.relu(self.bn2(self.conv2(h)))        
            h = self.relu(self.bn3(self.conv3(h)))
            h = self.relu(self.bn4(self.conv4(h)))
            h = h.view(-1,int(imgsize / 4) * int(imgsize / 4)*16)
            h = self.relu(self.fc_bn2(self.fc2(h)))
        else:    
            b_dim = x.size(0)
            h = self.sparse_relu(self.bn1(self.conv1(x)))
            hskip = h.view(b_dim,-1)
            h = self.relu(self.bn2(self.conv2(h)))        
            h = self.relu(self.bn3(self.conv3(h)))
            h = self.relu(self.bn4(self.conv4(h)))
            h = h.view(-1,int(imgsize / 4) * int(imgsize / 4)*16)
            h = self.relu(self.fc_bn2(self.fc2(h)))

        return self.fc31(h), self.fc32(h), self.fc33(h), self.fc34(h), hskip # mu, log_var

    def activations(self, x, retinal=False, hskip = None): # returns shape, color, scale, location, and skip(l1) latent activations
        if hskip is not None:
            mu_shape, log_var_shape, mu_color, log_var_color, hskip = self.encoder(x, hskip)
        
        elif retinal is True:    #passing in a cropped+ location as input
            x, theta = self.stn_encode(x)
            mu_shape, log_var_shape, mu_color, log_var_color, hskip = self.encoder(x)
        
        else:  #passing in just cropped image
            mu_shape, log_var_shape, mu_color, log_var_color, hskip = self.encoder(x)
        
        z_shape = self.sampling(mu_shape, log_var_shape)
        z_color = self.sampling(mu_color, log_var_color)

        if retinal is True:
            z_scale = theta[:,:1]
            z_location = theta[:,1:]
        else:
            z_scale = 0
            z_location = 0

        out_dict = {'shape':z_shape, 'color':z_color, 'scale':z_scale, 'location':z_location, 'skip':hskip}

        return out_dict

    def decoder(self, activations, which_decode): #generic decoder function
        assert which_decode in ['shape', 'color', 'cropped', 'retinal', 'shape_retinal', 'color_retinal'], f'which_decode: {which_decode} is not valid. Must be one of: \'shape\', \'color\', \'cropped\', \'retinal\', \'shape_retinal\', \'color_retinal\''

        if which_decode == 'shape':
            assert 'shape' in activations, 'the shape activation is missing, must have key: \'shape\''
            return self.decoder_shape(activations['shape'], 0, 0)
        
        elif which_decode == 'color':
            assert 'color' in activations, 'the color activation is missing, must have key: \'color\''
            return self.decoder_color(0, activations['color'], 0)
        
        elif which_decode == 'cropped':
            assert 'color' in activations, 'the color activation is missing, both shape are color are needed for cropped, must have key: \'color\''
            assert 'shape' in activations, 'the shape activation is missing, both shape are color are needed for cropped, must have key: \'shape\''
            return self.decoder_cropped(activations['shape'], activations['color'], 0)

        elif 'retinal' in which_decode:
            assert 'location' in activations and 'scale' in activations, 'the scale or location activation is missing, must have keys: \'scale\' and \'location\''
            theta = torch.cat([activations['scale'], activations['location']], 1)
            retinal_decode = None
            shape_act, color_act = 0, 0

            if 'shape' in which_decode:
                assert 'shape' in activations, 'the shape activation is missing, must have key: \'shape\''
                retinal_decode = 'shape'
                shape_act = activations['shape']
            
            elif 'color' in which_decode:
                assert 'color' in activations, 'the color activation is missing, must have key: \'color\''
                retinal_decode = 'color'
                color_act = activations['color']

            return self.decoder_retinal(shape_act, color_act, theta, retinal_decode)

        else:
            print(f'invalid which_decode: {which_decode}')

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder_retinal(self, z_shape, z_color, theta, whichdecode = None):
        # digit recon
        b_dim = z_shape.size(0)
        if whichdecode == 'shape':
            h = (F.relu(self.fc4s(z_shape)) * self.shape_scale)
        elif whichdecode == 'color':
            h = (F.relu(self.fc4c(z_color)) * self.color_scale)
        else:
            h = (F.relu(self.fc4c(z_color)) * self.color_scale) + (F.relu(self.fc4s(z_shape)) * self.shape_scale)
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize/4), int(imgsize/4))
        h = self.relu(self.bn5(self.conv5(h)))
        h = self.relu(self.bn6(self.conv6(h)))
        h = self.relu(self.bn7(self.conv7(h)))
        h = self.conv8(h).detach().view(-1, 3, imgsize, imgsize) #detach conv
        h = torch.sigmoid(h)
        crop_out = h.clone()

        h = self.stn_decode(h, theta)

        if self.training:
            return {'recon':h, 'crop':crop_out}
        else:
            return h

    def decoder_color(self, z_shape, z_color, hskip=0):
        h = F.relu(self.fc4c(z_color)) * self.color_scale
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize / 4), int(imgsize / 4))
        h = self.relu(self.bn5(self.conv5(h)))
        if self.training:
            h = self.dropout(h)
        h = self.relu(self.bn6(self.conv6(h)))
        if self.training:
            h = self.dropout(h)
        h = self.relu(self.bn7(self.conv7(h)))
        if self.training:
            h = self.dropout(h)
        h = self.conv8(h).view(-1, 3, imgsize, imgsize)
        return torch.sigmoid(h)

    def decoder_shape(self, z_shape, z_color=0, hskip=0):
        h = F.relu(self.fc4s(z_shape)) * self.shape_scale
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize / 4), int(imgsize / 4))
        h = self.relu(self.bn5(self.conv5(h)))
        if self.training:
            h = self.dropout(h)
        h = self.relu(self.bn6(self.conv6(h)))
        if self.training:
            h = self.dropout(h)
        h = self.relu(self.bn7(self.conv7(h)))
        if self.training:
            h = self.dropout(h)
        h = self.conv8(h).view(-1, 3, imgsize, imgsize)
        return torch.sigmoid(h)

    def decoder_cropped(self, z_shape, z_color, z_location=0, hskip=0):
        h = (F.relu(self.fc4c(z_color)) * self.color_scale) + (F.relu(self.fc4s(z_shape)) * self.shape_scale)
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize / 4), int(imgsize / 4))
        h = self.relu(self.bn5(self.conv5(h)))
        if self.training:
            h = self.dropout(h)
        h = self.relu(self.bn6(self.conv6(h)))
        if self.training:
            h = self.dropout(h)
        h = self.relu(self.bn7(self.conv7(h)))
        if self.training:
            h = self.dropout(h)
        h = self.conv8(h).view(-1, 3, imgsize, imgsize)
        return torch.sigmoid(h)

    def decoder_skip_cropped(self, z_shape, z_color, z_location, hskip):
        h= self.fc8(hskip)#hskip
        if self.training:
            h = self.dropout(h)
        h = self.relu(self.skip_bn(h.view(-1,16,28,28)))
        h = self.conv8(h.view(-1,16,28,28)).view(-1, 3, imgsize, imgsize)
        return torch.sigmoid(h)

        
    def decoder_skip_retinal(self, z_shape, z_color, z_location, hskip):
        # digit recon
        h = F.relu(hskip)
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize/4), int(imgsize/4))
        h = self.relu(self.bn5(self.conv5(h)))
        h = self.relu(self.bn6(self.conv6(h)))
        h = self.relu(self.bn7(self.conv7(h)))
        h = self.conv8(h).view(-1, 3, imgsize, imgsize).detach()
        h = torch.sigmoid(h)
        # location vector recon
        l = z_location.detach() #cont. repr of location
        l = l.view(-1,1,1,8)
        l = torch.sigmoid(l)
        l = l.expand(-1, 3, imgsize, 8) # reshape to concat
        # combine into retina
        h = torch.cat([h,l], dim = 3)
        b_dim = h.size()[0]*h.size()[2]
        h = h.view(b_dim,-1)
        h = self.relu(self.fc6(h))
        h = self.fc7(h).view(-1,3,imgsize,retina_size)
        return torch.sigmoid(h)

    def forward(self, x, whichdecode='noskip', keepgrad=[]):
        if type(x) == list or type(x) == tuple:    #passing in a cropped+ location as input
            x = x[0].cuda()
            x, theta = self.stn_encode(x)
            mu_shape, log_var_shape, mu_color, log_var_color, hskip = self.encoder(x)
        else:  #passing in just cropped image
            x = x.cuda()
            mu_shape, log_var_shape, mu_color, log_var_color, hskip = self.encoder(x)

        # the maps that are used in the training process.. the others are detached to zero out those gradients
        if ('shape' in keepgrad):
            z_shape = self.sampling(mu_shape, log_var_shape)
        else:
            z_shape = self.sampling(mu_shape, log_var_shape).detach()

        if ('color' in keepgrad):
            z_color = self.sampling(mu_color, log_var_color)
        else:
            z_color = self.sampling(mu_color, log_var_color).detach()

        if ('skip' in keepgrad):
            hskip = hskip
        else:
            hskip = hskip.detach()

        if(whichdecode == 'cropped'):
            output = self.decoder_cropped(z_shape,z_color, 0, hskip)
        elif (whichdecode == 'retinal'):
            output = self.decoder_retinal(z_shape,z_color, theta)
            #output = self.stn_decode(x, theta)
        elif (whichdecode == 'skip_cropped'):
            output = self.decoder_skip_cropped(0, 0, 0, hskip)
        elif (whichdecode == 'skip_retinal'):
            output = self.decoder_skip_retinal(0, 0, 0, hskip)
        elif (whichdecode == 'color'):
            output = self.decoder_color(0, z_color , 0)
        elif (whichdecode == 'shape'):
            output = self.decoder_shape(z_shape,0, 0)
        elif (whichdecode == 'location'):
            output = self.decoder_location(0, 0, 0)
        elif (whichdecode == 'scale'):
            output = self.decoder_scale(0, 0, 0, z_scale=0)

        return output, mu_color, log_var_color, mu_shape, log_var_shape

# function to build a model instance
def vae_builder(vae_type = vae_type_flag, x_dim = x_dim, h_dim1 = h_dim1, h_dim2 = h_dim2, z_dim = z_dim, l_dim = l_dim, sc_dim = sc_dim):
    vae = VAE_CNN(x_dim, h_dim1, h_dim2, z_dim, l_dim, sc_dim)

    return vae, z_dim


######the loss functions
#pixelwise loss for the entire retina (dimensions are cropped image height x retina_size)
def loss_function(recon_x, x, crop_x):
    if crop_x is not None:
        x = place_crop(crop_x,x[2].clone())
    else:
        x=x[0].clone()
    x = x.cuda()
    BCE = F.binary_cross_entropy(recon_x.view(-1, 3, retina_size, retina_size), x.view(-1, 3, retina_size, retina_size), reduction='sum')
    return BCE

#pixelwise loss for just the cropped image
def loss_function_crop(recon_x, x):
    if len(x) <= 5:
        x = x[1].clone().cuda()
    else:
        x = x.clone().cuda()
    BCE = F.binary_cross_entropy(recon_x.view(-1, imgsize * imgsize * 3), x.view(-1, imgsize * imgsize * 3), reduction='sum')
    return BCE


# loss for shape in a cropped image
def loss_function_shape(recon_x, x, mu, log_var):
    if len(x) <= 5:
        x = x[1].clone().cuda()
    else:
        x = x.clone().cuda()
    # make grayscale reconstruction
    gray_x = x.view(-1, 3, imgsize, imgsize).mean(1)
    gray_x = torch.stack([gray_x, gray_x, gray_x], dim=1)
    
    BCEGray = F.binary_cross_entropy(recon_x.view(-1, imgsize * imgsize * 3), gray_x.view(-1,imgsize * imgsize * 3), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCEGray + KLD

#loss for just color in a cropped image
def loss_function_color(recon_x, x, mu, log_var):
    if len(x) <= 5:
        x = x[1].clone().cuda()
    else:
        x = x.clone().cuda()
    # make color-only (no shape) reconstruction and use that as the loss function
    recon = recon_x.clone().view(-1, 3 * imgsize * imgsize)
    # compute the maximum color for the r,g and b channels for each digit separately
    maxr, maxi = torch.max(x[:, 0, :], -1, keepdim=True)
    maxg, maxi = torch.max(x[:, 1, :], -1, keepdim=True)
    maxb, maxi = torch.max(x[:, 2, :], -1, keepdim=True)
    newx = x.clone()
    newx[:, 0, :] = maxr
    newx[:, 1, :] = maxg
    newx[:, 2, :] = maxb
    newx = newx.view(-1, imgsize * imgsize * 3)
    BCE = F.binary_cross_entropy(recon, newx, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

#loss for just location
def loss_function_location(recon_x, x, mu, log_var):
    x = x[2].clone().cuda()
    BCE = F.binary_cross_entropy(recon_x.view(-1,2,retina_size), x.view(-1,2,retina_size), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

#loss for just scale
def loss_function_scale(recon_x, x, mu, log_var):
    x = x[3].clone().cuda()
    BCE = F.binary_cross_entropy(recon_x.view(-1,retina_size,retina_size), x.view(-1,retina_size,retina_size), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

# test recreate img with different features
def progress_out(vae, data, checkpoint_folder):
    device = next(vae.parameters()).device 
    sample_size = 25
    vae.eval()
    sample = data[:sample_size].to(device)
    activations = vae.activations(sample, False)
    recon = vae.decoder_cropped(activations['shape'],activations['color'])
    skip = vae.decoder_skip_cropped(0, 0, 0, activations['skip'])
    shape = vae.decoder_shape(activations['shape'], 0, 0)
    color = vae.decoder_color(0, activations['color'], 0)
    vae.train()
     

    utils.save_image(
            torch.cat([sample.view(sample_size, 3, imgsize, imgsize)[:25], recon.view(sample_size, 3, imgsize, imgsize)[:25], skip.view(sample_size, 3, imgsize, imgsize)[:25],
                       shape.view(sample_size, 3, imgsize, imgsize)[:25], color.view(sample_size, 3, imgsize, imgsize)[:25]], 0),
            f'training_samples/{checkpoint_folder}/cropped_sample.png',
            nrow=sample_size, normalize=False)

def test_loss(vae, test_data, whichdecode = []):
    loss_dict = {}
    vae.eval()

    for decoder in whichdecode:
        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape = vae(test_data, decoder)
        
        if decoder == 'retinal':
            loss = loss_function(recon_batch, test_data, None)
        
        elif decoder == 'cropped':
            loss = loss_function_crop(recon_batch, test_data[1])
        
        loss_dict[decoder] = loss.item()

    return loss_dict

def update_seen_labels(batch_labels, current_labels):
    new_label_lst = []
    for i in range(len(batch_labels)):
        s = batch_labels[0][i].item() # shape label
        c = batch_labels[1][i].item() # color label
        r = batch_labels[2][i].item() # retina location label
        new_label_lst += [(s, c, r)]
    seen_labels = set(new_label_lst) | set(current_labels) # creates a new set 
    return seen_labels

def place_crop(crop_data,loc): # retina placement on GPU for training
    #print(loc.size())
    #resize = torch_transforms.Resize((28, 28))
    #crop_data = resize(torch_transforms.functional.to_pil_image(crop_data))
    #crop_data = torch_transforms.ToTensor(crop_data)
    print(crop_data.size())
    b_dim = crop_data.size(0)
    out_retina = torch.zeros(b_dim,3,retina_size,retina_size).cuda()
    for i in range(len(out_retina)):
        j,x = torch.max(loc[i][0],dim=0)
        z,y = torch.max(loc[i][1],dim=0)
        #print(x,y)
        out_retina[i,:,(retina_size-y)-imgsize:retina_size-y,x:x+imgsize] = crop_data[i]
    #print(out_retina.size())
    return out_retina

def component_to_grad(comp): # determine gradient for componeent training
    if comp == 'shape':
        return ['shape']
    elif comp == 'color':
        return ['color']
    elif comp == 'cropped':
        return ['shape', 'color']
    elif comp == 'skip_cropped':
        return ['skip']
    elif comp == 'retinal':
        return []
    elif comp == 'location':
        return ['location']
    else:
        raise Exception(f'Invalid component: {comp}')

def train(vae, optimizer, epoch, dataloaders, return_loss = False, seen_labels = {}, components = {}, max_iter = 600, checkpoint_folder=None):
    vae.train()
    train_loader_noSkip, emnist_skip, fmnist_skip, test_loader, sample_loader, block_loader = dataloaders[0], dataloaders[1], dataloaders[2], dataloaders[3], dataloaders[4], dataloaders[5]
    train_loss = 0
    dataiter_noSkip = iter(train_loader_noSkip) # the latent space is trained on EMNIST, MNIST, and f-MNIST
    block_iter = iter(block_loader)
    if fmnist_skip != None:
        #dataiter_emnist_skip= iter(emnist_skip) # The skip connection is trained on pairs from EMNIST, MNIST, and f-MNIST composed on top of each other
        dataiter_fmnist_skip= iter(fmnist_skip)
    test_iter = iter(test_loader)
    #sample_iter = iter(sample_loader)
    count = 0
    loader=tqdm(train_loader_noSkip, total = max_iter)

    retinal_loss_train, cropped_loss_train = 0, 0 # loss metrics returned to Training.py
    block_data, block_labels = next(block_iter)

    for i,j in enumerate(loader):
        count += 1
        data, batch_labels = next(dataiter_noSkip)

        # shuffle in the block dataset
        z =5 # random.randint(0,10)
        if z <= 1:
            data = block_data
        
        optimizer.zero_grad()
        
        # determine which component is being trained
        comp_ind = count % len(components)
        whichdecode_use = components[comp_ind]
        keepgrad = component_to_grad(whichdecode_use)

        if whichdecode_use == 'skip_cropped':
            data_skip = next(dataiter_fmnist_skip)
            r = random.randint(0,1)
            if r == 1:
                data = data_skip[0]
            else:
                data = data[1]
        
        if whichdecode_use in ['cropped', 'shape', 'color']:
            data = data[1]
        
        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape = vae(data, whichdecode_use, keepgrad)
            
        if whichdecode_use == 'shape':  # shape
            loss = loss_function_shape(recon_batch, data, mu_shape, log_var_shape)

        elif whichdecode_use == 'color': # color
            loss = loss_function_color(recon_batch, data, mu_color, log_var_color)

        elif whichdecode_use == 'retinal': # retinal
            #loss = loss_function(recon_batch['recon'], data, recon_batch['crop'])
            loss = loss_function(recon_batch['recon'], data, None)
            retinal_loss_train = loss.item()
            if count >= 0.9*max_iter:
                utils.save_image(
                    torch.cat([data[0].view(-1, 3, retina_size, retina_size)[:25].cpu(), recon_batch['recon'].view(-1, 3, retina_size, retina_size)[:25].cpu() 
                               #,place_crop(recon_batch['crop'],data[2]).view(-1, 3, retina_size, retina_size)[:25].cpu()
                               ], 0),
                    f"training_samples/{checkpoint_folder}/retinal_recon_test{epoch%3}.png",
                    nrow=25, normalize=False)

        elif whichdecode_use == 'cropped': # cropped
            loss = loss_function_crop(recon_batch, data)
            cropped_loss_train = loss.item()

        elif whichdecode_use == 'skip_cropped': # skip training
            loss = loss_function_crop(recon_batch, data)

        elif whichdecode_use == 'scale': # scale training
            loss = loss_function_crop(recon_batch, data)
        
        #l1_norm = sum(p.abs().sum() for p in vae.parameters())
        #loss += l1_norm*0.0001
        loss.backward()

        train_loss += loss.item()
        optimizer.step()
        loader.set_description((f'epoch: {epoch}; mse: {loss.item():.5f};'))
        seen_labels = None #update_seen_labels(batch_labels,seen_labels)

        if count % int(0.9*max_iter) == 0:
            test_data, j = next(test_iter)
            progress_out(vae, test_data[1], checkpoint_folder)
        #elif count % 500 == 0: not for RED GREEN
         #   data = data_noSkip[0][1] + data_skip[0]
          #  progress_out(vae, data, epoch, count, skip= True)
        
        if i == max_iter +1:
            break

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader_noSkip.dataset)))
    
    if return_loss is True:
        # get test losses for cropped and retinal
        test_data = next(test_iter)
        test_data = test_data[0]

        test_loss_dict = test_loss(vae, test_data, ['retinal', 'cropped'])
    
        return [retinal_loss_train, test_loss_dict['retinal'], cropped_loss_train, test_loss_dict['cropped']], seen_labels