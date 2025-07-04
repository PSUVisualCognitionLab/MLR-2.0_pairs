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
from MLR_src.vit import ViT, PatchEmbed
from MLR_src.vit_decode import ViTDecoder
from MLR_src.latent_compete import ViTSlotCompetition, refine_masks_and_images, create_patch_masks

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
h_dim2 = 300 #128
z_dim = 256
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

        # localization network for stn
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
            nn.Linear(32, 4), # 3
        )

        # localization network for crop stn
        self.obj_localization = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(4,16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4,32),#BatchNorm2d(32),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4,16),#BatchNorm2d(16),
            nn.ReLU(),)

        self.obj_regressor = nn.Sequential(
            nn.Linear(int(imgsize / 4) * int(imgsize / 4)*16, 32),
            nn.ReLU(),
            nn.Linear(32, 4), # 3
        )

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.1)
        self.sparse_relu = nn.Threshold(threshold=0.5, value=0)

        self.patch_embed = PatchEmbed(
            image_size=28,
            patch_size=4,
            dim=512)

        self.vit_obj_encoder = ViT(
            image_size=28,
            patch_size=4,
            num_classes=h_dim2,
            dim=512,
            depth=6,
            heads=16,
            mlp_dim=128,
            patch_embed=self.patch_embed,
            dropout=0.1,
            emb_dropout=0.1,
            masked='center'
        )

        self.vit_back_encoder = ViT(
            image_size=28,
            patch_size=4,
            num_classes=h_dim2,
            dim=512,
            depth=6,
            heads=16,
            mlp_dim=128,
            patch_embed=self.patch_embed,
            dropout=0.1,
            emb_dropout=0.1,
            masked='outer'
        )

        self.vit_decoder = ViTDecoder(
            latent_dim = 2*h_dim2,
            num_tokens = 16,
            embed_dim = 512,
            num_layers = 6,
            n_heads = 16,
            patch_size = 7,
            img_size = 28
        )

        self.vit_enable = True
        self.compete_enable = True

        # map scalars
        self.shape_scale = 1 #1.9 #obj
        self.color_scale = 1 #2 #background

    def construct_theta(self, z_where):
        # [s,x,y,r] -> [[s,0,x],
        #             [0,s,y]]
        n = z_where.size(0)
        out = torch.zeros(n, 2, 3).to(z_where.device)
        out[:,0,0] = z_where[:,0]
        #out[:,0,1] = z_where[:,3] #
        out[:,0,2] = z_where[:,1]
        #out[:,1,0] = z_where[:,3] #
        out[:,1,1] = z_where[:,0]
        out[:,1,2] = z_where[:,2]
        out = out.view(n, 2, 3)
       
        return out

    def invert_theta(self, z_where):
        # [s,x,y,r] -> [1/s,-x/s,-y/s,-r]
        n = z_where.size(0)
        out = torch.cat((z_where.new_ones(n, 1), -z_where[:, 1:3]), 1)  # Invert translation
        out = out / z_where[:, 0:1]  # Divide by scale
        out = torch.cat((out, -z_where[:, 3:4]), 1)  # Invert rotation
        return out

    def stn_encode(self, x):
        B = x.shape[0]
        # x is [B, 3, 64, 64]
        z=self.localization(x)
        theta = self.regressor(z.view(-1, int(retina_size / 4) * int(retina_size / 4)*16))  # [B, 2, 3]
        theta = theta.view(-1, 4).to(x.device)
        grid = F.affine_grid(self.construct_theta(theta), (B,3,64,64), align_corners=False)  # same 64×64 resolution
        x_transformed = F.grid_sample(x, grid, align_corners=False)

        # crop by slicing out the 28×28 region centered by the stn
        crop = x_transformed[:, :, 18:46, 18:46]
        return crop, theta

    def stn_decode(self, crop, theta):
        B = crop.shape[0]

        # put the 28×28 crop back into a 64×64 retina
        canvas = torch.zeros(B, 3, 64, 64, device=crop.device)
        #canvas[:, :, 18:46, 18:46] = crop  # place the 28×28 patch back
        #canvas[:, :, 18:46, 18:46] = torch.rot90(crop, k=2, dims=(-2, -1))
        theta = self.invert_theta(theta)
        theta = self.construct_theta(theta).to(crop.device)
        
        # reconstruct full retina
        grid = F.affine_grid(theta, (B, 3, 64, 64), align_corners=False)
        retina = F.grid_sample(canvas, grid, align_corners=False)

        return retina

    def obj_stn_encode(self, x):
        B = x.shape[0]
        # x is [B, 3, 64, 64]
        z=self.obj_localization(x)
        theta = self.obj_regressor(z.view(-1, int(imgsize / 4) * int(imgsize / 4)*16))  # [B, 2, 3]
        theta = theta.view(-1, 4).to(x.device)
        grid = F.affine_grid(self.construct_theta(theta), (B,3,28,28), align_corners=False)  # same 64×64 resolution
        x_transformed = F.grid_sample(x, grid, align_corners=False)
        return x_transformed, theta

    def obj_stn_decode(self, crop, theta):
        B = crop.shape[0]

        # put the 28×28 crop back into a 64×64 retina
        #canvas[:, :, 18:46, 18:46] = crop  # place the 28×28 patch back
        #canvas[:, :, 18:46, 18:46] = torch.rot90(crop, k=2, dims=(-2, -1))
        theta = self.invert_theta(theta)
        theta = self.construct_theta(theta).to(crop.device)
        
        # reconstruct full retina
        grid = F.affine_grid(theta, (B, 3, 28, 28), align_corners=False)
        crop = F.grid_sample(crop, grid, align_corners=False)

        return crop
    
    def encode_retina(self, retina):
        x, theta = self.stn_encode(retina)
        mu_shape, log_var_shape, mu_color, log_var_color, hskip = self.encoder(x)
        z_shape = self.sampling(mu_shape, log_var_shape)
        z_color = self.sampling(mu_color, log_var_color)

        return z_shape, z_color, theta
    
    def encoder(self, x):
        if self.vit_enable == True:
            device = next(self.parameters()).device
            hskip = torch.tensor([0]).to(device)
            #print(obj_init_mask.size())
            x, theta = self.obj_stn_encode(x)
            
            # Step 2: First-stage encoding
            h1 = self.vit_obj_encoder(x)
            h2 = self.vit_back_encoder(x)
            
            h1 = self.vit_obj_encoder.patch_to_latent(h1)
            h2 = self.vit_back_encoder.patch_to_latent(h2)
        else:
            b_dim = x.size(0)
            h = self.sparse_relu(self.bn1(self.conv1(x)))
            hskip = h.view(b_dim,-1)
            h = self.relu(self.bn2(self.conv2(h)))        
            h = self.relu(self.bn3(self.conv3(h)))
            h = self.relu(self.bn4(self.conv4(h)))
            h = h.view(-1,int(imgsize / 4) * int(imgsize / 4)*16)
            h = self.relu(self.fc_bn2(self.fc2(h)))
            h1 = h
            h2 = h
            theta = 0 

        return self.fc31(h1), self.fc32(h1), self.fc33(h2), self.fc34(h2), hskip, theta # mu, log_var

    def encode_back(self, x):
        h2 = self.vit_back_encoder(x)
        #h2 = h2.mean(dim=1) if self.vit_back_encoder.pool == 'mean' else h2[:, 0]
        h2 = self.vit_back_encoder.patch_to_latent(h2)
        return self.sampling(self.fc33(h2), self.fc34(h2))

    def activations(self, x, activations, retinal=False): # returns shape, color, location latent activations
        if retinal is True:    #passing in a cropped+ location as input
            x, theta = self.stn_encode(x)
            mu_shape, log_var_shape, mu_color, log_var_color, hskip, obj_theta = self.encoder(x)
        else:  #passing in just cropped image
            mu_shape, log_var_shape, mu_color, log_var_color, hskip, obj_theta = self.encoder(x)
        
        z_shape = self.sampling(mu_shape, log_var_shape) #background
        z_color = self.sampling(mu_color, log_var_color) #obj
        z_scale = theta[0]
        z_location = theta[1:]

        return z_shape, z_color, obj_theta
    
    def activations_l1(self, x): # returns skip connection activations
        if type(x) == list or type(x) == tuple:    #passing in a cropped+ location as input
            l = x[2].cuda()
            #sc = x[3].cuda()
            x = x[1].cuda()
            mu_shape, log_var_shape, mu_color, log_var_color, mu_location, log_var_location, mu_scale, log_var_scale, hskip = self.encoder(x, l)
        else:  #passing in just cropped image
            x = x.cuda()
            #sc = torch.zeros(x.size()[0], sc_dim).cuda()
            l = torch.zeros(x.size()[0], self.l_dim).cuda()
            mu_shape, log_var_shape, mu_color, log_var_color, mu_location, log_var_location, mu_scale, log_var_scale, hskip = self.encoder(x, l)
        
        return hskip

    def location_encoder(self, l):
        return self.sampling_location(self.fc35(l), self.fc36(l))

    def sampling_location(self, mu, log_var):
        std = (0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder_location(self, z_shape, z_color, z_location):
        h = self.fc4l(z_location)
        return torch.sigmoid(h).view(-1,2,retina_size)

    def decoder_scale(self, z_shape, z_color, z_scale):
        h = self.fc4sc(z_scale)
        return torch.sigmoid(h).view(-1,10)

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
            return {'recon':h}
        else:
            return h

    def decoder_color(self, z_shape, z_color, hskip, theta):
        #h = F.relu(self.fc4c(z_color)) * self.color_scale
        h = torch.stack([F.relu(self.fc4c(z_color)), torch.zeros_like(self.fc4c(z_color))], 1)
        if self.vit_enable == True:
            h = h.view(z_color.size(0), -1)
            h = self.vit_decoder(h)
            h = self.obj_stn_decode(h, theta)
        else:    
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
            
        return torch.sigmoid(h).view(-1, 3, imgsize, imgsize)

    def decoder_shape(self, z_shape, z_color, hskip, theta):
        #h = F.relu(self.fc4s(z_shape)) * self.shape_scale
        #print('shape',z_shape.size())
        h = torch.stack([torch.zeros_like(self.fc4s(z_shape)), F.relu(self.fc4s(z_shape))], 1)
        if self.vit_enable == True:
            h = h.view(z_shape.size(0), -1)
            h = self.vit_decoder(h)
            h = self.obj_stn_decode(h, theta)
        else:
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
        return torch.sigmoid(h).view(-1, 3, imgsize, imgsize)

    def decoder_cropped(self, z_shape, z_color, hskip, theta):
        #print('crop',z_shape.size())
        h = torch.stack([F.relu(self.fc4c(z_color)), F.relu(self.fc4s(z_shape))], 1)
        if self.vit_enable == True:
            h = h.view(z_shape.size(0), -1)
            h = self.vit_decoder(h)
            h = self.obj_stn_decode(h, theta)
        
        else:
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
        return torch.sigmoid(h).view(-1, 3, imgsize, imgsize)

    def decoder_skip_cropped(self, z_shape, z_color, z_location, hskip):
        #mu_skip = self.fc8(hskip)
        #log_var_skip = self.fc9(hskip)
        #hskip = self.sampling(mu_skip, log_var_skip)
        h= self.fc8(hskip)#hskip#
        if self.training:
            h = self.dropout(h)
        h = self.relu(self.skip_bn(h.view(-1,16,28,28))) #self.skip_bn(h.view(-1,16,28,28)) self.skip_bn(h.view(-1,32,14,14))
        #h = self.relu(self.fc9(h))
        #h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize/4), int(imgsize/4))
        #h = self.relu(self.bn5(self.conv5(h.view(-1,16,7,7))))
        #h = self.relu(self.bn6(self.conv6(h.view(-1,64,7,7))))
        #h = self.relu(self.bn7(self.conv7(h.view(-1,32,14,14)))) #
        #ind = h[:,784:].long()
        #h = h[:,:784]
        #h = self.unpool(h.view(-1,28,28),ind.view(-1,28,28))
        h = self.conv8(h.view(-1,16,28,28)).view(-1, 3, imgsize, imgsize) #skip.view(-1,16,28,28)
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

    '''def activations(self, z_shape, z_color, z_location):
        h = F.relu(self.fc4c(z_color)) + F.relu(self.fc4s(z_shape)) + F.relu(self.fc4l(z_location))
        fc4c = self.fc4c(z_color)
        fc4s = self.fc4s(z_shape)
        fc4l = self.fc4l(z_location)
        fc5 = self.fc5(h)
        return fc4c, fc4s, fc4l, fc5
    '''
    def forward_layers(self, l1, l2, layernum, whichdecode):
        hskip = l1
        if layernum == 1:
            h = F.relu(self.bn2(self.conv2(l1)))
            h = self.relu(self.bn3(self.conv3(h)))
            h = self.relu(self.bn4(self.conv4(h)))
            h = h.view(-1, int(imgsize / 4) * int(imgsize / 4) * 16)
            h = self.relu(self.fc_bn2(self.fc2(h)))
            hskip = self.fc8(h)
            mu_shape = self.fc31(h)
            log_var_shape = self.fc32(h)
            mu_color = self.fc33(h)
            log_var_color = self.fc34(h)
            z_shape = self.sampling(mu_shape, log_var_shape)
            z_color = self.sampling(mu_color, log_var_color)

        elif layernum == 2:
            h = self.relu(self.bn3(self.conv3(l2)))
            h = self.relu(self.bn4(self.conv4(h)))
            h = h.view(-1, int(imgsize / 4) * int(imgsize / 4) * 16)
            h = self.relu(self.fc_bn2(self.fc2(h)))
            hskip = self.fc8(h)
            mu_shape = self.fc31(h)
            log_var_shape = self.fc32(h)
            mu_color = self.fc33(h)
            log_var_color = self.fc34(h)
            z_shape = self.sampling(mu_shape, log_var_shape)
            z_color = self.sampling(mu_color, log_var_color)

        elif layernum == 3:
            h=hskip
            #hskip = F.relu(hskip)
            #h = self.relu(self.fc9(hskip))
            #l1 = F.relu(self.fc9(hskip))
            #ind = hskip[:,784:].long()
            #h = hskip[:,:784]
            #l1 = self.unpool(h.view(-1,28,28),ind.view(-1,28,28))
            h = self.relu(self.bn2(self.conv2(h.view(-1,16,28,28)))) ####
            
            h = h.view(-1,32,14,14)
            h = self.relu(self.bn3(self.conv3(h)))
            h = self.relu(self.bn4(self.conv4(h)))
            h = h.view(-1, int(imgsize / 4) * int(imgsize / 4) * 16)
            h = self.relu(self.fc_bn2(self.fc2(h)))
            mu_shape = self.fc31(h)
            #print(f'{whichdecode}',l1.size()[0],mu_shape.size())
            log_var_shape = self.fc32(h)
            mu_color = self.fc33(h)
            log_var_color = self.fc34(h)
            z_shape = self.sampling(mu_shape, log_var_shape)
            z_color = self.sampling(mu_color, log_var_color)

        if (whichdecode == 'cropped'):
            output = self.decoder_cropped(z_shape, z_color, 0, hskip)
        elif (whichdecode == 'skip_cropped'):
            output = self.decoder_skip_cropped(z_shape, z_color, 0, hskip)

        return output, mu_color, log_var_color, mu_shape, log_var_shape

    def forward(self, x, whichdecode='noskip', keepgrad=[]):
        if type(x) == list or type(x) == tuple:    #passing in full retina
            xx = x[0].cuda()
            xx, theta = self.stn_encode(xx)
            x = x[1].cuda()
            mu_shape, log_var_shape, mu_color, log_var_color, hskip, obj_theta = self.encoder(x)
        else:  #passing in just cropped image
            x = x.cuda()
            mu_shape, log_var_shape, mu_color, log_var_color, hskip, obj_theta = self.encoder(x)

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
            output = self.decoder_cropped(z_shape,z_color, 0, hskip, obj_theta)
        elif (whichdecode == 'retinal'):
            output = self.decoder_retinal(z_shape,z_color, theta)
        elif (whichdecode == 'skip_cropped'):
            output = self.decoder_skip_cropped(0, 0, 0, hskip)
        elif (whichdecode == 'skip_retinal'):
            output = self.decoder_skip_retinal(0, 0, 0, hskip)
        elif (whichdecode == 'color'):
            output = self.decoder_color(0, z_color , 0, obj_theta)
        elif (whichdecode == 'shape'):
            output = self.decoder_shape(z_shape,0, 0, obj_theta)
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
    # here's a loss BCE based only on the grayscale reconstruction.  Use this in the return statement to kill color learning
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
def progress_out(vae, data, epoch, count, skip = False, filename = None):
    sample_size = 25
    vae.eval()
    #make a filename if none is provided
    if filename == None:
        filename = f'sample_{vae_type_flag}_{data_set_flag}/{str(epoch + 1).zfill(5)}_{str(count).zfill(5)}.png'
        filename1 = f'sample_{vae_type_flag}_{data_set_flag}/{str(epoch + 1).zfill(5)}_crop_{str(count).zfill(5)}.png'

    if skip:
        sample = data[:sample_size]
        with torch.no_grad():
            shape_color_dim = imgsize
            reconds, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'skip_cropped') #digit from skip
            recond, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'cropped') #digit
            reconc, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'color') #color
            recons, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'shape') #shape
            utils.save_image(
            torch.cat([sample.view(sample_size, 3, imgsize, shape_color_dim).cuda(), reconds.view(sample_size, 3, imgsize, shape_color_dim).cuda(), recond.view(sample_size, 3, imgsize, shape_color_dim).cuda(),
                    reconc.view(sample_size, 3, imgsize, shape_color_dim).cuda(), recons.view(sample_size, 3, imgsize, shape_color_dim).cuda()], 0),
            filename,
            nrow=sample_size, normalize=False, range=(-1, 1),)

    else:
        sample = data
        #print('\n',len(sample))
        with torch.no_grad():
            #reconl, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location,mu_scale,log_var_scale = vae(sample, 'location') #location
            reconb, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location,mu_scale,log_var_scale = vae(sample, 'retinal') #retina
            recond, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location,mu_scale,log_var_scale = vae(sample, 'cropped') #digit
            reconc, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location,mu_scale,log_var_scale = vae(sample, 'color') #color
            recons, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location,mu_scale,log_var_scale = vae(sample, 'shape') #shape

        crop_retina = place_crop(reconb['crop'].cuda(), sample[2].cuda())
        reconb = reconb['recon'].cuda()
        loc_background = torch.zeros(sample_size,3,retina_size-2,retina_size).cuda()
        line1 = torch.ones((1,2)) * 0.5
        line1 = line1.view(1,1,1,2)
        line2 = line1.view(1,1,1,2)
        #line3 = line1.expand(sample_size, 3, 2, 2).cuda()
        line1 = line1.expand(sample_size, 3, imgsize, 2).cuda()
        line2 = line2.expand(sample_size, 3, retina_size, 2).cuda()
        

        #reconl = reconl.view(sample_size,1,2,retina_size)
        #reconl = reconl.expand(sample_size,3,2,retina_size)
        n_reconc = torch.cat((reconc,line1),dim = 3).cuda()
        n_recons = torch.cat((recons,line1),dim = 3).cuda()
        #n_reconl = torch.cat((reconl,loc_background),dim = 2).cuda()
        #n_reconl = torch.cat((n_reconl,line2),dim = 3).cuda()
        n_recond = torch.cat((recond,line1),dim = 3).cuda()
        crop_retina = torch.cat((crop_retina.cuda(),line2.cuda()),dim = 3).cuda()
        shape_color_dim = retina_size + 2
        shape_color_dim1 = imgsize + 2
        sample = torch.cat((sample[0].cuda(),line2),dim = 3).cuda()
        reconb = torch.cat((reconb,line2.cuda()),dim = 3).cuda()

        utils.save_image(
            torch.cat([sample.view(sample_size, 3, retina_size, shape_color_dim)[:25], crop_retina.view(sample_size, 3, retina_size, shape_color_dim)[:25], reconb.view(sample_size, 3, retina_size, shape_color_dim)[:25]], 0),
            filename,
            nrow=sample_size, normalize=False)

        utils.save_image(
            torch.cat([n_recond.view(sample_size, 3, imgsize, shape_color_dim1)[:25], n_reconc.view(sample_size, 3, imgsize, shape_color_dim1)[:25], n_recons.view(sample_size, 3, imgsize, shape_color_dim1)[:25]], 0),
            filename1,
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

def train(vae, optimizer, epoch, dataloaders, return_loss = False, seen_labels = {}, components = {}, max_iter=60):
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
    #loader=tqdm(train_loader_noSkip, total = max_iter)
    track_enable = False
    c = torch.ones(1,3,28,28)
    c[:,:,6:22,6:22] = torch.ones(1,3,16,16) * 0.4
    c = c.to(next(vae.parameters()).device)
    b = torch.zeros(1,3,28,28).to(next(vae.parameters()).device)
    r = torch.ones(1,3,28,28).to(next(vae.parameters()).device)
    #dataiter_noSkip = dataiter_noSkip.to(next(vae.parameters()).device)

    retinal_loss_train, cropped_loss_train = 0, 0 # loss metrics returned to Training.py
    block_data, block_labels = next(block_iter)

    for i in enumerate(range(max_iter)): #for i,j in enumerate(loader): to use tqdm
        count += 1
        data, batch_labels = next(dataiter_noSkip)

        # shuffle in the block dataset
        z = random.randint(0,10)
        if z < 0:
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
        
        if  whichdecode_use == 'cropped' and 'retinal' not in components:
            data_in = data[1].to(next(vae.parameters()).device)       
            if count % 10 == 0:
                data_in = torch.cat([c, b, r, data_in], 0)
        else:
            data_in = data
        
        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape = vae(data_in, whichdecode_use, keepgrad)

        if count >= 0.9*max_iter:
            recon_back, mu_color, log_var_color, mu_shape, log_var_shape = vae(data_in, 'shape')
            recon_obj, mu_color, log_var_color, mu_shape, log_var_shape = vae(data_in, 'color')

            
        if whichdecode_use == 'shape':  # shape
            loss = loss_function_shape(recon_batch, data, mu_shape, log_var_shape)

        elif whichdecode_use == 'color': # color
            loss = loss_function_color(recon_batch, data, mu_color, log_var_color)

        elif whichdecode_use == 'retinal': # retinal
            #loss = loss_function(recon_batch['recon'], data, recon_batch['crop'])
            loss = loss_function(recon_batch['recon'], data, None)
            retinal_loss_train = loss.item()
            if count >= 0.1*max_iter:
                utils.save_image(
                    torch.cat([data[0].view(-1, 3, retina_size, retina_size)[:25].cpu(), recon_batch['recon'].view(-1, 3, retina_size, retina_size)[:25].cpu() 
                               #,place_crop(recon_batch['crop'],data[2]).view(-1, 3, retina_size, retina_size)[:25].cpu()
                               ], 0),
                    f"retinal_recon_test{epoch%5}.png",
                    nrow=25, normalize=False)

        elif whichdecode_use == 'cropped': # cropped
            loss = loss_function_crop(recon_batch, data_in)
            cropped_loss_train = loss.item()
            if count >= 0.9*max_iter:
                utils.save_image(
                    torch.cat([data_in.view(-1, 3, imgsize, imgsize)[:25].cpu(), recon_batch.view(-1, 3, imgsize, imgsize)[:25].cpu(),
                               recon_back.view(-1, 3, imgsize, imgsize)[:25].cpu(), recon_obj.view(-1, 3, imgsize, imgsize)[:25].cpu()
                               #,place_crop(recon_batch['crop'],data[2]).view(-1, 3, retina_size, retina_size)[:25].cpu()
                               ], 0),
                    f"cropped_recon_test{epoch%5}.png",
                    nrow=25, normalize=False)

        elif whichdecode_use == 'skip_cropped': # skip training
            loss = loss_function_crop(recon_batch, data)

        elif whichdecode_use == 'scale': # scale training
            loss = loss_function_crop(recon_batch, data)
        
        #l1_norm = sum(p.abs().sum() for p in vae.parameters())
        #loss += l1_norm*0.0001
        loss.backward()

        train_loss += loss.item()
        optimizer.step()
        if track_enable:
            loader.set_description((f'epoch: {epoch}; mse: {loss.item():.5f};'))
        seen_labels = None #update_seen_labels(batch_labels,seen_labels)
    
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

#compute avg loss of retinal recon w/ skip, w/o skip, increase fc?
# no longer used
def test(whichdecode, test_loader_noSkip, test_loader_skip, bs):
    vae.eval()
    global numcolors
    test_loss = 0
    testiter_noSkip = iter(test_loader_noSkip)  # the latent space is trained on MNIST and f-MNIST
    testiter_skip = iter(test_loader_skip)  # The skip connection is trained on notMNIST
    with torch.no_grad():
        for i in range(1, len(test_loader_noSkip)): # get the next batch


            data = testiter_noSkip.next()
            data = data[0]
            recon, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(data, 'retinal')

            # sum up batch loss
            #test_loss += loss_function_shape(recon, data, mu_shape, log_var_shape).item()
            #test_loss += loss_function_color(recon, data, mu_color, log_var_color).item()
            test_loss += loss_function(recon, data, mu_shape, log_var_shape, mu_color, log_var_color).item()

    print('Example reconstruction')
    datac = data[0].cuda()
    datac=datac.view(bs, 3, imgsize, retina_size)
    save_image(datac[0:8], f'{args.dir}/orig.png')
    pos = torch.zeros((64,100)).cuda()
    for i in range(len(pos)):
        pos[i][random.randint(0,99)] = 1
    pos_mu = vae.fc35(pos)
    pos_logvar = vae.fc36(pos)

    # current imagining of shape and color results in random noise
    # generate a
    print('Imagining a shape')
    with torch.no_grad():  # shots off the gradient for everything here
        zc = torch.randn(64, z_dim).cuda() * 0
        zs = torch.randn(64, z_dim).cuda() * 1
        zl = vae.sampling(pos_mu, pos_logvar)
        sample = vae.decoder_retinal(zs, zc, zl, 0).cuda()
        sample=sample.view(64, 3, imgsize, retina_size)
        save_image(sample[0:8], f'{args.dir}/sampleshape.png')


    print('Imagining a color')
    with torch.no_grad():  # shots off the gradient for everything here
        zc = torch.randn(64, z_dim).cuda() * 1
        zs = torch.randn(64, z_dim).cuda() * 0
        zl = vae.sampling(pos_mu, pos_logvar)
        sample = vae.decoder_retinal(zs, zc, zl, 0).cuda()
        sample=sample.view(64, 3, imgsize, retina_size)
        save_image(sample[0:8], f'{args.dir}/samplecolor.png')

    test_loss /= len(test_loader_noSkip.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
