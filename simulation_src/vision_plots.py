# simulations that do not involve the binding pool
colornames = ["red", "blue","green","purple","yellow","cyan","orange","brown","pink","teal"]

# prerequisites
import torch
import sys
import os
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import save_image
import math
from scipy import stats
import gc
from PIL import Image

#internal imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MLR_src.dataset_builder import Dataset, Colorize_specific
import MLR_src.mVAE as mlr
from MLR_src.BP_functions import BPTokens_binding_all, BPTokens_retrieveByToken, BPTokens_storage, BPTokens_with_labels
import random


from PIL import Image, ImageOps, ImageEnhance#, __version__ as PILLOW_VERSION
convert_tensor = transforms.ToTensor()
convert_image = transforms.ToPILImage()

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA')
else:
    device = 'cpu'

imgsize = 28
retina_size = 64

@torch.no_grad()
def fig_2n(vae, folder_path):
    print('bengali reconstructions')
    vae.eval()
    all_imgs = []
    imgsize = 28
    numimg = 6

    #load in some examples of Bengali Characters
    for i in range (1,numimg+1):
        color = Colorize_specific(random.randint(0,9))
        img = Image.open(f'data/current_bengali/{i}_thick.png')
        img_new = convert_tensor(color(img))
        
        all_imgs.append(img_new)
    all_imgs = torch.stack(all_imgs)
    imgs = all_imgs.view(-1, 3, imgsize, imgsize).cuda()
    activations = vae.activations(imgs, False)
    z_img = activations['shape']
    z_color = activations['color']
    hskip = activations['skip']
    recon_sample = vae.decoder_cropped(z_img, z_color, 0)
    output = vae.decoder_skip_cropped(0, 0, 0, hskip)
    out_img = torch.cat([imgs.view(-1, 3, imgsize, imgsize),output,recon_sample],dim=0)

    save_image(out_img,f'{folder_path}bengali_recon.png', numimg)

@torch.no_grad()
def fig_retinal_mod(vae, folder_path):
    vae.eval()
    bs = 10
    mnist_transforms = {'retina':True, 'colorize':True, 'scale':True}
    mnist_loader= Dataset('mnist', mnist_transforms).get_loader(bs)
    
    dataiter_mnist = iter(mnist_loader)
    data, labels = next(dataiter_mnist)
    data = data[0].cuda()

    activations = vae.activations(data, True)
    
    shape = activations['shape']
    color = activations['color']
    scale = activations['scale']
    location = activations['location']

    mod_location = torch.tensor([-0.4, -0.4] * bs).view(bs,2).cuda()
    mod_scale = torch.tensor([0.8] * bs).view(bs,1).cuda()

    theta = torch.cat([scale, location], 1)
    theta_mod_loc = torch.cat([scale, mod_location], 1)
    theta_mod_scale = torch.cat([mod_scale, location], 1)
    theta_mod_both = torch.cat([mod_scale, mod_location], 1)

    recon = vae.decoder_retinal(shape, color, theta)
    recon_mod_loc = vae.decoder_retinal(shape, color, theta_mod_loc)
    recon_mod_scale = vae.decoder_retinal(shape, color, theta_mod_scale)
    recon_mod_both = vae.decoder_retinal(shape, color, theta_mod_both)

    save_image(
        torch.cat([data, recon, recon_mod_scale, recon_mod_loc, recon_mod_both], 0),
        f'{folder_path}figure_retinal_mod.png', pad_value=0.6,
        nrow=bs, normalize=False)

   

def fig_loc_compare(vae, folder_path):
    vae.eval()
    bs = 15
    mnist_transforms = {'retina':True, 'colorize':True, 'scale':False, 'location_targets':{'left':[0,1,2,3,4],'right':[5,6,7,8,9]}}
    mnist_test_transforms = {'retina':True, 'colorize':True, 'scale':False, 'location_targets':{'right':[0,1,2,3,4],'left':[5,6,7,8,9]}}
    train_loader_noSkip = Dataset('mnist',mnist_transforms).get_loader(bs)
    test_loader_noSkip = Dataset('mnist',mnist_test_transforms, train=False).get_loader(bs)
    imgsize = 28
    numimg = 10
    
    dataiter_noSkip_test = iter(test_loader_noSkip)
    dataiter_noSkip_train = iter(train_loader_noSkip)
    #skipd = iter(train_loader_skip)
    #skip = skipd.next()
    #print(skip[0].size())
    #print(type(dataiter_noSkip_test))
    data_test = next(dataiter_noSkip_test)
    data_train = next(dataiter_noSkip_train)

    data = data_train[0].copy()

    data[0] = torch.cat((data_test[0][0], data_train[0][0]),dim=0) #.cuda()
    data[1] = torch.cat((data_test[0][1], data_train[0][1]),dim=0)
    data[2] = torch.cat((data_test[0][2], data_train[0][2]),dim=0)

    sample = data
    sample_size = 15
    #print(sample[0].size(),sample[1].size(),sample[2].size())
    with torch.no_grad():
        reconl, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location,mu_scale,log_var_scale = vae(sample, 'location') #location
        reconb, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location,mu_scale,log_var_scale = vae(sample, 'retinal') #retina

    line1 = torch.ones((1,2)) * 0.5
    line2 = line1.view(1,1,1,2)
    line3 = line2.expand(sample_size, 3, retina_size, 2).cuda()
    line2 = line2.expand(sample_size*2, 3, retina_size, 2).cuda()
    #reconb = reconb['recon']

    shape_color_dim = retina_size + 2
    shape_color_dim1 = imgsize + 2
    sample = torch.cat((sample[0].cuda(),line2),dim = 3).cuda()
    reconb = torch.cat((reconb,line2.cuda()),dim = 3).cuda()
    shape_color_dim = retina_size + 2
    sample_test = sample[:sample_size] #torch.cat((sample[0][:sample_size],line3),dim = 3).cuda()
    sample_train = sample[sample_size:] # torch.cat((sample[0][sample_size:],line3),dim = 3).cuda()
    utils.save_image(
        torch.cat([sample_train.view(sample_size, 3, retina_size, shape_color_dim), reconb[sample_size:(2*sample_size)].view(sample_size, 3, retina_size, shape_color_dim), 
                   sample_test.view(sample_size, 3, retina_size, shape_color_dim), reconb[:(sample_size)].view(sample_size, 3, retina_size, shape_color_dim)], 0),
        f'{folder_path}figure_location_compare.png',
        nrow=sample_size, normalize=False)
    
    image_pil = Image.open(f'{folder_path}figure_location_compare.png')
    trained_label = "Trained Data"
    untrained_label = "Untrained Data"
    # Add trained and untrained labels to the image using PIL's Draw module
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.load_default()  # You can choose a different font or size

    # Trained data label at top left
    trained_label_position = (10, 10)  # Adjust the position of the text
    draw.text(trained_label_position, trained_label, fill=(255, 255, 255), font=font)

    # Untrained data label at bottom left
    image_width, image_height = image_pil.size
    untrained_label_position = (10, image_height//2)  # Adjust the position of the text
    draw.text(untrained_label_position, untrained_label, fill=(255, 255, 255), font=font)

    # Save the modified image with labels
    image_pil.save(f'{folder_path}figure_location_compare.png')

    print("Images with labels saved successfully.")
    return []

def obj_back_swap(vae, device, folder_path):
    print('swapping obj-background')
    imgsize= 28

    #data
    cifar_transforms = {'retina':False, 'colorize':False, 'scale':False}
    mnist_transforms = {'retina':False, 'colorize':True, 'scale':False}

    mnist_dataloader = iter(Dataset('mnist', mnist_transforms).get_loader(2))
    cifar_dataloader = iter(Dataset('cifar10', cifar_transforms).get_loader(10))

    mnist_data = next(mnist_dataloader)[0].to(device)
    cifar_data = next(cifar_dataloader)[0].to(device)

    data = torch.cat([mnist_data, cifar_data]) #, torch.zeros(1,3,28,28).to(device)

    z_back_w, z_obj_w, junk_theta = vae.activations(torch.zeros(len(data),3,imgsize,imgsize).to(device))
    #z_back_w = vae.encode_back(torch.ones(len(data),3,imgsize,imgsize).to(device))
    z_back, z_obj, obj_theta  = vae.activations(data)
    act_in = {'shape':[z_back, 1], 'color':[z_obj, 1]}
    BPOut_all, Tokenbindings_all = BPTokens_storage(bpsize, bpPortion, act_in, 12,normalize_fact_novel)
        
    act_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut_all, Tokenbindings_all, act_in, 12,normalize_fact_novel)


    out = vae.decoder_cropped(z_back, z_obj, 0, obj_theta )
    out_bp = vae.decoder_cropped(act_out['shape'], act_out['color'], 0, obj_theta )
    out_obj = vae.decoder_color(z_back, z_obj, 0, obj_theta )
    out_back = vae.decoder_shape(z_back, z_obj, 0, obj_theta )
    out_swap = vae.decoder_cropped(torch.cat((z_back[-1:].clone(), z_back[:-1]), dim=0), z_obj, 0, obj_theta )
    out_back_w = vae.decoder_cropped(z_back_w, z_obj, 0, obj_theta )
    out_obj_w = vae.decoder_cropped(z_back, z_obj_w, 0, obj_theta )

    utils.save_image(
        torch.cat([data.view(-1, 3, imgsize, imgsize).cpu(), out.view(-1, 3, imgsize, imgsize).cpu(),
                    out_bp.view(-1, 3, imgsize, imgsize).cpu(),
                    out_back.view(-1, 3, imgsize, imgsize).cpu(), out_obj.view(-1, 3, imgsize, imgsize).cpu(),
                    out_swap.view(-1, 3, imgsize, imgsize).cpu(), out_back_w.view(-1, 3, imgsize, imgsize).cpu(),
                    out_obj_w.view(-1, 3, imgsize, imgsize).cpu() ], 0),
        f"{folder_path}figure_swap_background1.png",
        nrow=len(data), normalize=False)