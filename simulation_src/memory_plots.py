# simulations that do utilize the binding pool
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


bs_testing = 1000     # number of images for testing. 20000 is the limit
shape_coeff = 1       #cofficient of the shape map
color_coeff = 1       #coefficient of the color map
location_coeff = 0    #Coefficient of Location map
l1_coeff = 1          #coefficient of layer 1
l2_coeff = 1          #coefficient of layer 2
shapeLabel_coeff= 1   #coefficient of the shape label
colorLabel_coeff = 1  #coefficient of the color label
location_coeff = 0  #coefficient of the color label

bpsize = 10000#00         #size of the binding pool
token_overlap =0.2
bpPortion = int(token_overlap *bpsize) # number binding pool neurons used for each item

normalize_fact_familiar=1
normalize_fact_novel=1


imgsize = 28
BP_std = 0

# simulation helper functions:

def compute_correlation(x, y):
    assert x.shape == y.shape, "Tensors must have the same shape"
    
    # Flatten tensors if they're multidimensional
    x = x.view(-1)
    y = y.view(-1)
    #x = replace_near_zero(x)
    #y = replace_near_zero(y)
    
    # Compute means
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    
    # Compute the numerator
    numerator = torch.sum((x - x_mean) * (y - y_mean))
    
    # Compute the denominator
    x_var = torch.sum((x - x_mean)**2)
    y_var = torch.sum((y - y_mean)**2)
    denominator = torch.sqrt(x_var * y_var)
    
    # Compute correlation
    correlation = numerator / denominator
    
    return correlation

def rgb_to_gray(batch: torch.Tensor, keep_channels: bool = True) -> torch.Tensor:

    if batch.ndim != 4 or batch.size(1) != 3:
        raise ValueError("expected tensor shape (B, 3, 28, 28)")

    # ITU-R BT.601 luminance coefficients
    coeffs = torch.tensor([0.299, 0.587, 0.114],
                          dtype=batch.dtype, device=batch.device).view(1, 3, 1, 1)

    gray = (batch * coeffs).sum(dim=1, keepdim=True)        # (B, 1, 28, 28)

    if keep_channels:
        gray = gray.repeat(1, 3, 1, 1)                      # (B, 3, 28, 28)

    return gray

# store retrieve

# store and retreive bengali
@torch.no_grad()
def cd_r_acc_vs_setsize(vae, task):
    def compute_dprime(no_change_vector, change_vector):
        no_change_vector = np.array(no_change_vector)
        change_vector = np.array(change_vector)
        
        # hit rate and false alarm rate
        hit_rate = np.mean(change_vector)
        false_alarm_rate = 1 - np.mean(no_change_vector)
        hit_rate = np.clip(hit_rate, 0.01, 0.99)
        false_alarm_rate = np.clip(false_alarm_rate, 0.01, 0.99)
        
        # z-scores
        z_hit = stats.norm.ppf(hit_rate)
        z_fa = stats.norm.ppf(false_alarm_rate)
        d_prime = z_hit - z_fa
        
        return d_prime

    def compute_correlation(x, y):
        assert x.shape == y.shape, "Tensors must have the same shape"
        
        # Flatten tensors if they're multidimensional
        x = x.view(-1)
        y = y.view(-1)
        #x = replace_near_zero(x)
        #y = replace_near_zero(y)
        
        # Compute means
        x_mean = torch.mean(x)
        y_mean = torch.mean(y)
        
        # Compute the numerator
        numerator = torch.sum((x - x_mean) * (y - y_mean))
        
        # Compute the denominator
        x_var = torch.sum((x - x_mean)**2)
        y_var = torch.sum((y - y_mean)**2)
        denominator = torch.sqrt(x_var * y_var)
        
        # Compute correlation
        correlation = numerator / denominator
        
        return correlation
    
    def build_partial(input_tensor, n, out_x=1):
        x, b, channels, height, width = input_tensor.shape
        output_tensor = torch.zeros(x, channels, height, width, dtype=input_tensor.dtype, device=input_tensor.device)

        for batch_idx in range(x):
            for img_idx in range(n):
                output_tensor[batch_idx] += input_tensor[batch_idx, img_idx]

        # Clamp the output tensor to the range [0, 1]
        output_tensor = torch.clamp(output_tensor, min=0.0, max=1.0)

        return output_tensor

    vae.eval()
    max_set_size = 8 #8
    out_r = {0:[], 1:[]} #[]
    out_dprime = {0:[], 1:[]} # []
    threshold = {0:[], 1:[]} #[]
    setsize_range = [2,3,4, 6, 8] #range(2, max_set_size+1, 1)
    for t in range(0,2): # must start at 0
        for i in setsize_range:
            if t == 0:
                frame_count = 1
            else:
                frame_count = i
            
            with torch.no_grad():
                if t == 0:
                    original_t = torch.load(f'simulation_src/original_{i}{task}.pth').cpu()
                    change_t = torch.load(f'simulation_src/change_{i}{task}.pth').cpu()
                else:
                    original_t = torch.load(f'simulation_src/original_frames_{i}{task}.pth').cpu()
                    change_t = torch.load(f'simulation_src/change_{i}{task}.pth').cpu()

                print(len(original_t),len(change_t))
                r_lst0 = []
                r_lst1 = []
                no_change_detected = []
                change_detected = []
                for b in range(0,1): #20
                    print(i,b)
                    torch.cuda.empty_cache()
                    samples = 20
                    batch_id = b*samples
                    original = original_t[batch_id: batch_id+samples].cuda()
                    change = change_t[batch_id: batch_id+samples].cuda()
                    #original_frames = torch.load(f'original_frames_{i}.pth').cuda()
                    batch_size = original.size(0)
                    #print(original_frames.size())

                    # store block arrays in BP via L1
                    l1_original = []
                    for n in range(samples):
                        shape_act, color_act, location_act = vae.activations(original[n].view(1,3,28,28))
                        l1_act = vae.activations_l1(original[n].view(1,3,28,28))
                        l1_original += [l1_act]
                    #l1_change, l2_act, shape_act, color_act, location_act = activations(change)
                    bp_original_l1 = []
                    bp_change_l1 = []
                    bp_junk = torch.zeros(frame_count,1).cuda()
                    for n in range(batch_size):
                        
                        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, l1_original[n].view(frame_count,-1), bp_junk, shape_act,color_act,bp_junk,0, 0,0,1,0,frame_count,normalize_fact_novel)
                        shape_out_all, color_out_all, location_out_all, BP_layer2_out, BP_layerI_original = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, l1_original[1].view(frame_count,-1), bp_junk, shape_act,color_act,bp_junk,frame_count,normalize_fact_novel)
                        #BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, l1_change[n,:].view(1,-1), bp_junk, bp_junk,bp_junk,bp_junk,0, 0,0,1,0,1,normalize_fact_novel)
                        #shape_out_all, color_out_all, location_out_all, BP_layer2_out, BP_layerI_change = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, l1_change[1].view(1,-1), bp_junk, bp_junk,bp_junk,bp_junk,1,normalize_fact_novel)
                        if t == 0:
                            bp_original_l1 += [BP_layerI_original]
                        else:
                            bp_original_l1 += [BP_layerI_original] #[[shape_out_all, color_out_all]]
                        #bp_change_l1 += [BP_layerI_change]

                    #bp_original_l1 = torch.cat(bp_original_l1, dim=0)
                    #bp_change_l1 = torch.cat(bp_change_l1, dim=0)
                    original_BP = []
                    for n in range(len(original)):
                        if t == 0:
                            recon, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(bp_original_l1[n].view(frame_count,-1),BP_layer2_out,3, 'skip_cropped')
                        else:
                            recon = recon, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(bp_original_l1[n].view(frame_count,-1),BP_layer2_out,3, 'skip_cropped') #vae.decoder_cropped(bp_original_l1[n][0].view(frame_count,-1),bp_original_l1[n][1].view(frame_count,-1),0)
                            recon = build_partial(recon.view(1,frame_count,3,28,28), len(recon))
                        original_BP += [recon]
                    
                    if t == 1:
                        original = build_partial(original, len(original[0]))
                        #change = build_partial(change, len(change[0]))

                    #change_BP, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(bp_change_l1.view(batch_size,-1),BP_layer2_out,3, 'skip_cropped')

                    save_image(torch.cat([original[0].view(1,3,28,28), original_BP[0]],dim=0), f'comp_memory.png', nrow = 1, normalize=False)
                    print(len(original_BP))
                    for j in range(len(original)):
                        #x, y, z = original[i].cpu().detach().view(1,-1), original_BP[i].cpu().detach().view(1,-1), change_BP[i].cpu().detach().view(1,-1)
                        x, y, z = original[j], original_BP[j].view(3,28,28), change[j]
                        r_original = compute_correlation(y,x) #cosine_similarity(x,y) # 
                        r_change = compute_correlation(y,z)#cosine_similarity(x,z) #
                        
                        r_lst0 += [r_original]
                        r_lst1 += [r_change]
                    
                    
                    del original
                    del change
                    del original_BP
                    del l1_original
                    del shape_act 
                    del color_act
                    del location_act
                    gc.collect()
                    torch.cuda.empty_cache()

            print('computing threshold')    
            avg_r0 = sum(r_lst0)/(len(r_lst0))
            avg_r1 = sum(r_lst1)/(len(r_lst1))
            out_r[t] += [[avg_r0.item(), avg_r1.item()]]
            #print(out_r[t])

            c_threshold = (avg_r0.item() + avg_r1.item())/2
            
            for l in range(len(r_lst0)):
                r_original = r_lst0[l]
                r_change = r_lst1[l]

                if r_original > c_threshold:
                    no_change_detected += [1]
                else:
                    no_change_detected += [0]

                if r_change <= c_threshold:
                    change_detected += [1]
                else:
                    change_detected += [0]

            out_dprime[t] += [compute_dprime(no_change_detected, change_detected)]
            threshold[t] += [c_threshold]
    print(out_r[0])
    torch.save([out_r[0][i][0] for i in range(len(out_r[0]))], 'location_change_detect_r.pth')
    plt.plot(setsize_range, [out_r[0][i][0] for i in range(len(out_r[0]))], label='no change')
    plt.plot(setsize_range, [out_r[0][i][1] for i in range(len(out_r[0]))], label='change')
    plt.plot(setsize_range, threshold[0], label='threshold')
    plt.xlabel('set size')
    plt.ylabel('r')
    plt.legend()
    plt.title(f'color change detection, {batch_size*2} trials, BP: {bpPortion}')
    plt.savefig(f'change_detect{task}.png')
    plt.close()

    plt.plot(setsize_range, [out_r[1][i][0] for i in range(len(out_r[1]))], label='no change')
    plt.plot(setsize_range, [out_r[1][i][1] for i in range(len(out_r[1]))], label='change')
    plt.plot(setsize_range, threshold[1], label='threshold, compositional')
    plt.xlabel('set size')
    plt.ylabel('r')
    plt.legend()
    plt.title(f'color change detection compositonal memory, {batch_size*2} trials, BP: {bpPortion}')
    plt.savefig(f'change_detect_compositional{task}.png')
    plt.close()

    plt.plot(setsize_range, out_dprime[0], label=f'dprime for whole memory')
    plt.plot(setsize_range, out_dprime[1], label=f'dprime for compositional memory')
    plt.xlabel('set size')
    plt.ylabel('dprime')
    plt.legend()
    plt.title(f'color change dprime vs set size, {batch_size*2} trials, BP: {bpPortion}')
    plt.savefig(f'change_detect_accuracy{task}.png')

@torch.no_grad()
def cd_r_bp_cnn(vae, task, filepath):
    def compute_dprime(no_change_vector, change_vector):
        no_change_vector = np.array(no_change_vector)
        change_vector = np.array(change_vector)
        
        # hit rate and false alarm rate
        hit_rate = np.mean(change_vector)
        false_alarm_rate = 1 - np.mean(no_change_vector)
        hit_rate = np.clip(hit_rate, 0.01, 0.99)
        false_alarm_rate = np.clip(false_alarm_rate, 0.01, 0.99)
        
        # z-scores
        z_hit = stats.norm.ppf(hit_rate)
        z_fa = stats.norm.ppf(false_alarm_rate)
        d_prime = z_hit - z_fa
        
        return d_prime

    def compute_correlation(x, y):
        assert x.shape == y.shape, "Tensors must have the same shape"
        
        # Flatten tensors if they're multidimensional
        x = x.view(-1)
        y = y.view(-1)
        #x = replace_near_zero(x)
        #y = replace_near_zero(y)
        
        # Compute means
        x_mean = torch.mean(x)
        y_mean = torch.mean(y)
        
        # Compute the numerator
        numerator = torch.sum((x - x_mean) * (y - y_mean))
        
        # Compute the denominator
        x_var = torch.sum((x - x_mean)**2)
        y_var = torch.sum((y - y_mean)**2)
        denominator = torch.sqrt(x_var * y_var)
        
        # Compute correlation
        correlation = numerator / denominator
        
        return correlation
    
    def build_partial(input_tensor, n, out_x=1):
        x, b, channels, height, width = input_tensor.shape
        output_tensor = torch.zeros(x, channels, height, width, dtype=input_tensor.dtype, device=input_tensor.device)

        for batch_idx in range(x):
            for img_idx in range(n):
                output_tensor[batch_idx] += input_tensor[batch_idx, img_idx]

        # Clamp the output tensor to the range [0, 1]
        output_tensor = torch.clamp(output_tensor, min=0.0, max=1.0)

        return output_tensor

    vae.eval()
    max_set_size = 8 #8
    out_r = {0:[], 1:[]} #[]
    out_dprime = {0:[], 1:[]} # []
    threshold = {0:[], 1:[]} #[]
    setsize_range = [2,3,4, 6, 8] #range(2, max_set_size+1, 1)
    for t in range(0,2): # must start at 0
        for i in setsize_range:
            frame_count = 1
            
            with torch.no_grad():
                if t <= 1:
                    original_t = torch.load(f'simulation_src/original_{i}{task}.pth').cpu()
                    change_t = torch.load(f'simulation_src/change_{i}{task}.pth').cpu()
                else:
                    original_t = torch.load(f'simulation_src/original_frames_{i}{task}.pth').cpu()
                    change_t = torch.load(f'simulation_src/change_{i}{task}.pth').cpu()

                print(len(original_t),len(change_t))
                r_lst0 = []
                r_lst1 = []
                no_change_detected = []
                change_detected = []
                rr = 6 if (t==0) else 9
                for b in range(0,rr): #20
                    print(i,b)
                    torch.cuda.empty_cache()
                    samples = 20
                    batch_id = b*samples
                    original = original_t[batch_id: batch_id+samples].cuda()
                    change = change_t[batch_id: batch_id+samples].cuda()
                    #original_frames = torch.load(f'original_frames_{i}.pth').cuda()
                    batch_size = original.size(0)
                    #print(original_frames.size())

                    # store block arrays in BP via L1
                    l1_original = []
                    for n in range(samples):
                        shape_act, color_act, location_act = vae.activations(original[n].view(1,3,28,28))
                        l1_act = vae.activations_l1(original[n].view(1,3,28,28))
                        l1_original += [l1_act]
                    #l1_change, l2_act, shape_act, color_act, location_act = activations(change)
                    bp_original_l1 = []
                    bp_change_l1 = []
                    bp_junk = torch.zeros(frame_count,1).cuda()
                    
                    if t == 0:
                        for n in range(batch_size):
                            
                            BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, l1_original[n].view(frame_count,-1), bp_junk, shape_act,color_act,bp_junk,0, 0,0,1,0,frame_count,normalize_fact_novel)
                            shape_out_all, color_out_all, location_out_all, BP_layer2_out, BP_layerI_original = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, l1_original[1].view(frame_count,-1), bp_junk, shape_act,color_act,bp_junk,frame_count,normalize_fact_novel)
                            #BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, l1_change[n,:].view(1,-1), bp_junk, bp_junk,bp_junk,bp_junk,0, 0,0,1,0,1,normalize_fact_novel)
                            #shape_out_all, color_out_all, location_out_all, BP_layer2_out, BP_layerI_change = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, l1_change[1].view(1,-1), bp_junk, bp_junk,bp_junk,bp_junk,1,normalize_fact_novel)
                            if t == 0:
                                bp_original_l1 += [BP_layerI_original]
                            else:
                                bp_original_l1 += [BP_layerI_original] #[[shape_out_all, color_out_all]]
                            #bp_change_l1 += [BP_layerI_change]

                    #bp_original_l1 = torch.cat(bp_original_l1, dim=0)
                    #bp_change_l1 = torch.cat(bp_change_l1, dim=0)
                    original_BP = []
                    original_l1 = []
                    for n in range(len(original)):
                        if t == 0:
                            recon, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(bp_original_l1[n].view(frame_count,-1),BP_layer2_out,3, 'skip_cropped')
                        else:
                            recon, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(l1_original[n].view(frame_count,-1),l1_original[n].view(frame_count,-1),3, 'skip_cropped')                        

                        original_BP += [recon]
                    
                    if t == 2:
                        original = build_partial(original, len(original[0]))
                        #change = build_partial(change, len(change[0]))

                    #change_BP, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(bp_change_l1.view(batch_size,-1),BP_layer2_out,3, 'skip_cropped')

                    save_image(torch.cat([original[0].view(1,3,28,28), original_BP[0]],dim=0), f'{filepath}comp_memory{t}{i}.png', nrow = 1, normalize=False)
                    print(len(original_BP))
                    for j in range(len(original)):
                        #x, y, z = original[i].cpu().detach().view(1,-1), original_BP[i].cpu().detach().view(1,-1), change_BP[i].cpu().detach().view(1,-1)
                        x, y, z = original[j], original_BP[j].view(3,28,28), change[j].view(3,28,28)
                        r_original = compute_correlation(y,x) #cosine_similarity(x,y) # 
                        r_change = compute_correlation(y,z)#cosine_similarity(x,z) #
                        
                        r_lst0 += [r_original]
                        r_lst1 += [r_change]
                    
                    
                    del original
                    del change
                    del original_BP
                    del l1_original
                    del shape_act 
                    del color_act
                    del location_act
                    gc.collect()
                    torch.cuda.empty_cache()

            print('computing threshold')    
            avg_r0 = sum(r_lst0)/(len(r_lst0))
            avg_r1 = sum(r_lst1)/(len(r_lst1))
            out_r[t] += [[avg_r0.item(), avg_r1.item()]]
            #print(out_r[t])

            c_threshold = (avg_r0.item() + avg_r1.item())/2
            
            for l in range(len(r_lst0)):
                r_original = r_lst0[l]
                r_change = r_lst1[l]

                if r_original > c_threshold:
                    no_change_detected += [1]
                else:
                    no_change_detected += [0]

                if r_change <= c_threshold:
                    change_detected += [1]
                else:
                    change_detected += [0]

            out_dprime[t] += [compute_dprime(no_change_detected, change_detected)]
            threshold[t] += [c_threshold]
    print(out_r[0])
    #torch.save([out_r[0][i][0] for i in range(len(out_r[0]))], 'location_change_detect_r.pth')
    plt.plot(setsize_range, [out_r[0][i][0] for i in range(len(out_r[0]))], label='no change')
    plt.plot(setsize_range, [out_r[0][i][1] for i in range(len(out_r[0]))], label='change')
    plt.plot(setsize_range, threshold[0], label='threshold')
    plt.xlabel('set size')
    plt.ylabel('r')
    plt.legend()
    plt.title(f'{task} change detection, {batch_size*20} trials, BP: {bpPortion}')
    plt.savefig(f'change_detect{task}.png')
    plt.close()

    plt.plot(setsize_range, [out_r[1][i][0] for i in range(len(out_r[1]))], label='no change')
    plt.plot(setsize_range, [out_r[1][i][1] for i in range(len(out_r[1]))], label='change')
    plt.plot(setsize_range, threshold[1], label='threshold, compositional')
    plt.xlabel('set size')
    plt.ylabel('r')
    plt.legend()
    plt.title(f'{task} change detection L1, {batch_size*20} trials, BP: {bpPortion}')
    plt.savefig(f'change_detect_L1{task}.png')
    plt.close()

    plt.plot(setsize_range, out_dprime[0], label=f'dprime for BP memory')
    plt.plot(setsize_range, out_dprime[1], label=f'dprime for L1 skip')
    plt.xlabel('set size')
    plt.ylabel('dprime')
    plt.legend()
    plt.title(f'{task} change dprime vs set size, {batch_size*20} trials, BP: {bpPortion}')
    plt.savefig(f'change_detect_accuracy{task}.png')

@torch.no_grad()
def cd_jiang_olson_chun_sim(vae):
    def compute_dprime(no_change_vector, change_vector):
        no_change_vector = np.array(no_change_vector)
        change_vector = np.array(change_vector)
        
        # hit rate and false alarm rate
        hit_rate = np.mean(change_vector)
        false_alarm_rate = 1 - np.mean(no_change_vector)
        hit_rate = np.clip(hit_rate, 0.01, 0.99)
        false_alarm_rate = np.clip(false_alarm_rate, 0.01, 0.99)
        
        # z-scores
        z_hit = stats.norm.ppf(hit_rate)
        z_fa = stats.norm.ppf(false_alarm_rate)
        d_prime = z_hit - z_fa
        
        return d_prime

    def compute_correlation(x, y):
        assert x.shape == y.shape, "Tensors must have the same shape"
        
        # Flatten tensors if they're multidimensional
        x = x.view(-1)
        y = y.view(-1)
        #x = replace_near_zero(x)
        #y = replace_near_zero(y)
        
        # Compute means
        x_mean = torch.mean(x)
        y_mean = torch.mean(y)
        
        # Compute the numerator
        numerator = torch.sum((x - x_mean) * (y - y_mean))
        
        # Compute the denominator
        x_var = torch.sum((x - x_mean)**2)
        y_var = torch.sum((y - y_mean)**2)
        denominator = torch.sqrt(x_var * y_var)
        
        # Compute correlation
        correlation = numerator / denominator
        
        return correlation

    def build_partial(input_tensor, n=4):
        x, batch_size, channels, height, width = input_tensor.shape
        output_tensor = torch.zeros(x, channels, height, width, dtype=input_tensor.dtype, device=input_tensor.device)

        for batch_idx in range(x):
            for img_idx in range(n):
                output_tensor[batch_idx] += input_tensor[batch_idx, img_idx]

        # Clamp the output tensor to the range [0, 1]
        output_tensor = torch.clamp(output_tensor, min=0.0, max=1.0)

        return output_tensor

    def build_single(input_tensor):
        x, batch_size, channels, height, width = input_tensor.shape
        output_tensor = torch.zeros(x, channels, height, width, dtype=input_tensor.dtype, device=input_tensor.device)

        for batch_idx in range(x):
            output_tensor[batch_idx] += input_tensor[batch_idx, 0]

        # Clamp the output tensor to the range [0, 1]
        output_tensor = torch.clamp(output_tensor, min=0.0, max=1.0)

        return output_tensor

    def zero_outside_radius(tensor, center_x, center_y, radius=6):
        result = tensor.clone()
        center_x = (28*center_x)/500
        center_y = (28*center_y)/500
        
        # Iterate over all positions in the 28x28 grid
        for i in range(28):
            for j in range(28):
                # Calculate the squared distance from the current position to the center
                distance_squared = (i - center_y)**2 + (j - center_x)**2
                
                # If the distance is greater than the radius, set all channel values to zero
                if distance_squared > radius**2:
                    result[:, i, j] = 0
        
        return result

    vae.eval()
    max_set_size = 8 #8
    out_r = {'total':[], 'partial':[], 'single':[]}
    out_dprime = {'total':[], 'partial':[], 'single':[]}
    out_accuracy = {'total':[], 'partial':[], 'single':[]} # []
    threshold = {0:[], 1:[]} #[]
    setsize_range = range(8,9)
    threshold_data = torch.load('location_change_detect_r.pth')
    #print(threshold_data)

    for b in range(0,1): #20
        for i in setsize_range:
            torch.cuda.empty_cache()
            samples = 50
            batch_id = b*samples
            #original = torch.load(f'original_{i}.pth')[batch_id: batch_id+samples].cuda()
            #change = torch.load(f'change_{i}.pth')[batch_id: batch_id+samples].cuda()
            original_frames = torch.load(f'original_frames_{i}.pth')[batch_id: batch_id+samples].cuda()
            change_frames = torch.load(f'change_frames_{i}.pth')[batch_id: batch_id+samples].cuda()
            original = build_partial(original_frames, n=len(original_frames[0]))
            change = build_partial(change_frames, n=len(change_frames[0]))
            position_lst = torch.load(f'positions_{i}.pth')[batch_id: batch_id+samples]
            print(position_lst[0][0])
            batch_size = original.size(0)
            print(i, batch_size)
            original_partial = build_partial(original_frames)
            change_partial = build_partial(change_frames)
            original_single = build_single(original_frames)
            change_single = build_single(change_frames)
            print(original_partial.size())
            #save_image(torch.cat([original, change, original_partial,change_partial, original_single, change_single],dim=0), f'changedetect_color.png', nrow = batch_size, normalize=False)
            
            # store block arrays in BP via L1

            l1_original, l2_act, shape_act, color_act, location_act = activations(original)
            #l1_change, l2_act, shape_act, color_act, location_act = activations(change)
            bp_original_l1 = []
            bp_change_l1 = []
            bp_junk = torch.zeros(1,1).cuda()
            for n in range(batch_size):
                BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, l1_original[n,:].view(1,-1), bp_junk, bp_junk,bp_junk,bp_junk,0, 0,0,1,0,1,normalize_fact_novel)
                shape_out_all, color_out_all, location_out_all, BP_layer2_out, BP_layerI_original = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, l1_original[1].view(1,-1), bp_junk, bp_junk,bp_junk,bp_junk,1,normalize_fact_novel)
                #BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, l1_change[n,:].view(1,-1), bp_junk, bp_junk,bp_junk,bp_junk,0, 0,0,1,0,1,normalize_fact_novel)
                #shape_out_all, color_out_all, location_out_all, BP_layer2_out, BP_layerI_change = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, l1_change[1].view(1,-1), bp_junk, bp_junk,bp_junk,bp_junk,1,normalize_fact_novel)
                
                bp_original_l1 += [BP_layerI_original]
                #bp_change_l1 += [BP_layerI_change]

            bp_original_l1 = torch.cat(bp_original_l1, dim=0)
            #bp_change_l1 = torch.cat(bp_change_l1, dim=0)
            original_BP, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(bp_original_l1.view(batch_size,-1),BP_layer2_out,3, 'skip_cropped')
            #change_BP, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(bp_change_l1.view(batch_size,-1),BP_layer2_out,3, 'skip_cropped')
            #save_image(torch.cat([original, original_BP],dim=0), f'changedetect_location_bp.png', nrow = batch_size, normalize=False)


            r_lst0 = []
            r_lst1 = []
            r_lst2 = []
            r_lst3 = []
            r_lst4 = []
            r_lst5  = []
            no_change_detected = []
            change_detected = []
            no_change_detected_partial = []
            change_detected_partial = []
            no_change_detected_single = []
            change_detected_single = []
            for j in range(len(original)):
                #x, y, z = original[i].cpu().detach().view(1,-1), original_BP[i].cpu().detach().view(1,-1), change_BP[i].cpu().detach().view(1,-1)
                x, y, z = original[j], original_BP[j], change[j]
                r_original_total = compute_correlation(x,y) #cosine_similarity(x,y) # 
                r_change_total = compute_correlation(y,z)#cosine_similarity(x,z) #
                
                r_original_partial = compute_correlation(y,original_partial[j]) #cosine_similarity(x,y) # 
                r_change_partial = compute_correlation(y,change_partial[j])
                
                pos = position_lst[j][0]
                y = zero_outside_radius(y,pos[0],pos[1])
                save_image(torch.cat([x.view(1,3,28,28),original_BP[j].view(1,3,28,28),y.view(1,3,28,28), original_single[j].view(1,3,28,28)],dim=0), f'single_masked.png', nrow = 1, normalize=False)
                
                r_original_single = compute_correlation(y,original_single[j]) #cosine_similarity(x,y) # 
                r_change_single = compute_correlation(y,change_single[j])
            
                r_lst0 += [r_original_total]
                r_lst1 += [r_change_total]

                
                r_lst2 += [r_original_partial]
                r_lst3 += [r_change_partial]
                
                # compute r only in area around probe object location
                r_lst4 += [r_original_single]
                r_lst5 += [r_change_single]
            
            avg_r0 = sum(r_lst0)/(len(r_lst0))
            avg_r1 = sum(r_lst1)/(len(r_lst1))
            avg_r2 = sum(r_lst2)/(len(r_lst2))
            avg_r3 = sum(r_lst3)/(len(r_lst3))
            avg_r4 = sum(r_lst4)/(len(r_lst4))
            avg_r5 = sum(r_lst5)/(len(r_lst5))

            if len(out_r['total']) == 0:
                out_r['total'] = [avg_r0.item(), avg_r1.item()] 
                out_r['partial'] = [avg_r2.item(), avg_r3.item()]
                out_r['single'] = [avg_r4.item(), avg_r5.item()]
            else:
                out_r['total'] = [(avg_r0.item()+out_r['total'][0])/2, (avg_r1.item()+out_r['total'][1])/2] 
                out_r['partial'] = [(avg_r2.item()+out_r['partial'][0])/2, (avg_r3.item()+out_r['partial'][1])/2]
                out_r['single'] = [(avg_r4.item()+out_r['single'][0])/2, (avg_r5.item()+out_r['single'][1])/2]
            
            threshold_scalar = 0.9
            c_threshold_total = threshold_data[7] #(avg_r0.item() + avg_r1.item())/2
            c_threshold_partial = threshold_data[3] * threshold_scalar #(avg_r2.item() + avg_r3.item())/2
            c_threshold_single = threshold_data[0] #(avg_r4.item() + avg_r5.item())/2
            
            for l in range(len(r_lst0)):
                r_original = r_lst0[l]
                r_change = r_lst1[l]

                if r_original > c_threshold_total:
                    no_change_detected += [1]
                else:
                    no_change_detected += [0]

                if r_change <= c_threshold_total:
                    change_detected += [1]
                else:
                    change_detected += [0]
            
            # partial
            for l in range(len(r_lst2)):
                r_original = r_lst2[l]
                r_change = r_lst3[l]

                if r_original > c_threshold_partial:
                    no_change_detected_partial += [1]
                else:
                    no_change_detected_partial += [0]

                if r_change <= c_threshold_partial:
                    change_detected_partial += [1]
                else:
                    change_detected_partial += [0]
            
            # single
            for l in range(len(r_lst4)):
                r_original = r_lst4[l]
                r_change = r_lst5[l]

                if r_original > c_threshold_single:
                    no_change_detected_single += [1]
                else:
                    no_change_detected_single += [0]

                if r_change <= c_threshold_single:
                    change_detected_single += [1]
                else:
                    change_detected_single += [0]

            out_dprime['total'] += [compute_dprime(no_change_detected, change_detected)]
            out_accuracy['total'] += [((sum(no_change_detected)/len(no_change_detected)) + (sum(change_detected)/len(change_detected)))/2]
            #threshold[t] += [c_threshold]

            out_dprime['partial'] += [compute_dprime(no_change_detected_partial, change_detected_partial)]
            out_accuracy['partial'] += [((sum(no_change_detected_partial)/len(no_change_detected_partial)) + (sum(change_detected_partial)/len(change_detected_partial)))/2]
            #threshold[t] += [c_threshold]

            out_dprime['single'] += [compute_dprime(no_change_detected_single, change_detected_single)]
            out_accuracy['single'] += [((sum(no_change_detected_single)/len(no_change_detected_single)) + (sum(change_detected_single)/len(change_detected_single)))/2]
            #threshold[t] += [c_threshold]
            # course 
    out_dprime['total'] = sum(out_dprime['total'])/(len(out_dprime['total']))
    out_dprime['partial'] = sum(out_dprime['partial'])/(len(out_dprime['partial']))
    out_dprime['single'] = sum(out_dprime['single'])/(len(out_dprime['single']))

    out_accuracy['total'] = sum(out_accuracy['total'])/(len(out_accuracy['total']))
    out_accuracy['partial'] = sum(out_accuracy['partial'])/(len(out_accuracy['partial']))
    out_accuracy['single'] = sum(out_accuracy['single'])/(len(out_accuracy['single']))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create x-axis ticks with gaps between pairs
    x = np.arange(3)
    bar_width = 0.4
    gap_width = bar_width/2

    # Plot the bars in pairs
    ax.bar(x[0], out_r['total'][0], width=bar_width, label = 'full, no change')
    ax.bar(x[0] + gap_width, out_r['total'][1], width=bar_width, label = 'full, change')
    ax.bar(x[1], out_r['partial'][0], width=bar_width, label = 'partial, no change')
    ax.bar(x[1] + gap_width, out_r['partial'][1], width=bar_width, label = 'partial, change')
    ax.bar(x[2], out_r['single'][0], width=bar_width, label = 'single, no change')
    ax.bar(x[2] + gap_width, out_r['single'][1], width=bar_width, label = 'single, change')
    ax.set_ylabel('r')
    ax.legend()
    ax.set_title('Correlation for location change detection, 50 trials')
    plt.savefig('change_detect_bar.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create x-axis ticks with gaps between pairs
    x = np.arange(3)
    bar_width = 0.4
    gap_width = 0.5

    # Plot the bars in pairs
    ax.bar(x[0], out_dprime['total'], width=bar_width, label='full context')
    ax.bar(x[1], out_dprime['partial'], width=bar_width, label='partial context')
    ax.bar(x[2], out_dprime['single'], width=bar_width, label='single context')
    ax.legend()
    ax.set_title('dprime for location change detection, 50 trials')
    ax.set_ylabel('dprime')
    plt.savefig('change_detect_accuracy_bar.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create x-axis ticks with gaps between pairs
    x = np.arange(3)
    bar_width = 0.4
    gap_width = 0.5

    # Plot the bars in pairs
    ax.bar(x[0], out_dprime['total'], width=bar_width, label='full context')
    ax.bar(x[1], out_dprime['partial'], width=bar_width, label='partial context')
    ax.bar(x[2], out_dprime['single'], width=bar_width, label='single context')
    ax.legend()
    ax.set_title('accuracy for location change detection, 50 trials')
    ax.set_ylabel('%')
    plt.savefig('change_detect_accuracy_bar_nodprime.png')

@torch.no_grad()
def cd_lines(vae):
    def compute_dprime(no_change_vector, change_vector):
        no_change_vector = np.array(no_change_vector)
        change_vector = np.array(change_vector)
        
        # hit rate and false alarm rate
        hit_rate = np.mean(change_vector)
        false_alarm_rate = 1 - np.mean(no_change_vector)
        hit_rate = np.clip(hit_rate, 0.01, 0.99)
        false_alarm_rate = np.clip(false_alarm_rate, 0.01, 0.99)
        
        # z-scores
        z_hit = stats.norm.ppf(hit_rate)
        z_fa = stats.norm.ppf(false_alarm_rate)
        d_prime = z_hit - z_fa
        
        return d_prime

    def compute_correlation(x, y):
        assert x.shape == y.shape, "Tensors must have the same shape"
        
        # Flatten tensors if they're multidimensional
        x = x.view(-1)
        y = y.view(-1)
        #x = replace_near_zero(x)
        #y = replace_near_zero(y)
        
        # Compute means
        x_mean = torch.mean(x)
        y_mean = torch.mean(y)
        
        # Compute the numerator
        numerator = torch.sum((x - x_mean) * (y - y_mean))
        
        # Compute the denominator
        x_var = torch.sum((x - x_mean)**2)
        y_var = torch.sum((y - y_mean)**2)
        denominator = torch.sqrt(x_var * y_var)
        
        # Compute correlation
        correlation = numerator / denominator
        
        return correlation
    
    def build_partial(input_tensor, n, out_x=1):
        x, b, channels, height, width = input_tensor.shape
        output_tensor = torch.zeros(x, channels, height, width, dtype=input_tensor.dtype, device=input_tensor.device)

        for batch_idx in range(x):
            for img_idx in range(n):
                #print(batch_idx, img_idx)
                output_tensor[batch_idx] += input_tensor[batch_idx, img_idx]

        # Clamp the output tensor to the range [0, 1]
        output_tensor = torch.clamp(output_tensor, min=0.0, max=1.0)

        return output_tensor

    vae.eval()
    out_r = defaultdict(list)
    out_dprime = defaultdict(list)
    threshold = defaultdict(list)
    out_acc = defaultdict(list)
    for s in range(0,2):
        for t in range(0,3):
            if s == 0:
                iden = 1
            else:
                iden = ''
            with torch.no_grad():
                if t == 0: # same color/ diff orientation
                    frames = torch.load(f'simulation_src/frames_8c{iden}.pth') #.view(-1,3,28,28)
                    frame_count = 8
                    name = '8c'
                elif t == 1: # diff color/ same orientation
                    frames = torch.load(f'simulation_src/frames_8o{iden}.pth') #.view(-1,3,28,28)
                    frame_count = 8
                    name = '8o'
                else: # diff color/ diff orientation
                    frames = torch.load(f'simulation_src/frames_4c4o{iden}.pth') #.view(-1,3,28,28)
                    frame_count = 4
                    name = '4c4o'

                if s == 0:
                    frame_count = frame_count//2
                frame_c =len(frames['original_frames'][0])
                #frame_count = 1
                print(frame_count)
                #print(frames.size())
                
                r_lst0 = []
                r_lst1 = []
                for b in range(0,2): #20
                    #print(i,b)
                    torch.cuda.empty_cache()
                    samples = 5
                    batch_id = b*samples
                    original_frames = frames['original_frames'][batch_id: batch_id+samples].cuda().view(-1,frame_count,3,28,28)
                    change_frames = frames['change_frames'][batch_id: batch_id+samples].cuda().view(-1,frame_count,3,28,28)
                    original = build_partial(original_frames, frame_c)
                    change = build_partial(change_frames, frame_c)
                    batch_size = samples

                    # store lines arrays in BP via L1 hollistic
                    #l1_original_frames = []
                    l1_original = []
                    '''for n in range(samples):
                        shape_act, color_act, location_act = vae.activations(original_frames[n].view(frame_count,3,28,28))
                        l1_act = vae.activations_l1(original_frames[n].view(frame_count,3,28,28))
                        l1_original_frames += [l1_act]'''

                    # store line arrays in BP compositional
                    for n in range(samples):
                        shape_act, color_act, location_act = vae.activations(original[n].view(1,3,28,28))
                        l1_act = vae.activations_l1(original[n].view(1,3,28,28))
                        l1_original += [l1_act]
                        
                    print(len(l1_original))
                    # hollistic
                    bp_original_l1 = []
                    bp_junk = torch.zeros(frame_count,1).cuda()
                    for n in range(len(l1_original)):
                        #print(l1_original[n].size())
                        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, l1_original[n].view(1,-1), bp_junk, shape_act,color_act,bp_junk,0, 0,0,1,0,1,normalize_fact_novel)
                        shape_out_all, color_out_all, location_out_all, BP_layer2_out, BP_layerI_original = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, l1_original[0].view(1,-1), bp_junk, shape_act,color_act,bp_junk,1,normalize_fact_novel)
                        
                        bp_original_l1 += [BP_layerI_original]

                    # compositional
                    '''bp_original_frames_l1 = []
                    bp_junk = torch.zeros(frame_count,1).cuda()
                    for n in range(batch_size):
                        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, l1_original_frames[n].view(frame_count,-1), bp_junk, shape_act,color_act,bp_junk,0, 0,0,1,0,frame_count,normalize_fact_novel)
                        shape_out_all, color_out_all, location_out_all, BP_layer2_out, BP_layerI_original = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, l1_original_frames[0].view(frame_count,-1), bp_junk, shape_act,color_act,bp_junk,frame_count,normalize_fact_novel)

                        bp_original_frames_l1 += [BP_layerI_original]'''
                    
                    #decode
                    original_holistic_BP = []
                    #original_comp_BP = []
                    for n in range(batch_size):
                        recon, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(bp_original_l1[n].view(1,-1),BP_layer2_out,3, 'skip_cropped')
                        original_holistic_BP += [recon]

                        #recon = recon, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(bp_original_frames_l1[n].view(frame_count,-1),BP_layer2_out,3, 'skip_cropped') #vae.decoder_cropped(bp_original_l1[n][0].view(frame_count,-1),bp_original_l1[n][1].view(frame_count,-1),0)
                        #recon = build_partial(recon.view(1,frame_count,3,28,28), len(recon))
                        #original_comp_BP += [recon]
                    
                        #change_BP, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(bp_change_l1.view(batch_size,-1),BP_layer2_out,3, 'skip_cropped')

                    #save_image(torch.cat([original[0:4].view(4,3,28,28).cpu(), torch.cat(original_holistic_BP[0:4],0).cpu()],dim=0), f'comp_memory{name}.png', nrow = 4, normalize=False) #, torch.cat(original_comp_BP[0:3],0).cpu()
                    for j in range(batch_size):
                        #x, y, z = original[i].cpu().detach().view(1,-1), original_BP[i].cpu().detach().view(1,-1), change_BP[i].cpu().detach().view(1,-1)
                        x, y, z = original[j].cpu().detach().view(1,-1), original_holistic_BP[j].view(3,28,28).cpu().detach().view(1,-1), change[j].view(3,28,28).cpu().detach().view(1,-1)
                        r_holistic = compute_correlation(y,x) #cosine_similarity(x,y) # 
                        r_change = compute_correlation(y,z)#cosine_similarity(x,z) #
                        
                        r_lst0 += [r_holistic]
                        r_lst1 += [r_change]
                    
                    
                    del original
                    del l1_original 
                    del shape_act 
                    del color_act
                    del location_act
                    torch.cuda.empty_cache()

                print('computing threshold')
                correct_predictions = 0
                total_predictions = len(r_lst0) + len(r_lst1)

                avg_r0 = sum(r_lst0)/(len(r_lst0))
                avg_r1 = sum(r_lst1)/(len(r_lst1))
                out_r[t] += [avg_r0.item(), avg_r1.item()]
                c_threshold = (avg_r0.item() + avg_r1.item())/2
                no_change_detected = []
                change_detected = []
                
                for l in range(len(r_lst0)):
                    r_original = r_lst0[l]
                    r_change = r_lst1[l]

                    if r_original > c_threshold:
                        no_change_detected += [1]
                        correct_predictions += 1
                    else:
                        no_change_detected += [0]

                    if r_change <= c_threshold:
                        change_detected += [1]
                        correct_predictions += 1
                    else:
                        change_detected += [0]

                out_dprime[t] += [compute_dprime(no_change_detected, change_detected)]
                threshold[t] += [c_threshold]
                out_acc[t] += [correct_predictions/total_predictions]

    print(out_r)
    print(out_dprime)
    #torch.save([out_r[0][i][0] for i in range(len(out_r[0]))], 'location_change_detect_r.pth')
    setsize_range = [4,8]
    plt.plot(setsize_range, out_dprime[2], label='conjunction')
    plt.plot(setsize_range, out_dprime[0], label='c')
    plt.plot(setsize_range, out_dprime[1], label='o')
    plt.xlabel('num features')
    plt.ylabel('dprime')
    plt.legend()
    plt.title(f'lines change detection, {500} trials')
    plt.savefig(f'change_detect_lines.png')
    plt.close()

    plt.plot(setsize_range, out_acc[2], label='conjunction', linewidth=3.5)
    plt.plot(setsize_range, out_acc[0], label='c', linewidth=3.5)
    plt.plot(setsize_range, out_acc[1], label='o', linewidth=3.5)
    plt.xlabel('num features')
    plt.ylabel('accuracy %')
    plt.legend()
    plt.title(f'lines change detection, {500} trials')
    plt.tick_params(axis='both', labelsize=14)
    plt.savefig(f'change_detect_lines_acc.png')
    plt.close()

    '''fig, ax = plt.subplots(figsize=(8, 6))

    # Create x-axis ticks with gaps between pairs
    x = np.arange(3)
    bar_width = 0.4
    gap_width = bar_width/2

    # Plot the bars in pairs
    ax.bar(x[0], out_dprime[0][0], width=bar_width, label = 'holistic 8c')
    #ax.bar(x[0] + gap_width, out_r[0][1], width=bar_width, label = 'compositonal 8c')
    ax.bar(x[1], out_dprime[1][0], width=bar_width, label = 'holistic 8o')
    #ax.bar(x[1] + gap_width, out_r[1][1], width=bar_width, label = 'compositional 8o')
    ax.bar(x[2], out_dprime[2][0], width=bar_width, label = 'holistic 4c4o')
    #ax.bar(x[2] + gap_width, out_r[2][1], width=bar_width, label = 'compositonal 4c4o')
    ax.set_ylabel('dprime')
    ax.legend()
    ax.set_title('dprime by display type, 500 trials')
    plt.savefig('change_detect_bar.png')
    plt.close()'''

######################## Figure 2a #######################################################################################
#store items using both features, and separately color and shape (memory retrievals)

@torch.no_grad()
def fig_2a(vae, folder_path):
    print('generating figure 2a, reconstructions from the binding pool')

    numimg= 6
    bs=numimg #number of images to display in this figure
    nw=2
    bs_testing = numimg # 20000 is the limit
    train_loader_noSkip = Dataset('mnist',{'colorize':True}).get_loader(bs)
    test_loader_noSkip = Dataset('mnist',{'colorize':True},train=False).get_loader(bs)

    test_loader_smaller = test_loader_noSkip
    images, shapelabels = next(iter(test_loader_smaller))#peel off a large number of images
    #orig_imgs = images.view(-1, 3 * 28 * 28).cuda()
    imgs = images.clone().cuda()

    #run them all through the encoder
    l1_act, l2_act, shape_act, color_act, location_act = vae.activations(imgs)  #get activations from this small set of images
    BPOut_all, Tokenbindings_all = BPTokens_storage(bpsize, bpPortion, l1_act, l2_act, shape_act,color_act,location_act,shape_coeff, color_coeff, location_coeff, l1_coeff,l2_coeff,1,normalize_fact_novel)
        
    shape_out_all, color_out_all, location_out_all, BP_layer2_out, BP_layerI_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut_all, Tokenbindings_all,l1_act, l2_act, shape_act,color_act,location_act,1,normalize_fact_novel)

    
    #memory retrievals from Bottleneck storage
    bothRet = vae.decoder_cropped(shape_out_all, color_out_all,0, 0).cuda()  # memory retrieval from the bottleneck
    #shapeRet = vae.decoder_shape(shape_out_BP_shapeonly, color_out_BP_shapeonly , 0).cuda()  #memory retrieval from the shape map
    #colorRet = vae.decoder_color(shape_out_BP_coloronly, color_out_BP_coloronly, 0).cuda()  #memory retrieval from the color map
    shapeRet = bothRet
    colorRet = bothRet
    save_image(
        torch.cat([imgs[0: numimg].view(numimg, 3, 28, 28), bothRet[0: numimg].view(numimg, 3, 28, 28),
                   shapeRet[0: numimg].view(numimg, 3, 28, 28), colorRet[0: numimg].view(numimg, 3, 28, 28)], 0),
        'output{num}/figure2a_BP_bottleneck_.png'.format(num=modelNumber),
        nrow=numimg,
        normalize=False,
        range=(-1, 1),
    )

    #memory retrievals when information was stored from L1 and L2
    BP_layer1_noskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out,BP_layer2_out, 1, 'noskip') #bp retrievals from layer 1
    BP_layer2_noskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out,BP_layer2_out, 2, 'noskip') #bp retrievals from layer 2

    save_image(
        torch.cat([
                   BP_layer2_noskip[0: numimg].view(numimg, 3, 28, 28), BP_layer1_noskip[0: numimg].view(numimg, 3, 28, 28)], 0),
        'output{num}/figure2a_layer2_layer1.png'.format(num=modelNumber),
        nrow=numimg,
        normalize=False,
        range=(-1, 1),
    )


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


    
  



@torch.no_grad()
def fig_2bt(vae, folder_path):
    all_imgs = []
    recon = list()
    print('generating Figure 2bt, Novel characters retrieved from memory of L1 and Bottleneck using Tokens')
    retina_size = 100
    imgsize = 28
    numimg = 6  #how many objects will we use here?
    vae.eval()

    #load in some examples of Bengali Characters
    for i in range (1,numimg+1):
        img_new = convert_tensor(Image.open(f'data/current_bengali/{i}_thick.png'))[0:3,:,:]
        all_imgs.append(img_new)

    #all_imgs is a list of length 3, each of which is a 3x28x28
    all_imgs = torch.stack(all_imgs)
    imgs = all_imgs.view(-1, 3 * imgsize * imgsize).cuda()   #dimensions are R+G+B + # pixels
    imgs[0], imgs[5] = imgs[5], imgs[0]
    imgmatrix = imgs.view(numimg,3,28,28)
    #push the images through the model
    activations = vae.activations(imgs.view(-1,3,28,28), False)
    l1_act = activations['skip']
    shape_act = activations['shape']
    color_act = activations['color']

    emptyshape = torch.empty((1,3,28,28)).cuda()
    # store 1 -> numimg items
    for i in range(len(l1_act)):
        z = len(l1_act[i])//2
        l1_act[i,:z]  *= -1
    
    BP_activations_l1 = {'l1': [l1_act.view(numimg,-1), 1]}

    for n in range(1,numimg+1):
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_l1, n, normalize_fact_novel)
        BP_act_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_l1, n,normalize_fact_novel)
        l1_out_all = BP_act_out['l1']       
        plt.close()

        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot the first distribution (post BP)
        ax2.hist(l1_out_all[0].cpu().detach())
        ax2.set_title('BP')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')
        for i in range(len(l1_out_all)):
            z = len(l1_out_all[i])//2
            l1_out_all[i,:z]  *= -1
        #l1_out_all = vae.sparse_relu(l1_out_all*(1/2)) 
        # Plot the second distribution (pre BP)
        ax1.hist(l1_act[0].cpu().detach()) 
        ax1.set_title('pre BP')
        #ax1.set_xlabel('Value')
        #ax1.set_ylabel('Frequency')

        # Adjust the layout and save the figure
        plt.tight_layout()
        plt.savefig('skipconhist.png')
        plt.close()
        recon_layer1_skip = vae.decoder_skip_cropped(0, 0, 0, l1_out_all.view(n,-1))
        #recon_layer1_skip= vae.decoder_skip_cropped(0,0,0,l1_out_all)      
        imgmatrix= torch.cat([imgmatrix,recon_layer1_skip],0)

        #now pad with empty images
        for i in range(n,numimg):
            imgmatrix= torch.cat([imgmatrix,emptyshape*0],0)

    save_image(imgmatrix, f'{folder_path}figure2bt.png',    nrow=numimg, normalize=False,   )


@torch.no_grad()
def fig_2c(vae, folder_path):
    vae.eval()
    print('generating Figure 2c, Familiar characters retrieved from Bottleneck using Tokens')
    retina_size = 100
    reconMap = list()
    reconL1 = list()
    imgsize = 28
    numimg = 7  #how many objects will we use here?
    #torch.set_default_dtype(torch.float64)
    #make the data loader, but specifically we are creating stimuli on the opposite to how the model was trained
    test_loader_noSkip= Dataset('mnist',{'colorize':True}, train=True).get_loader(numimg)
    
    #Code showing the data loader for how the model was trained, empty dict in 3rd param is for any color:
    '''train_loader_noSkip, train_loader_skip, test_loader_noSkip, test_loader_skip = dataset_builder('mnist',bs,
            {},True,{'right':list(range(0,5)),'left':list(range(5,10))}) '''    

    dataiter_noSkip = iter(test_loader_noSkip)
    data = next(dataiter_noSkip)
    data = data[0] #.cuda()
    
    sample_data = data
    sample_size = numimg
    sample_data = sample_data[:sample_size]
    sample = sample_data.cuda()
    
    
    #push the images through the model
    activations = vae.activations(sample.view(-1,3,28,28), False)
    l1_act = activations['skip']
    shape_act = activations['shape']
    color_act = activations['color']
    reconb = vae.decoder_cropped(shape_act, color_act, 0)
    #reconb, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location,mu_scale,log_var_scale = vae(sample, 'cropped')
    #l1_act, l2_act, shape_act, color_act, location_act = vae.activations(sample[1].view(-1,3,28,28).cuda())
    
    #shape_act = vae.sampling(mu_shape, log_var_shape).cuda()
    #color_act = vae.sampling(mu_color, log_var_color).cuda()
    reconskip = vae.decoder_skip_cropped(0, 0, 0, l1_act.view(numimg,-1))
    #reconskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(l1_act.view(numimg,-1), l2_act, 3, 'skip_cropped') 

    emptyshape = torch.empty((1,3,28,28)).cuda()
    imgmatrixMap = torch.cat([sample.view(numimg,3,28,28).cuda(), reconb],0)
    imgmatrixL1 = torch.cat([sample.view(numimg,3,28,28).cuda(), reconskip],0)
    shape_act_in = shape_act

    BP_activations_sc = {'shape': [shape_act.view(numimg,-1), 1], 'color': [color_act.view(numimg,-1), 1]}
    BP_activations_l1 = {'l1': [l1_act.view(numimg,-1), 1]}
    # store 1 -> numimg items
    for n in range(1,numimg+1):
        #Store and retrieve the map versions
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_sc, n,normalize_fact_novel)
        BP_activations_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_sc, n,normalize_fact_novel)
        shape_out_all, color_out_all = BP_activations_out['shape'], BP_activations_out['color']
        z = torch.randn(numimg-n,8).cuda()
        #shape_out_all = torch.cat([shape_out_all,z],0)
        #color_out_all = torch.cat([color_out_all,z],0)
        #shape_out_all = shape_act[:n]
        #color_out_all = color_act[:n] #torch.cat([color_out_all,z],0)
        #shape_out_all = torch.cat([shape_out_all,z],0)
        #color_out_all = torch.cat([color_out_all,z],0)
        #retrievals = []
        #for i in range(n):
         #   print(color_out_all[i].view(-1,vae.z_dim).size())
         #   retrievals += [vae.decoder_cropped(shape_out_all[i].view(-1,vae.z_dim), color_out_all[i].view(-1,vae.z_dim),0,0).cuda()]
        retrievals = vae.decoder_cropped(shape_out_all, color_out_all,0,0).cuda()
        #retrievals = retrievals[:n]
        #Store and retrieve the L1 version
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_l1, n,normalize_fact_novel)
        BP_activations_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_l1, n,normalize_fact_novel)
        #l1_out_all=l1_act[:n] #remove
        l1_out_all = BP_activations_out['l1']
        recon_layer1_skip = vae.decoder_skip_cropped(0, 0, 0, l1_out_all.view(n,-1))

        #imgmatrixMap= torch.cat([imgmatrixMap] + retrievals,0)
        
        imgmatrixMap= torch.cat([imgmatrixMap, retrievals],0)
        imgmatrixL1= torch.cat([imgmatrixL1,recon_layer1_skip],0)

        #now pad with empty images
        for i in range(n,numimg):
            imgmatrixMap= torch.cat([imgmatrixMap,emptyshape*0],0)
            imgmatrixL1= torch.cat([imgmatrixL1,emptyshape*0],0)
 
    save_image(imgmatrixL1, f'{folder_path}figure2cL1.png',  nrow=numimg,        normalize=False) #range=(-1, 1))
    save_image(imgmatrixMap, f'{folder_path}figure2cMap.png',  nrow=numimg,        normalize=False) #,range=(-1, 1))