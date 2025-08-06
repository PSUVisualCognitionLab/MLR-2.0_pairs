from simulation_src.plots_novel import cd_jiang_olson_chun_sim, cd_r_acc_vs_setsize, fig_loc_compare, load_checkpoint, cd_lines, cd_r_bp_cnn, obj_back_swap
from simulation_src.memory_plots import fig_2b, fig_2bt, fig_2c, fig_repeat_recon, fig_efficient_rep
from simulation_src.vision_plots import fig_2n, fig_retinal_mod, fig_visual_synthesis
from MLR_src.label_network import vae_shape_labels, vae_color_labels, load_checkpoint_shapelabels, s_classes
import torch
import os
import matplotlib.pyplot as plt
import argparse
from joblib import load

parser = argparse.ArgumentParser(description="Simulations using MLR-2.0")
parser.add_argument("--c_folder", type=str, default='test', help="where to find the vae checkpoint/")
parser.add_argument("--run_name", type=str, default='test', help="where to store simulation outputs/")
args = parser.parse_args()

folder_name = args.c_folder #'stn_mnist'
run_name = args.run_name

checkpoint_folder_path = f'checkpoints/{folder_name}/' # the output folder for the trained model versions
d = 1
vae = load_checkpoint(f'{checkpoint_folder_path}/mVAE_checkpoint.pth', d, True)
vae.eval()
load_checkpoint_shapelabels(f'{checkpoint_folder_path}/label_network_checkpoint.pth', d)
clf_shapeS = load(f'{checkpoint_folder_path}/ss.joblib')
device = torch.device(f'cuda:{d}')
torch.cuda.set_device(d)
print('checkpoint loaded')

simulation_folder_path = f'simulations/{run_name}/'
if not os.path.exists('simulations/'):
    os.mkdir('simulations/')
    
if not os.path.exists(simulation_folder_path):
    os.mkdir(simulation_folder_path)


#Following do work
fig_retinal_mod(vae, simulation_folder_path)
fig_efficient_rep(vae, simulation_folder_path)
fig_visual_synthesis(vae, vae_shape_labels, s_classes, clf_shapeS, simulation_folder_path)
fig_repeat_recon(vae, simulation_folder_path)
fig_2n(vae, simulation_folder_path)    #  Bengali reconstructions
fig_2bt(vae, simulation_folder_path)   #  set size retrieval of digits using L1
fig_2c(vae, simulation_folder_path)    #  set size retrieval of digits using shape and color maps


# following do not work
#fig_2b(vae, simulation_folder_path)   
#x= cd_r_bp_cnn(vae, '_color', simulation_folder_path)
#x = fig_loc_compare(vae, simulation_folder_path)
#x= obj_back_swap(vae, device, simulation_folder_path)
