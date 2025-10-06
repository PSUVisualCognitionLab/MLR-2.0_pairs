from simulation_src.figure_panels import individuated, interference, novel, addressability, generative, synthesis, compositional
from MLR_src.mVAE import load_checkpoint
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
clf_shapeS = load(f'{checkpoint_folder_path}/ess.joblib')
device = torch.device(f'cuda:{d}')
torch.cuda.set_device(d)
print('checkpoint loaded')

simulation_folder_path = f'simulations/{run_name}/'
if not os.path.exists('simulations/'):
    os.mkdir('simulations/')
    
if not os.path.exists(simulation_folder_path):
    os.mkdir(simulation_folder_path)

#individuated(vae, simulation_folder_path)
#interference(vae, simulation_folder_path)
#novel(vae, simulation_folder_path)
#addressability(vae, simulation_folder_path)
#generative(vae, simulation_folder_path)
synthesis(vae, vae_shape_labels, s_classes, clf_shapeS, simulation_folder_path)
#compositional(vae, simulation_folder_path)