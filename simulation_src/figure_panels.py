import simulation_src.figures as figures
import torch

# panels:

@torch.no_grad()
def interference(vae, folder_path):
    folder_path = folder_path + "interference/"
    figures.fig_efficient_rep(vae, folder_path)
    pass

@torch.no_grad()
def individuated(vae, folder_path):
    folder_path = folder_path + "individuated/"
    figures.fig_repeat_recon(vae, folder_path)
    pass

@torch.no_grad()
def novel(vae, folder_path):
    folder_path = folder_path + "novel/"
    figures.fig_novel_representations(vae, folder_path)
    pass

@torch.no_grad()
def addressability(vae, folder_path):
    folder_path = folder_path + "addressability/"
    # TODO: implement fig_addressable()
    #figures.fig_novel_representations(vae, folder_path)
    pass

@torch.no_grad()
def generative(vae, folder_path):
    folder_path = folder_path + "generative/"
    # TODO: implement fig_generative_noise()
    #figures.fig_repeat_recon(vae, folder_path)
    pass

@torch.no_grad()
def synthesis(vae, folder_path):
    folder_path = folder_path + "synthesis/"
    figures.fig_visual_synthesis(vae, folder_path)
    pass

@torch.no_grad()
def compositional(vae, folder_path):
    folder_path = folder_path + "compositional/"
    figures.fig_retinal_mod(vae, folder_path)
    pass