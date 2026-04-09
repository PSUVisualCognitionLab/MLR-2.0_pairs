import simulation_src.figures as figures
import torch
import os
# panels:

@torch.no_grad()
def interference(vae, folder_path):
    folder_path = folder_path + "interference/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    figures.fig_efficient_rep(vae, folder_path)
    pass

@torch.no_grad()
def individuated(vae, folder_path):
    folder_path = folder_path + "individuated/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    figures.fig_repeat_recon(vae, folder_path)
    pass

@torch.no_grad()
def novel(vae, folder_path):
    folder_path = folder_path + "novel/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    figures.fig_novel_representations(vae, folder_path)
    pass

@torch.no_grad()
def addressability(vae, folder_path):
    folder_path = folder_path + "addressability/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    # TODO: implement fig_addressable()
    #figures.fig_novel_representations(vae, folder_path)
    pass

@torch.no_grad()
def generative(vae, shape_label, s_classes, color_label, c_classes, folder_path):
    folder_path = folder_path + "generative/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    # TODO: implement fig_generative_noise()
    figures.fig_generative_noise(vae, shape_label, s_classes, color_label, c_classes, folder_path)

@torch.no_grad()
def synthesis(vae, shape_label, s_classes, shape_classifier, folder_path):
    folder_path = folder_path + "synthesis/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    figures.fig_visual_synthesis(vae, shape_label, s_classes, shape_classifier, folder_path)
    pass

@torch.no_grad()
def compositional(vae, folder_path):
    folder_path = folder_path + "compositional/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    figures.fig_retinal_mod(vae, folder_path)
    pass

@torch.no_grad()
def flexibility(vae, shape_label, s_classes, folder_path):
    folder_path = folder_path + "flexibility/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    #figures.fig_retinal_mod(vae, folder_path)
    pass

@torch.no_grad()
def holistic(vae, folder_path):
    folder_path = folder_path + "holistic/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    #figures.fig_retinal_mod(vae, folder_path)
    pass