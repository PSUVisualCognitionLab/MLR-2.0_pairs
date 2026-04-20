# Visualize shape and object latent spaces using t-SNE and UMAP
# Usage: python maplatents.py --folder test --components shape color

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

from sklearn.manifold import TSNE
try:
    import umap
    HAS_UMAP = True
except ImportError:
    print("UMAP not installed or configured correctly")
    HAS_UMAP = False

from MLR_src.mVAE import load_checkpoint, load_dimensions
from MLR_src.dataset_builder import Dataset
from training_constants import training_components, training_datasets
from itertools import cycle

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default='test', help="Where to find checkpoints in checkpoints/")
parser.add_argument("--checkpoint_name", type=str, default='mVAE_checkpoint.pth', help="file name of checkpoint .pth")
parser.add_argument("--components", nargs='+', type=str, default=['shape', 'color'], help="Which latent spaces to visualize")
parser.add_argument("--n_samples", type=int, default=5000, help="Number of samples to collect")
parser.add_argument("--cuda_device", type=int, default=0)
parser.add_argument("--use_mu", action='store_true', default=True, help="Use mu (True) or sampled z (False)")
args = parser.parse_args()

checkpoint_path = f'checkpoints/{args.folder}/{args.checkpoint_name}'
output_dir = f'latent_visualizations/{args.folder}/'

# setup
if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.cuda_device}')
    torch.cuda.set_device(args.cuda_device)
else:
    device = 'cpu'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# load model
vae = load_checkpoint(checkpoint_path, args.cuda_device)
vae.eval()

# build dataloaders for each component
bs = 100
dataloaders = {}
for component in args.components:
    if component not in training_components:
        print(f"Skipping unknown component: {component}")
        continue
    for dataset_name in training_components[component][0]:
        if dataset_name not in dataloaders:
            base_name = dataset_name.split('-')[0]
            transforms = training_datasets[dataset_name]
            loader = cycle(Dataset(base_name, transforms).get_loader(bs))
            dataloaders[dataset_name] = iter(loader)

# collect latent activations
def collect_latents(vae, dataloaders, component, n_samples, use_mu=True):
    """Collect latent vectors and labels for a given component"""
    dataset_name = training_components[component][0][0]
    dataloader = dataloaders[dataset_name]
    
    all_latents = []
    all_shape_labels = []
    all_color_labels = []
    collected = 0

    while collected < n_samples:
        data, labels = next(dataloader)
        #print(f"label range: {labels[0].min()} to {labels[0].max()}")
        if type(data) == list:
            image = data[1].to(device)
        else:
            image = data.to(device)

        with torch.no_grad():
            mu_shape, log_var_shape, mu_color, log_var_color, hskip = vae.encoder(image)

            if component in ['shape', 'object']:
                mu = mu_shape
                log_var = log_var_shape
            elif component == 'color':
                mu = mu_color
                log_var = log_var_color
            else:
                print(f"Unknown component: {component}")
                return None, None, None

            if use_mu:
                z = mu
            else:
                z = vae.sampling(mu, log_var)

        all_latents.append(z.cpu().numpy())
        all_shape_labels.append(labels[0].numpy())
        all_color_labels.append(labels[1].numpy())
        collected += len(image)

    latents = np.concatenate(all_latents, axis=0)[:n_samples]
    shape_labels = np.concatenate(all_shape_labels, axis=0)[:n_samples]
    color_labels = np.concatenate(all_color_labels, axis=0)[:n_samples]

    return latents, shape_labels, color_labels


# label name maps for readability
#emnist_label_names = {i: chr(65 + i) for i in range(26)}  # 0-25 -> A-Z
emnist_label_names = {i: chr(55 + i) for i in range(10, 36)}  # 10=A, 11=B, ... 35=Z
color_label_names = {0: 'red', 1: 'green', 2: 'blue', 3: 'purple', 4: 'yellow',
                     5: 'cyan', 6: 'orange', 7: 'brown', 8: 'pink', 9: 'white'}


def plot_embedding_named(embedding, labels, title, save_path, label_names=None):
    """Plot with named labels"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    cmap = plt.cm.get_cmap('tab20', n_labels) if n_labels <= 20 else plt.cm.get_cmap('nipy_spectral', n_labels)

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = label_names[label] if label_names and label in label_names else str(label)
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cmap(i)], s=8, alpha=0.5, label=name)

    ax.set_title(title, fontsize=16)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=3, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# main loop
for component in args.components:
    if component not in training_components:
        continue

    print(f"\nCollecting {args.n_samples} samples for {component} latent space...")
    latents, shape_labels, color_labels = collect_latents(vae, dataloaders, component, args.n_samples, args.use_mu)

    if latents is None:
        continue

    print(f"  Latent shape: {latents.shape}")
    print(f"  Unique shape labels: {np.unique(shape_labels)}")
    print(f"  Unique color labels: {np.unique(color_labels)}")

    # determine label names
    if component in ['shape']:
        label_names = emnist_label_names
    else:
        label_names = None

    # t-SNE
    print(f"  Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    tsne_embedding = tsne.fit_transform(latents)

    plot_embedding_named(tsne_embedding, shape_labels,
                         f't-SNE of {component} latent (colored by identity)',
                         os.path.join(output_dir, f'tsne_{component}_by_identity.png'),
                         label_names)

    plot_embedding_named(tsne_embedding, color_labels,
                         f't-SNE of {component} latent (colored by color)',
                         os.path.join(output_dir, f'tsne_{component}_by_color.png'),
                         color_label_names)

    # UMAP
    if HAS_UMAP:
        print(f"  Running UMAP...")
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        umap_embedding = reducer.fit_transform(latents)

        plot_embedding_named(umap_embedding, shape_labels,
                             f'UMAP of {component} latent (colored by shape/identity)',
                             os.path.join(output_dir, f'umap_{component}_by_shape.png'),
                             label_names)

        plot_embedding_named(umap_embedding, color_labels,
                             f'UMAP of {component} latent (colored by color)',
                             os.path.join(output_dir, f'umap_{component}_by_color.png'),
                             color_label_names)

print("\nDone! Check", output_dir)
