"""
Weak-segmentation on CIFAR-10 with an extra ‘background’ logit.

• backbone   : ResNet-50 (lightly modified to keep more spatial detail)
• supervision: ONLY image-level labels (no pixel masks)
• loss       : BCE-with-logits on a smooth-max pooled vector
               (11 detectors: 10 classes + 1 background ↦ target 0)

After every epoch a PNG called  epoch_##.png  is written in the
working directory, showing the first 10 test images and their
predicted foreground masks (white = object, black = background).
"""

import math, os, itertools, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet152
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from dataset_builder import Dataset
from packaging import version
import torchvision

def make_resnet50(pretrained=True):
    """
    Return a torchvision.models.resnet50, handling the API change:
        • torchvision < 0.13  →  resnet50(pretrained=bool)
        • torchvision ≥ 0.13  →  resnet50(weights="DEFAULT"|None)
    """
    if version.parse(torchvision.__version__) < version.parse("0.13"):
        return torchvision.models.resnet101(pretrained=pretrained)
    else:
        # For 0.13+: 'DEFAULT' gives the ImageNet-1k weights
        return torchvision.models.resnet101(weights="DEFAULT" if pretrained else None)



# ----------------------------------------------------------------------
# 1.  Model
# ----------------------------------------------------------------------
class WeakResNet50Seg(nn.Module):
    """ResNet-50 turned into a tiny fully-conv ‘segmenter’."""
    def __init__(self, n_classes: int = 10, pretrained=True):
        super().__init__()
        self.n_fg = n_classes
        self.n_total = n_classes + 1                      # +1 for background

        backbone = make_resnet50(pretrained)

        # (a) keep input resolution
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()                  # remove early 2× ↓

        # (b) trunk without GAP / FC
        self.features = nn.Sequential(*list(backbone.children())[:-2])

        # (c) 1×1 conv → per-pixel logits (B,11,H,W)
        self.logit_head = nn.Conv2d(2048, self.n_total, kernel_size=1)
        nn.init.kaiming_normal_(self.logit_head.weight, nonlinearity="relu")

    # ------------ helpers ------------------------------------------------
    @staticmethod
    def _smooth_max(x, t=0.5, dims=(-2, -1)):
        #   logmeanexp ≈ maxi  (smooth):  log( mean( e^(t·x) ) ) / t
        return (torch.log(torch.exp(x * t).mean(dim=dims))) / t

    def aggregate(self, seg_logits):
        """Spatially pool (B,C,H,W) → (B,C) with smooth-max."""
        return self._smooth_max(seg_logits, t=0.5, dims=(-2, -1))

    # ------------ forward ------------------------------------------------
    def forward(self, x):
        feats = self.features(x)                          # B×2048×H'×W'
        seg   = self.logit_head(feats)                    # B×11×H'×W'
        seg   = F.interpolate(seg, size=x.shape[-2:],     # back to 32×32
                              mode="bilinear",
                              align_corners=False)
        return seg


# ----------------------------------------------------------------------
# 2.  Training utilities
# ----------------------------------------------------------------------
def default_loaders(
        batch_size=128, num_workers=4,
        root=".", download=True):
    train_ld = Dataset('cifar10', {'colorize':False}).get_loader(batch_size)
    test_ld = Dataset('cifar10', {'colorize':False}, train=False).get_loader(batch_size)
    return train_ld, test_ld


# ----------------------------------------------------------------------
# 3.  Visualisation
# ----------------------------------------------------------------------
@torch.no_grad()
@torch.no_grad()
def save_epoch_preview(model, test_loader, device,
                       epoch, out_dir="."):
    """
    After each epoch save previews/epoch_XX.png with
    ( original , object-only ) pairs for 10 samples.
    """
    model.eval()
    images, _ = next(iter(test_loader))
    images = images[:10].to(device)                      # use 10 samples

    # -------- logits → per-pixel foreground mask ------------------
    seg        = model(images)                           # B×11×32×32
    agg        = model.aggregate(seg)                    # B×11
    pred_class = agg[:, :-1].argmax(1)                   # drop bg, B

    sel        = torch.arange(seg.size(0), device=device)
    cls_map    = seg[sel, pred_class]                    # B×32×32
    attn       = torch.sigmoid(cls_map.unsqueeze(1))     # B×1×32×32
    fg_mask = torch.zeros_like(attn)
    fg_mask[attn > 0.5] = 1.0                         # strong evidence
    fg_mask[(attn > 0.3) & (attn <= 0.5)] = 0.7       # weak evidence
    fg_mask[(attn > 0.2) & (attn <= 0.3)] = 0.5       # weak evidence


    # -------- denormalise originals -------------------------------
    #mean = torch.tensor([0.4914, 0.4822, 0.4465],
     #                   device=device).view(1, 3, 1, 1)
    #std  = torch.tensor([0.2023, 0.1994, 0.2010],
    #                    device=device).view(1, 3, 1, 1)
    #originals = torch.clamp(images * std + mean, 0, 1)   # B×3×32×32

    # -------- apply mask: background → black ----------------------
    # fg_mask is (B,1,32,32) ➜ broadcast to channels
    cropped = images * fg_mask                        # B×3×32×32

    # -------- make a grid: 5 pairs per row -----------------------
    pair_list = list(itertools.chain.from_iterable(
        zip(images.cpu(), cropped.cpu())))
    grid = make_grid(pair_list, nrow=5, padding=2)

    save_image(grid, f"../training_samples/seg1/epoch_{epoch:02d}.png")


# ----------------------------------------------------------------------
# 4.  Trainer
# ----------------------------------------------------------------------
def train(model, train_loader, test_loader,
          epochs=10, lr=0.05, device="cuda"):
    model = model.to(device)
    model = nn.DataParallel(model)  # multi-GPU if available

    criterion = nn.BCEWithLogitsLoss()
    optim     = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=1e-4)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=epochs * len(train_loader),
        eta_min=1e-3)

    n_total = model.module.n_total          # 11

    for ep in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep:02d}")
        for x, y in pbar:
            x, y = x.to(device), y[0].to(device)

            seg  = model(x)                          # B×11×32×32
            agg  = model.module.aggregate(seg)       # B×11

            foreground_logits = seg[:, :-1]            # drop background channel
            # pixel-wise BCE pushes the average foreground logit low
            sparsity_loss = 0.001 * torch.sigmoid(foreground_logits).mean()

            # one-hot + background label 0
            y_vec = F.one_hot(y, model.module.n_fg).float()  # B×10
            bg    = torch.zeros((y.size(0), 1),
                                 device=device, dtype=torch.float)
            target = torch.cat([y_vec, bg], dim=1)            # B×11

            loss = criterion(agg, target) + sparsity_loss

            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()

            pred = agg[:, :-1].argmax(1)
            acc  = (pred == y).float().mean().item()
            pbar.set_postfix(loss=loss.item(), acc=f"{acc*100:.1f}%")

        # ---- epoch-end preview & evaluation -------------------------
        save_epoch_preview(model.module, test_loader,
                           device=device, epoch=ep, out_dir="previews")

        model.eval()
        test_acc, num = 0.0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y[0].to(device)
                seg  = model(x)
                agg  = model.module.aggregate(seg)
                pred = agg[:, :-1].argmax(1)
                test_acc += (pred == y).sum().item()
                num      += y.size(0)
        print(f"↳ test accuracy: {100*test_acc/num:.2f}%\n")


# ----------------------------------------------------------------------
# 5.  Kick-off (run this from a main-guard if you save as .py)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    train_ld, test_ld = default_loaders(batch_size=128, num_workers=8)
    net = WeakResNet50Seg()
    train(net, train_ld, test_ld, epochs=30, lr=0.05)
