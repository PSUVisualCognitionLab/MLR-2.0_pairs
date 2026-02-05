import os
import random
import numpy as np
import torch

def set_all_seeds(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics(logits_l: torch.Tensor, logits_r: torch.Tensor, y: torch.Tensor):
    pred_l = logits_l.argmax(1)
    pred_r = logits_r.argmax(1)
    acc_l = (pred_l == y[:, 0]).float().mean().item()
    acc_r = (pred_r == y[:, 1]).float().mean().item()
    pair_acc = ((pred_l == y[:, 0]) & (pred_r == y[:, 1])).float().mean().item()
    return acc_l, acc_r, pair_acc
