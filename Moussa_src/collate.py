import torch

def pair_collate_concat(batch, normalize: bool = True):
    """
    batch: list of (l_uint8, r_uint8, y_pair, pair_idx)
    returns:
      x:   [B,1,28,56] float32 (normalized)
      y:   [B,2] long
      pid: [B]  long
    """
    l = torch.stack([b[0] for b in batch], 0)  # [B,1,28,28] uint8
    r = torch.stack([b[1] for b in batch], 0)  # [B,1,28,28] uint8
    x = torch.cat([l, r], dim=3)               # [B,1,28,56] uint8

    x = x.float().div_(255)                    # [0,1]
    if normalize:
        x = x.sub_(0.5).div_(0.5)              # [-1,1]

    y = torch.stack([b[2] for b in batch], 0)  # [B,2]
    pid = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return x, y, pid
