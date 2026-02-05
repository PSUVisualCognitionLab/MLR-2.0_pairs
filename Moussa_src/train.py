import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .dataset_fast import QuickDrawPairDatasetFast
from .collate import pair_collate_concat
from .model import SmallTwoHeadCNN
from .utils import set_all_seeds, compute_metrics


def seed_worker(worker_id: int):
    # ensure distinct RNG per worker
    import random, numpy as np
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(root: str,
                class_pairs,
                batch_size: int = 512,
                num_workers: int = 4,
                items_per_pair: int = 50000,
                seed: int = 123):
    ds = QuickDrawPairDatasetFast(
        root=root,
        class_pairs=class_pairs,
        classes_dirname="",   # change to "" if your files aren't under a "bitmaps" subfolder
        items_per_pair=items_per_pair,
        seed=seed,
    )

    # Older PyTorch: seed the global RNG so DataLoader's shuffle is reproducible
    torch.manual_seed(seed)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pair_collate_concat,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,   # keeps NumPy/Python RNGs distinct per worker
        # persistent_workers and generator are not supported on your version
    )
    return loader, ds


def train_one_epoch(model, loader, device, optimizer, loss_fn):
    model.train()
    total_loss, n = 0.0, 0
    acc_l_sum = acc_r_sum = pair_acc_sum = 0.0

    for xb, yb, _pid in tqdm(loader, desc="train", ncols=80):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits_l, logits_r = model(xb)
        loss = loss_fn(logits_l, yb[:, 0]) + loss_fn(logits_r, yb[:, 1])
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        acc_l, acc_r, pair_acc = compute_metrics(logits_l.detach(), logits_r.detach(), yb)
        acc_l_sum += acc_l * bs
        acc_r_sum += acc_r * bs
        pair_acc_sum += pair_acc * bs
        n += bs

    return total_loss / n, acc_l_sum / n, acc_r_sum / n, pair_acc_sum / n


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss, n = 0.0, 0
    acc_l_sum = acc_r_sum = pair_acc_sum = 0.0

    for xb, yb, _pid in tqdm(loader, desc="eval", ncols=80):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits_l, logits_r = model(xb)
        loss = loss_fn(logits_l, yb[:, 0]) + loss_fn(logits_r, yb[:, 1])

        bs = xb.size(0)
        total_loss += loss.item() * bs
        acc_l, acc_r, pair_acc = compute_metrics(logits_l, logits_r, yb)
        acc_l_sum += acc_l * bs
        acc_r_sum += acc_r * bs
        pair_acc_sum += pair_acc * bs
        n += bs

    return total_loss / n, acc_l_sum / n, acc_r_sum / n, pair_acc_sum / n


def main():
    parser = argparse.ArgumentParser(description="Train a two-head CNN on on-the-fly QuickDraw class pairs")
    parser.add_argument("--data_root", type=str, default="data", help="root folder containing the 'bitmaps/' subfolder")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--items_per_pair_train", type=int, default=50000,
                        help="train epoch size per pair")
    parser.add_argument("--items_per_pair_eval", type=int, default=4000,
                        help="val epoch size per pair")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    # edit this list to your pairs; make sure files exist in data/bitmaps/
    CLASS_PAIRS = [
        ("airplane", "bird"),
        ("car", "cat"),
        ("dog", "duck"),
        ("frog", "horse"),
        ("sailboat", "truck"),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_all_seeds(args.seed)

    train_loader, train_ds = make_loader(
        root=args.data_root,
        class_pairs=CLASS_PAIRS,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        items_per_pair=args.items_per_pair_train,
        seed=args.seed,
    )
    val_loader, _ = make_loader(
        root=args.data_root,
        class_pairs=CLASS_PAIRS,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        items_per_pair=args.items_per_pair_eval,
        seed=args.seed + 1,
    )

    num_classes = len(train_ds.classes)
    print(f"Unique classes ({num_classes}): {train_ds.classes}")

    model = SmallTwoHeadCNN(num_classes=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        train_loader.dataset.set_epoch(epoch)
        val_loader.dataset.set_epoch(epoch)

        tr_loss, tr_acc_l, tr_acc_r, tr_pair = train_one_epoch(model, train_loader, device, optimizer, loss_fn)
        va_loss, va_acc_l, va_acc_r, va_pair = evaluate(model, val_loader, device, loss_fn)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} | L {tr_acc_l:.3f} R {tr_acc_r:.3f} Pair {tr_pair:.3f} || "
            f"val loss {va_loss:.4f} | L {va_acc_l:.3f} R {va_acc_r:.3f} Pair {va_pair:.3f}"
        )


if __name__ == "__main__":
    # required on Windows when num_workers>0
    main()
