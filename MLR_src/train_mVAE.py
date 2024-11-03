# prerequisites
import torch
from MLR_src.mVAE import train
import torch.optim as optim

def train_mVAE(dataloaders, vae, epoch_count, checkpoint_folder, use_wandb, start_epoch = 1):
    if use_wandb is True:
        import wandb
        from MLR_src.wandb_setup import initialize_wandb, log_system_metrics
        initialize_wandb('2d-retina-train', {'version':'MLR_2.0_2D_RETINA'})

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    seen_labels = {}
    components = ['color'] + ['shape'] + ['cropped'] * 2 + ['skip_cropped'] + ['location'] * 2

    for epoch in range(start_epoch, epoch_count):
        if epoch >= 120:
            components = ['location', 'retinal', 'retinal']

        loss_lst, seen_labels = train(vae, optimizer, epoch, dataloaders, True, seen_labels, components)

        if use_wandb is True:
            wandb.log({
            'epoch': epoch,
            'retinal/training_loss': loss_lst[0],
            'retinal/test_loss': loss_lst[1],
            'cropped/training_loss': loss_lst[2],
            'cropped/test_loss': loss_lst[3]
            })

        torch.cuda.empty_cache()
        vae.eval()
        checkpoint =  {
            'state_dict': vae.state_dict(),
            #'labels': seen_labels
                    }
        torch.save(checkpoint, f'checkpoints/{checkpoint_folder}/mVAE_checkpoint.pth')
    
    if use_wandb is True:
        wandb.finish()