"""
Training Module
===============

This module handles the training loop for the AC-WGAN-GP model.
It manages data loading, model initialization, optimization, logging, and checkpointing.
"""

import os
import subprocess
from datetime import datetime
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from models import Generator, Discriminator
from utils import (
    seed_everything,
    weights_init,
    compute_gradient_penalty,
    compute_accuracy,
    LinearDecayLR,
    make_attribute_grid
)
from data import get_dataloaders


def prepare_fid_folders(netG: nn.Module, test_loader: torch.utils.data.DataLoader, run_name: str, epoch: int) -> Tuple[str, str]:
    """
    Generates real and fake images for Frechet Inception Distance (FID) calculation.

    Args:
        netG (nn.Module): The generator model.
        test_loader (DataLoader): DataLoader for real images.
        run_name (str): Identifier for the current training run.
        epoch (int): Current epoch number.

    Returns:
        Tuple[str, str]: Paths to the directories containing real and fake images.
    """
    real_dir = os.path.join(config.FID_OUTPUTS_DIR, "real")
    fake_dir = os.path.join(config.FID_OUTPUTS_DIR, f"fake_{run_name}_ep{epoch}")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    # Save Real (Only once if empty)
    if len(os.listdir(real_dir)) < 500: # Take a sample
        print("Generating Real FID directory...")
        count = 0
        for imgs, _ in test_loader:
            for i in range(imgs.size(0)):
                vutils.save_image(imgs[i], os.path.join(real_dir, f"{count}.png"), normalize=True, value_range=(-1, 1))
                count += 1
                if count >= 2000: break
            if count >= 2000: break

    # Save Fake
    print("Generating Fake FID directory...")
    netG.eval()
    count = 0
    with torch.no_grad():
        while count < 2000:
            z = torch.randn(config.BATCH_SIZE, config.Z_DIM, 1, 1).to(config.DEVICE)
            # Random labels
            labels = (torch.rand(config.BATCH_SIZE, config.NUM_CLASSES) > 0.5).float().to(config.DEVICE)
            fakes = netG(z, labels)
            for i in range(fakes.size(0)):
                vutils.save_image(fakes[i], os.path.join(fake_dir, f"{count}.png"), normalize=True, value_range=(-1, 1))
                count += 1
                if count >= 2000: break
    netG.train()
    return real_dir, fake_dir

def train():
    """
    Main training function.
    """
    seed_everything(42)
    
    # 1. Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(config.IMG_DIR, config.ATTR_FILE)
    
    # 2. Models
    netG = Generator(config.Z_DIM, config.NUM_CLASSES, img_size=config.IMAGE_SIZE).to(config.DEVICE)
    netD = Discriminator(config.NUM_CLASSES, img_size=config.IMAGE_SIZE).to(config.DEVICE)
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # 3. Optimizers & Loss
    optimizerG = optim.Adam(netG.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))
    optimizerD = optim.Adam(netD.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))
    
    # Weighted Loss for attributes
    class_weights = torch.tensor(config.CLASS_WEIGHTS).to(config.DEVICE)
    cls_criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    # Schedulers
    schedulerG = torch.optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=LinearDecayLR(config.NUM_EPOCHS, 0, config.DECAY_START_EPOCH).step)
    schedulerD = torch.optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda=LinearDecayLR(config.NUM_EPOCHS, 0, config.DECAY_START_EPOCH).step)

    # 4. Logging
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M_V3_ULTIMATE")
    writer = SummaryWriter(os.path.join(config.LOG_DIR, run_name))
    ckpt_path = os.path.join(config.CKPT_DIR, run_name)
    os.makedirs(ckpt_path, exist_ok=True)
    
    # Fixed Visualization Vectors
    fixed_z = torch.randn(64, config.Z_DIM, 1, 1).to(config.DEVICE)
    fixed_labels = (torch.rand(64, config.NUM_CLASSES) > 0.5).float().to(config.DEVICE)
    
    best_fid = float('inf')
    print(f"🚀 Starting Run: {run_name}")

    # 5. Training Loop
    global_step = 0
    for epoch in range(config.NUM_EPOCHS):
        for i, (real_imgs, real_labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            real_imgs = real_imgs.to(config.DEVICE)
            real_labels = real_labels.float().to(config.DEVICE)
            b_size = real_imgs.size(0)
            
            # --- TRAIN D ---
            optimizerD.zero_grad()
            
            # Real
            real_validity, real_cls = netD(real_imgs)
            d_loss_real = -torch.mean(real_validity)
            d_loss_cls = cls_criterion(real_cls, real_labels)
            
            # Fake
            z = torch.randn(b_size, config.Z_DIM, 1, 1).to(config.DEVICE)
            fake_labels_in = (torch.rand(b_size, config.NUM_CLASSES) > 0.5).float().to(config.DEVICE)
            fake_imgs = netG(z, fake_labels_in)
            
            fake_validity, _ = netD(fake_imgs.detach())
            d_loss_fake = torch.mean(fake_validity)
            
            # Gradient Penalty
            gp = compute_gradient_penalty(netD, real_imgs.data, fake_imgs.data)
            
            d_loss = d_loss_real + d_loss_fake + config.LAMBDA_GP * gp + config.LAMBDA_CLS * d_loss_cls
            
            # --- CALCUL ACCURACY (REAL) ---
            acc_real = compute_accuracy(real_cls, real_labels)
            
            d_loss.backward()
            optimizerD.step()
            
            # --- TRAIN G ---
            if i % config.N_CRITIC == 0:
                optimizerG.zero_grad()
                
                fake_validity, fake_cls = netD(fake_imgs)
                g_loss_adv = -torch.mean(fake_validity)
                g_loss_cls = cls_criterion(fake_cls, fake_labels_in)
                
                # --- CALCUL ACCURACY (FAKE) ---
                acc_fake = compute_accuracy(fake_cls, fake_labels_in)
                
                g_loss = g_loss_adv + config.LAMBDA_CLS * g_loss_cls
                g_loss.backward()
                optimizerG.step()
                
                if global_step % 50 == 0:
                    # 1. Main Losses
                    writer.add_scalar('Loss/D_Total', d_loss.item(), global_step)
                    writer.add_scalar('Loss/G_Total', g_loss.item(), global_step)
                    
                    # 2. WGAN Details
                    writer.add_scalar('WGAN/Wasserstein_Dist', -(d_loss_real+d_loss_fake).item(), global_step)
                    writer.add_scalar('WGAN/Gradient_Penalty', gp.item(), global_step)
                    
                    # 3. Classification Details (AC-GAN)
                    writer.add_scalar('Classification/Real_Images', d_loss_cls.item(), global_step)
                    writer.add_scalar('Classification/Fake_Images', g_loss_cls.item(), global_step)
                    
                    # 4. Accuracy (Attributes)
                    writer.add_scalar('Accuracy_Attributes/Real', acc_real.item(), global_step)
                    writer.add_scalar('Accuracy_Attributes/Fake', acc_fake.item(), global_step)
                    
            global_step += 1
            
        # End of Epoch
        schedulerG.step()
        schedulerD.step()
        
        # Save & Visu
        with torch.no_grad():
            netG.eval()
            
            # 1. Random Grid
            fake = netG(fixed_z, fixed_labels).detach().cpu()
            grid = vutils.make_grid(fake, padding=2, normalize=True)
            writer.add_image('Generated_Random', grid, epoch)
            
            # 2. Disentanglement Grid
            dis_grid = make_attribute_grid(netG, config.Z_DIM, config.NUM_CLASSES, config.DEVICE)
            writer.add_image('Disentanglement_Grid_Interpolation', dis_grid, epoch)
            
            netG.train()
            
        # Checkpoint
        torch.save(netG.state_dict(), os.path.join(ckpt_path, f'netG_ep{epoch}.pth'))
        torch.save(netD.state_dict(), os.path.join(ckpt_path, f'netD_ep{epoch}.pth'))
        
        # FID Check (Every few epochs)
        if epoch > 0 and epoch % config.FID_INTERVAL == 0:
            try:
                r_dir, f_dir = prepare_fid_folders(netG, test_loader, run_name, epoch)
                cmd = f"python -m pytorch_fid \"{r_dir}\" \"{f_dir}\" --device {config.DEVICE.type}"
                result = subprocess.check_output(cmd, shell=True).decode()
                fid = float(result.split('FID:')[-1].strip())
                writer.add_scalar('Metrics/FID', fid, epoch)
                print(f"📊 FID: {fid}")
                
                if fid < best_fid:
                    best_fid = fid
                    torch.save(netG.state_dict(), os.path.join(ckpt_path, 'best_fid_model.pth'))
            except Exception as e:
                print(f"Error computing FID: {e}")

    print("Training finished.")