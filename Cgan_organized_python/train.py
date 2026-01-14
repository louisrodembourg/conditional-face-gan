import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from tqdm import tqdm

from config import *
from models import Generator, Discriminator
from utils import seed_everything, weights_init, compute_gradient_penalty, compute_accuracy, LinearDecayLR, make_attribute_grid
from data import get_dataloaders

# Import pour le FID (nécessite fonction locale ou script externe)
# Nous allons utiliser un appel système simple pour pytorch-fid
import subprocess

def prepare_fid_folders(netG, test_loader, run_name, epoch):
    """Génère les images réelles et fausses pour le calcul FID"""
    real_dir = os.path.join(FID_OUTPUTS_DIR, "real")
    fake_dir = os.path.join(FID_OUTPUTS_DIR, f"fake_{run_name}_ep{epoch}")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    # Save Real (Une seule fois si vide)
    if len(os.listdir(real_dir)) < 500: # On prend un échantillon
        print("Génération dossier Real FID...")
        count = 0
        for imgs, _ in test_loader:
            for i in range(imgs.size(0)):
                vutils.save_image(imgs[i], os.path.join(real_dir, f"{count}.png"), normalize=True, value_range=(-1, 1))
                count += 1
                if count >= 2000: break
            if count >= 2000: break

    # Save Fake
    print("Génération dossier Fake FID...")
    netG.eval()
    count = 0
    with torch.no_grad():
        while count < 2000:
            z = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(DEVICE)
            # Random labels
            labels = (torch.rand(BATCH_SIZE, NUM_CLASSES) > 0.5).float().to(DEVICE)
            fakes = netG(z, labels)
            for i in range(fakes.size(0)):
                vutils.save_image(fakes[i], os.path.join(fake_dir, f"{count}.png"), normalize=True, value_range=(-1, 1))
                count += 1
                if count >= 2000: break
    netG.train()
    return real_dir, fake_dir

def train():
    seed_everything(42)
    
    # 1. Data
    print("Chargement des données...")
    train_loader, val_loader, test_loader = get_dataloaders(IMG_DIR, ATTR_FILE)
    
    # 2. Models
    netG = Generator(Z_DIM, NUM_CLASSES, img_size=IMAGE_SIZE).to(DEVICE)
    netD = Discriminator(NUM_CLASSES, img_size=IMAGE_SIZE).to(DEVICE)
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # 3. Optimizers & Loss
    optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, BETA2))
    optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, BETA2))
    
    # Weighted Loss pour les attributs
    class_weights = torch.tensor(CLASS_WEIGHTS).to(DEVICE)
    cls_criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    # Schedulers
    schedulerG = torch.optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=LinearDecayLR(NUM_EPOCHS, 0, DECAY_START_EPOCH).step)
    schedulerD = torch.optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda=LinearDecayLR(NUM_EPOCHS, 0, DECAY_START_EPOCH).step)

    # 4. Logging
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M_V3_ULTIMATE")
    writer = SummaryWriter(os.path.join(LOG_DIR, run_name))
    ckpt_path = os.path.join(CKPT_DIR, run_name)
    os.makedirs(ckpt_path, exist_ok=True)
    
    # Visu Fixe
    fixed_z = torch.randn(64, Z_DIM, 1, 1).to(DEVICE)
    fixed_labels = (torch.rand(64, NUM_CLASSES) > 0.5).float().to(DEVICE)
    
    best_fid = float('inf')
    print(f"🚀 Démarrage Run: {run_name}")

    # 5. Boucle
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        for i, (real_imgs, real_labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            real_imgs = real_imgs.to(DEVICE)
            real_labels = real_labels.float().to(DEVICE)
            b_size = real_imgs.size(0)
            
            # --- TRAIN D ---
            optimizerD.zero_grad()
            
            # Real
            real_validity, real_cls = netD(real_imgs)
            d_loss_real = -torch.mean(real_validity)
            d_loss_cls = cls_criterion(real_cls, real_labels)
            
            # Fake
            z = torch.randn(b_size, Z_DIM, 1, 1).to(DEVICE)
            fake_labels_in = (torch.rand(b_size, NUM_CLASSES) > 0.5).float().to(DEVICE)
            fake_imgs = netG(z, fake_labels_in)
            
            fake_validity, _ = netD(fake_imgs.detach())
            d_loss_fake = torch.mean(fake_validity)
            
            # GP
            gp = compute_gradient_penalty(netD, real_imgs.data, fake_imgs.data)
            
            d_loss = d_loss_real + d_loss_fake + LAMBDA_GP * gp + LAMBDA_CLS * d_loss_cls
            
            # --- CALCUL ACCURACY (REAL) ---
            acc_real = compute_accuracy(real_cls, real_labels)
            
            d_loss.backward()
            optimizerD.step()
            
            # --- TRAIN G ---
            if i % N_CRITIC == 0:
                optimizerG.zero_grad()
                
                fake_validity, fake_cls = netD(fake_imgs)
                g_loss_adv = -torch.mean(fake_validity)
                g_loss_cls = cls_criterion(fake_cls, fake_labels_in)
                
                # --- CALCUL ACCURACY (FAKE) ---
                acc_fake = compute_accuracy(fake_cls, fake_labels_in)
                
                g_loss = g_loss_adv + LAMBDA_CLS * g_loss_cls
                g_loss.backward()
                optimizerG.step()
                
                if global_step % 50 == 0:
                    # 1. Pertes principales
                    writer.add_scalar('Loss/D_Total', d_loss.item(), global_step)
                    writer.add_scalar('Loss/G_Total', g_loss.item(), global_step)
                    
                    # 2. Détails WGAN
                    writer.add_scalar('WGAN/Wasserstein_Dist', -(d_loss_real+d_loss_fake).item(), global_step)
                    writer.add_scalar('WGAN/Gradient_Penalty', gp.item(), global_step)
                    
                    # 3. Détails Classification (AC-GAN)
                    writer.add_scalar('Classification/Real_Images', d_loss_cls.item(), global_step)
                    writer.add_scalar('Classification/Fake_Images', g_loss_cls.item(), global_step)
                    
                    # 4. Accuracy (Reconnaissance des Attributs)
                    writer.add_scalar('Accuracy_Attributes/Real', acc_real.item(), global_step)
                    writer.add_scalar('Accuracy_Attributes/Fake', acc_fake.item(), global_step)
                    
            global_step += 1
            
        # Fin Epoch
        schedulerG.step()
        schedulerD.step()
        
        # Save & Visu
        with torch.no_grad():
            netG.eval()
            
            # 1. Grille Aléatoire standard
            fake = netG(fixed_z, fixed_labels).detach().cpu()
            grid = vutils.make_grid(fake, padding=2, normalize=True)
            writer.add_image('Generated_Random', grid, epoch)
            
            # 2. Grille de Disentanglement (Variation Attributs)
            dis_grid = make_attribute_grid(netG, Z_DIM, NUM_CLASSES, DEVICE)
            writer.add_image('Disentanglement_Grid_Interpolation', dis_grid, epoch)
            
            netG.train()
            
        # Checkpoint
        torch.save(netG.state_dict(), os.path.join(ckpt_path, f'netG_ep{epoch}.pth'))
        torch.save(netD.state_dict(), os.path.join(ckpt_path, f'netD_ep{epoch}.pth'))
        
        # FID Check (Tous les 5 epochs)
        if epoch > 0 and epoch % FID_INTERVAL == 0:
            try:
                r_dir, f_dir = prepare_fid_folders(netG, test_loader, run_name, epoch)
                cmd = f"python -m pytorch_fid \"{r_dir}\" \"{f_dir}\" --device {DEVICE.type}"
                result = subprocess.check_output(cmd, shell=True).decode()
                fid = float(result.split('FID:')[-1].strip())
                writer.add_scalar('Metrics/FID', fid, epoch)
                print(f"📊 FID: {fid}")
                
                if fid < best_fid:
                    best_fid = fid
                    torch.save(netG.state_dict(), os.path.join(ckpt_path, 'best_fid_model.pth'))
            except Exception as e:
                print(f"Erreur FID: {e}")

    print("Fin de l'entraînement.")