import torch
import torch.nn as nn
import numpy as np
import random
import os
import torchvision.utils as vutils
from config import DEVICE

# 1. Reproductibilité
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Seed fixée à : {seed}")

# 2. Initialisation des poids (DCGAN paper)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
        # InstanceNorm a des params affines si affine=True
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# 3. WGAN-GP : Gradient Penalty
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(DEVICE)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # Pass interpolates through Discriminator (get validity only)
    d_interpolates, _ = D(interpolates)
    
    fake = torch.ones(real_samples.size(0), requires_grad=False).to(DEVICE)
    
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates.view(-1),
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# 4. Metrics
def compute_accuracy(preds, labels):
    # Les preds sont des logits. Si > 0 -> Classe 1
    predicted_labels = (preds > 0).float()
    correct = (predicted_labels == labels).float()
    return correct.mean()

def compute_pos_weight(labels):
    """pos_weight for BCEWithLogitsLoss: (#negatives / #positives) per attribute."""
    labels = torch.as_tensor(labels, dtype=torch.float32)
    pos = labels.sum(dim=0).clamp(min=1.0)
    neg = labels.size(0) - pos
    return neg / pos

def sample_real_labels(label_pool, n, device):
    """
    Samples n attribute vectors from the empirical label distribution (rows of the dataset).
    Keeps rare attributes (e.g. Eyeglasses) at their real frequency and only produces
    attribute combinations that actually exist, instead of independent Bernoulli(0.5) draws.
    """
    idx = torch.randint(0, label_pool.size(0), (n,))
    return label_pool[idx].to(device)

# 5. Scheduler
class LinearDecayLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
# 6. Disentanglement Visualization
def make_attribute_grid(netG, z_dim, num_classes, device, n_points=10):
    """
    Crée une grille de visualisation où chaque ligne correspond à l'interpolation d'un attribut (0 -> 1).
    On utilise le MÊME vecteur de bruit z pour toute la grille.
    """
    netG.eval()
    
    # 1. Un z fixe pour une personne de base (Identity)
    # (1, 100, 1, 1)
    base_z = torch.randn(1, z_dim, 1, 1).to(device)
    
    # 2. Des labels de base (Tout à 0 par défaut pour bien voir l'effet de l'ajout)
    # 0 = Femme, Brune, Pas Souriante, Pas Lunettes, Vieux
    base_labels = torch.zeros(1, num_classes).to(device) 
    
    all_imgs = []
    
    # Pour chaque attribut (chaque ligne de la grille)
    for attr_idx in range(num_classes):
        # On crée n_points images pour la transition
        vals = torch.linspace(0, 1, n_points).to(device)
        
        # On duplique le z et les labels de base
        batch_z = base_z.repeat(n_points, 1, 1, 1)
        batch_labels = base_labels.repeat(n_points, 1)
        
        # On fait varier SEULEMENT l'attribut courant
        batch_labels[:, attr_idx] = vals
        
        # Génération
        with torch.no_grad():
            gen_imgs = netG(batch_z, batch_labels)
        
        all_imgs.append(gen_imgs)
        
    # Concaténer tout (Num_Attr * N_Points, 3, 128, 128)
    all_imgs = torch.cat(all_imgs, dim=0)
    
    # Faire une grille (nrow = n_points pour avoir une ligne par attribut)
    return vutils.make_grid(all_imgs, nrow=n_points, normalize=True, padding=2)
