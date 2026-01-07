import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, num_classes, img_size=128):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.input_dim = z_dim + num_classes
        
        # Bloc initial commun (Z -> 4x4)
        self.fc = nn.Sequential(
            nn.ConvTranspose2d(self.input_dim, 512, 4, 1, 0, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(True)
        )
        
        # Couches progressives (4 -> 8 -> 16 -> 32 -> 64)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.InstanceNorm2d(256, affine=True), nn.ReLU(True)),
            nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.InstanceNorm2d(128, affine=True), nn.ReLU(True)),
            nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.InstanceNorm2d(64, affine=True), nn.ReLU(True)),
            nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False), nn.InstanceNorm2d(32, affine=True), nn.ReLU(True))
        ])
        
        if self.img_size == 128:
            # 64 -> 128
            self.extra_layer = nn.Sequential(nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False), nn.InstanceNorm2d(16, affine=True), nn.ReLU(True))
            last_channel = 16
        else:
            last_channel = 32

        self.to_rgb = nn.Sequential(
            nn.ConvTranspose2d(last_channel, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        labels = labels.view(labels.size(0), labels.size(1), 1, 1)
        x = torch.cat((noise, labels), 1)
        x = self.fc(x)
        for layer in self.layers:
            x = layer(x)
        if self.img_size == 128:
            x = self.extra_layer(x)
        return self.to_rgb(x)
class GeneratorLegacy128(nn.Module):
    """
    Architecture V2 'Hardcodée' pour 128x128.
    C'est celle qui correspond à ton checkpoint netG_epoch_49.
    """
    def __init__(self, z_dim, num_classes):
        super(GeneratorLegacy128, self).__init__()
        self.input_dim = z_dim + num_classes
        
        self.main = nn.Sequential(
            # Input is Z -> 4x4
            nn.ConvTranspose2d(self.input_dim, 512, 4, 1, 0, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(True),
            
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True),
            
            # 32x32 -> 64x64 (C'est cette couche qui manquait dans Legacy 64)
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(True),
            
            # 64x64 -> 128x128 (Sortie Finale)
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        labels = labels.view(labels.size(0), labels.size(1), 1, 1)
        gen_input = torch.cat((noise, labels), 1)
        return self.main(gen_input)
# --- AJOUT POUR COMPATIBILITÉ (V2) ---
class GeneratorLegacy(nn.Module):
    """L'ancienne architecture V2 pour charger les vieux checkpoints"""
    def __init__(self, z_dim, num_classes):
        super(GeneratorLegacy, self).__init__()
        self.input_dim = z_dim + num_classes
        
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(self.input_dim, 512, 4, 1, 0, bias=False),
            nn.InstanceNorm2d(512, affine=True), 
            nn.ReLU(True),
            # State size: (512) x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(True),
            # State size: (256) x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(True),
            # State size: (128) x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True),
            # State size: (64) x 32 x 32
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output size: (3) x 64 x 64
        )

    def forward(self, noise, labels):
        labels = labels.view(labels.size(0), labels.size(1), 1, 1)
        gen_input = torch.cat((noise, labels), 1)
        return self.main(gen_input)
    

class Discriminator(nn.Module):
    def __init__(self, num_classes, img_size=128, conv_dim=64):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        layers = []
        
        # Si 128px, on ajoute une couche au début (128 -> 64)
        if img_size == 128:
            layers.append(nn.Conv2d(3, conv_dim // 2, 4, 2, 1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = conv_dim // 2
        else:
            in_dim = 3
            
        # Architecture standard (64 -> 32 -> 16 -> 8 -> 4 -> 2)
        # Note: Pas de Batch Norm (WGAN-GP)
        channel_mults = [1, 2, 4, 8, 16] # Multiplicateurs de channels
        
        curr_dim = in_dim
        for mult in channel_mults:
            out_dim = conv_dim * mult
            layers.append(nn.Conv2d(curr_dim, out_dim, 4, 2, 1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            curr_dim = out_dim
        
        self.main = nn.Sequential(*layers)
        
        # Heads (Critic & Aux)
        self.conv_critic = nn.Conv2d(curr_dim, 1, 2, 1, 0, bias=False) # Kernel 2x2 pour finir
        self.conv_aux = nn.Conv2d(curr_dim, num_classes, 2, 1, 0, bias=False)

    def forward(self, img):
        out = self.main(img)
        validity = self.conv_critic(out).squeeze()
        label_preds = self.conv_aux(out).squeeze()
        return validity, label_preds