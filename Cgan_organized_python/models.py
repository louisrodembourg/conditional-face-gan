import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, num_classes, img_size=128):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.input_dim = z_dim + num_classes
        
        # 1. Input Z -> 4x4
        self.block0 = nn.Sequential(
            nn.ConvTranspose2d(self.input_dim, 512, 4, 1, 0, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(True)
        )
        
        # 2. 4x4 -> 8x8
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(True)
        )
        
        # 3. 8x8 -> 16x16
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(True)
        )
        
        # 4. 16x16 -> 32x32
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True)
        )
        
        # 5. 32x32 -> 64x64
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(True)
        )
        
        # 6. Final Layer
        if self.img_size == 128:
            # 64x64 -> 128x128
            self.final = nn.Sequential(
                nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
                nn.Tanh()
            )
        else:
            # Fallback (non utilisé normalement)
            self.final = nn.Identity()

    def forward(self, noise, labels):
        labels = labels.view(labels.size(0), labels.size(1), 1, 1)
        x = torch.cat((noise, labels), 1)
        
        x = self.block0(x) # 4
        x = self.block1(x) # 8
        x = self.block2(x) # 16
        x = self.block3(x) # 32
        x = self.block4(x) # 64
        
        if self.img_size == 128:
            x = self.final(x) # 128
        
        return x

class Discriminator(nn.Module):
    def __init__(self, num_classes, img_size=128):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        
        # 128x128 -> 64x64 (Si img_size=128)
        self.block0 = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 64x64 -> 32x32
        # Si img_size=64, input channels = 3. Si 128, input = 32.
        in_ch = 32 if img_size == 128 else 3
        
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 32x32 -> 16x16
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 16x16 -> 8x8
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 8x8 -> 4x4
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Heads (Input 4x4, 512 ch)
        self.conv_critic = nn.Conv2d(512, 1, 4, 1, 0, bias=False)
        self.conv_aux = nn.Conv2d(512, num_classes, 4, 1, 0, bias=False)
        

    def forward(self, img):
        if self.img_size == 128:
            out = self.block0(img)
        else:
            out = img
            
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        
        validity = self.conv_critic(out).squeeze()
        label_preds = self.conv_aux(out).squeeze()
        
        return validity, label_preds
