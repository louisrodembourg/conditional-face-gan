"""
GAN Models Module
=================

This module defines the Generator and Discriminator architectures for the AC-WGAN-GP.
It uses standard PyTorch `nn.Module` classes.

Classes:
    Generator: The generator network that produces images from latent vectors and class labels.
    Discriminator: The discriminator network that classifies images as real/fake and predicts class labels.
"""

import torch
import torch.nn as nn
from typing import Tuple

class Generator(nn.Module):
    """
    Generator Network.

    This network takes a latent vector z and a class label vector as input,
    concatenates them, and upsamples them to an image using transposed convolutions.

    Args:
        z_dim (int): Dimension of the latent vector z.
        num_classes (int): Number of classes for the conditional input.
        img_size (int): Target image size. Defaults to 128.
    """
    def __init__(self, z_dim: int, num_classes: int, img_size: int = 128):
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
            # Fallback (unused in current project configuration)
            self.final = nn.Identity()

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Generator.

        Args:
            noise (torch.Tensor): Latent vector (Batch, Z_Dim, 1, 1).
            labels (torch.Tensor): Class labels (Batch, Num_Classes).

        Returns:
            torch.Tensor: Generated images (Batch, Channels, Height, Width).
        """
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
    """
    Discriminator Network.

    This network takes an image as input and downsamples it.
    It outputs two things:
    1. A validity score (how real the image looks).
    2. A class prediction (multi-label classification).

    Args:
        num_classes (int): Number of classes to predict.
        img_size (int): Input image size. Defaults to 128.
    """
    def __init__(self, num_classes: int, img_size: int = 128):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        
        # 128x128 -> 64x64 
        self.block0 = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 64x64 -> 32x32
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
        

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Discriminator.

        Args:
            img (torch.Tensor): Input images (Batch, C, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - validity: Real/Fake score (Batch,).
                - label_preds: Predicted class scores (Batch, Num_Classes).
        """
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
