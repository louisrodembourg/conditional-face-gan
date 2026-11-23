"""
Configuration Module
====================

This module contains all the hyperparameters, paths, and hardware settings for the AC-WGAN-GP project.
Centralizing configuration here ensures consistency across training, generation, and evaluation scripts.
"""

import os
import torch

# --- HARDWARE CONFIGURATION ---
# Automatic detection for Apple Silicon (MPS), CUDA, or CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

# --- PATHS ---
# Base directory of the source code (src)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
# Project root directory
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
ATTR_FILE = os.path.join(DATA_ROOT, 'list_attr_celeba.csv')
# Image directory. Adjust consistency with your dataset structure.
# Often CelebA is unzipped as img_align_celeba/img_align_celeba/xxxxx.jpg
IMG_DIR = os.path.join(DATA_ROOT, 'img_align_celeba/img_align_celeba') 

LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
CKPT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'generated_results')
FID_OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'fid_outputs')

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FID_OUTPUTS_DIR, exist_ok=True)

# --- HYPERPARAMETERS ---
IMAGE_SIZE = 128
Z_DIM = 100
BATCH_SIZE = 64        # Increase to 128 or 256 if VRAM allows
NUM_EPOCHS = 50

# Optimizer settings
LR = 0.0002            # Standard GAN learning rate (often 0.0002)
BETA1 = 0.5            # Adam Beta1
BETA2 = 0.999          # Adam Beta2

# Training ratios
N_CRITIC = 5           # Train Discriminator 5 times for every 1 Generator update (WGAN-GP standard)

# --- LOSS WEIGHTS ---
LAMBDA_GP = 10.0       # Gradient Penalty coefficient
LAMBDA_CLS = 10.0      # Classification loss coefficient (Control attribute strength)

# --- ATTRIBUTES ---
# List of selected attributes from CelebA
ATTR_NAMES = ['Male', 'Blond_Hair', 'Smiling', 'Eyeglasses', 'Young']
NUM_CLASSES = len(ATTR_NAMES)

# Class weights (BCE pos_weight) to handle dataset imbalance (e.g. Eyeglasses are rare).
# None -> computed automatically from the training set as (#negatives / #positives) per attribute.
# Set a list (same order as ATTR_NAMES) to override manually.
CLASS_WEIGHTS = None

# --- SCHEDULING & LOGGING ---
DECAY_START_EPOCH = 25 # Epoch to start linear learning rate decay
FID_INTERVAL = 5       # Calculate FID every X epochs (expensive operation)
RUN_TAG = "acwgan_gp"  # Suffix appended to the timestamped run name
