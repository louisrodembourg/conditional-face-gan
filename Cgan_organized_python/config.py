import torch
import os

# --- HARDWARE ---
# Détection automatique MPS / CUDA / CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# --- PATHS ---
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Dossier contenant config.py (Cgan_organized_python)
PROJECT_ROOT = os.path.dirname(BASE_DIR) # Dossier parent (Cgan_project_Parme)

DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
ATTR_FILE = os.path.join(DATA_ROOT, 'list_attr_celeba.csv')
IMG_DIR = os.path.join(DATA_ROOT, 'img_align_celeba/img_align_celeba') # Chemin corrigé pour matcher le notebook (img_align_celeba/img_align_celeba)
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
CKPT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'generated_results')
FID_OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'fid_outputs')

# --- HYPERPARAMETRES ---
IMAGE_SIZE = 128
Z_DIM = 100
BATCH_SIZE = 64        # Passe à 128 si ta RAM le permet
NUM_EPOCHS = 50
LR = 0.0003
BETA1 = 0.5
BETA2 = 0.999
N_CRITIC = 5           # Ratio WGAN (5 D pour 1 G)

# --- WGAN & LOSS ---
LAMBDA_GP = 10.0       # Gradient Penalty
LAMBDA_CLS = 10.0      # Poids de la classification (Force les attributs)

# --- ATTRIBUTS & POIDS ---
# Liste des 5 attributs
ATTR_NAMES = ['Male', 'Blond_Hair', 'Smiling', 'Eyeglasses', 'Young']
NUM_CLASSES = len(ATTR_NAMES)

# Poids pour compenser le déséquilibre (Ex: Lunettes très rares)
# [Male, Blond, Smiling, Eyeglasses, Young]
CLASS_WEIGHTS = [1.0, 3.0, 1.0, 15.0, 1.0]

# --- TRAINING OPTIONS ---
DECAY_START_EPOCH = 25 # Début de la baisse du LR
FID_INTERVAL = 5       # Calcul du FID toutes les X époques