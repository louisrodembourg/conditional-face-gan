import torch
import os

# --- HARDWARE ---
# Détection automatique MPS / CUDA / CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# --- PATHS ---
DATA_ROOT = 'Project/Cgan_project_Parme/data'
ATTR_FILE = os.path.join(DATA_ROOT, 'list_attr_celeba.csv')
IMG_DIR = os.path.join(DATA_ROOT, 'img_align_celeba') # Vérifie ton chemin exact
LOG_DIR = 'Project/Cgan_project_Parme/logs'
CKPT_DIR = 'Project/Cgan_project_Parme/checkpoints'
RESULTS_DIR = 'Project/Cgan_project_Parme/generated_results'

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