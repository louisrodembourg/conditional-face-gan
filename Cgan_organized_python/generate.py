import torch
import os
import torchvision.utils as vutils
from config import *
from models import Generator

def generate_inference(weights_path, specific_attrs=None):
    
    if not os.path.exists(weights_path):
        print(f"❌ Erreur: Le fichier {weights_path} n'existe pas.")
        return

    if specific_attrs is None:
        specific_attrs = (torch.rand(NUM_CLASSES) > 0.5).int().tolist()
    
    print(f"⏳ Chargement du modèle depuis : {os.path.basename(weights_path)}...")
    
    try:
        # On essaie de charger avec l'architecture par défaut (128x128)
        # Si le modèle a été entraîné avec img_size=64, il faudrait détecter ou passer un argument.
        # Ici on suppose 128 par défaut (config.py).
        netG = Generator(Z_DIM, NUM_CLASSES, img_size=IMAGE_SIZE).to(DEVICE)
        netG.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        print("✅ Modèle chargé avec succès.")
    except Exception as e:
        print(f"❌ Erreur lors du chargement des poids : {e}")
        print("Vérifiez que le checkpoint correspond à l'architecture actuelle.")
        return

    # --- SUITE DU SCRIPT ---
    netG.eval()

    z = torch.randn(1, Z_DIM, 1, 1).to(DEVICE)
    labels = torch.tensor([specific_attrs]).float().to(DEVICE)

    print(f"🎨 Génération en cours pour les attributs : {specific_attrs}")
    with torch.no_grad():
        fake_image = netG(z, labels).detach().cpu()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    attr_str = "_".join([f"{name[:4]}_{val}" for name, val in zip(ATTR_NAMES, specific_attrs)])
    save_filename = f"gen_{attr_str}.png"
    save_path = os.path.join(RESULTS_DIR, save_filename)

    vutils.save_image(fake_image, save_path, normalize=True, value_range=(-1, 1))
    print(f"✅ Image sauvegardée : {save_path}")
