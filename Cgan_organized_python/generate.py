import torch
import os
import torchvision.utils as vutils
from config import *
# On importe les 3 classes
from models import Generator, GeneratorLegacy, GeneratorLegacy128 

def generate_inference(weights_path, specific_attrs=None):
    
    if not os.path.exists(weights_path):
        print(f"❌ Erreur: Le fichier {weights_path} n'existe pas.")
        return

    if specific_attrs is None:
        specific_attrs = (torch.rand(NUM_CLASSES) > 0.5).int().tolist()
    
    # --- LOGIQUE DE CHARGEMENT ROBUSTE ---
    print(f"⏳ Analyse du fichier : {os.path.basename(weights_path)}...")
    netG = None
    
    # ESSAI 1 : Architecture V3 (Modular 128)
    try:
        model = Generator(Z_DIM, NUM_CLASSES, img_size=IMAGE_SIZE).to(DEVICE)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        print("✅ Succès : Architecture V3 détectée.")
        netG = model
    except:
        pass

    # ESSAI 2 : Architecture Legacy 128 (Ton checkpoint epoch 49)
    if netG is None:
        try:
            # On tente le modèle 128 "hardcodé"
            model = GeneratorLegacy128(Z_DIM, NUM_CLASSES).to(DEVICE)
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            print("✅ Succès : Architecture Legacy 128px détectée.")
            netG = model
        except:
            pass

    # ESSAI 3 : Architecture Legacy 64
    if netG is None:
        try:
            model = GeneratorLegacy(Z_DIM, NUM_CLASSES).to(DEVICE)
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            print("✅ Succès : Architecture Legacy 64px détectée.")
            netG = model
        except:
            print("❌ ÉCHEC TOTAL : Aucune architecture ne correspond à ce fichier.")
            return

    # --- SUITE DU SCRIPT ---
    netG.eval()

    z = torch.randn(1, Z_DIM, 1, 1).to(DEVICE)
    labels = torch.tensor([specific_attrs]).float().to(DEVICE)

    print(f"🎨 Génération en cours...")
    with torch.no_grad():
        fake_image = netG(z, labels).detach().cpu()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    attr_str = "_".join([f"{name[:4]}_{val}" for name, val in zip(ATTR_NAMES, specific_attrs)])
    save_filename = f"gen_{attr_str}.png"
    save_path = os.path.join(RESULTS_DIR, save_filename)

    vutils.save_image(fake_image, save_path, normalize=True, value_range=(-1, 1))
    print(f"✅ Image sauvegardée : {save_path}")