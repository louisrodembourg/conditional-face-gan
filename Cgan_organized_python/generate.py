import torch
import os
import itertools
import re
import torchvision.utils as vutils
from PIL import Image, ImageDraw, ImageFont
from config import *
from models import Generator

def get_model_info(weights_path):
    """
    Extrait le nom du run et l'époque/info du fichier de poids.
    """
    weights_path = os.path.abspath(weights_path)
    dirname = os.path.dirname(weights_path)
    run_name = os.path.basename(dirname)
    
    filename = os.path.basename(weights_path)
    if "best_fid" in filename:
        epoch = "Best FID"
    else:
        # Chercher 'ep' suivi de chiffres
        match = re.search(r'ep(\d+)', filename)
        if match:
            epoch = f"Epoch {match.group(1)}"
        else:
            epoch = filename
            
    return run_name, epoch

def add_banner_to_image(image_path, run_name, epoch, attr_text):
    """
    Ajoute un bandeau blanc avec les infos en haut de l'image.
    Redimensionne l'image si elle est trop petite (< 512px de large) pour une meilleure visibilité.
    """
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"⚠️ Impossible d'ouvrir l'image pour ajouter le bandeau : {e}")
        return

    # --- UPSCALING ---
    # Si l'image est petite (ex: 128px), on l'agrandit pour qu'elle soit visible
    original_w, original_h = img.size
    target_min_width = 512
    
    if original_w < target_min_width:
        scale_factor = target_min_width / original_w
        new_w = int(original_w * scale_factor)
        new_h = int(original_h * scale_factor)
        # NEAREST pour garder le côté "pixel art" réel du modèle, ou BICUBIC pour lisser
        # On choisit NEAREST pour ne pas tricher sur la qualité, ou LANCZOS pour faire propre.
        # Ici on prend LANCZOS (haute qualité) pour que ce soit "beau".
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        print(f"🔍 Upscaling image: {original_w}x{original_h} -> {new_w}x{new_h}")

    width, height = img.size
    
    # Ajustement de la taille de police selon la largeur de l'image
    if width < 200:
        font_size = 9 
    else:
        font_size = max(14, int(width / 40)) # Police un peu plus grosse car image upscalée
    
    # Chargement de la police
    font = None
    try:
        # Essai Mac/Windows
        font = ImageFont.truetype("Arial.ttf", font_size)
    except:
        try:
            # Essai Linux
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            pass
            
    if font is None:
        # Fallback
        font = ImageFont.load_default()

    # Surface temporaire pour calculer les tailles
    draw_temp = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    
    def get_text_width(text, font):
        if hasattr(draw_temp, 'textlength'):
            return draw_temp.textlength(text, font=font)
        else:
            return draw_temp.textsize(text, font=font)[0]

    max_width = width - 10 # Marge de 5px
    lines_to_draw = []
    
    # Infos brutes
    raw_lines = [f"Run: {run_name}", f"Model: {epoch}", f"Attrs: {attr_text}"]
    
    for raw_line in raw_lines:
        # Si la ligne est trop longue, on vérifie
        current_width = get_text_width(raw_line, font)
        if current_width <= max_width:
            lines_to_draw.append(raw_line)
        else:
            # Découpage caractère par caractère si ça dépasse
            current_line = ""
            for char in raw_line:
                test_line = current_line + char
                if get_text_width(test_line, font) < max_width:
                    current_line = test_line
                else:
                    lines_to_draw.append(current_line)
                    current_line = char
            if current_line:
                lines_to_draw.append(current_line)

    # Espacement vertical
    # Pour load_default(), la taille est fixe, donc line_height doit être suffisant
    if font_size <= 10:
        line_height = 11
    else:
        line_height = font_size + 3

    banner_height = 10 + (len(lines_to_draw) * line_height) + 5
    
    # Création nouvelle image avec bandeau
    new_img = Image.new('RGB', (width, height + banner_height), color='white')
    
    # Collage de l'image originale
    new_img.paste(img, (0, banner_height))
    
    # Dessin du texte
    draw = ImageDraw.Draw(new_img)
    y_text = 8
    for line in lines_to_draw:
        draw.text((5, y_text), line, fill='black', font=font)
        y_text += line_height
        
    new_img.save(image_path)
    print(f"✅ Bandeau ajouté : {run_name} | {epoch}")

def save_annotated_grid(tensor_batch, combinations, save_path):
    """
    Crée une grille où chaque image a ses attributs écrits au-dessus.
    """
    n_imgs = len(tensor_batch)
    cols = 8
    rows = (n_imgs + cols - 1) // cols
    
    # Configuration des dimensions (Upscaling pour lisibilité)
    cell_img_size = 256 
    header_height = 40
    cell_total_height = cell_img_size + header_height
    cell_total_width = cell_img_size
    
    # Création de l'image de la grille entière
    grid_img = Image.new('RGB', (cols * cell_total_width, rows * cell_total_height), color='white')
    draw = ImageDraw.Draw(grid_img)
    
    # Police
    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    # Noms courts pour l'affichage
    short_names = {
            'Male': 'Male',
            'Blond_Hair': 'Blond',
            'Smiling': 'Smile',
            'Eyeglasses': 'Glasses',
            'Young': 'Young'
    }

    for idx in range(n_imgs):
        # 1. Traitement de l'image (Tensor -> PIL)
        t = tensor_batch[idx] # (3, 128, 128)
        # Denormalize (-1, 1) -> (0, 1)
        t = (t + 1) / 2.0
        t = t.clamp(0, 1)
        
        # Conversion numpy
        ndarr = t.mul(255).byte().permute(1, 2, 0).numpy()
        img = Image.fromarray(ndarr)
        
        # Upscale (Nearest pour garder le grain ou Lanczos pour lisser)
        img = img.resize((cell_img_size, cell_img_size), Image.Resampling.LANCZOS)
        
        # 2. Récupération des attributs actifs
        combo = combinations[idx]
        active = [short_names.get(ATTR_NAMES[i], ATTR_NAMES[i]) for i, val in enumerate(combo) if val == 1]
        
        if not active:
            label = "None"
        elif len(active) > 3:
            label = ", ".join(active)
        else:
            label = ", ".join(active)
            
        # 3. Collage dans la grille
        r = idx // cols
        c = idx % cols
        x = c * cell_total_width
        y = r * cell_total_height
        
        # Centrage du texte
        try:
            # draw.textbbox is available in newer Pillow versions
            # If not available, we can skip centering or use heuristic
            if hasattr(draw, 'textbbox'):
                bbox = draw.textbbox((0, 0), label, font=font)
                text_w = bbox[2] - bbox[0]
                text_x = x + (cell_total_width - text_w) / 2
            else:
                 # heuristic
                 text_x = x + 10
        except:
             text_x = x + 10

        draw.text((text_x, y + 10), label, fill='black', font=font)

        # Collage image
        grid_img.paste(img, (x, y + header_height))
        
    grid_img.save(save_path)

def generate_all_combinations(weights_path, num_grids=5):
    if not os.path.exists(weights_path):
        print(f"❌ Erreur: Le fichier {weights_path} n'existe pas.")
        return

    run_name, epoch_info = get_model_info(weights_path)
    print(f"⏳ Chargement du modèle : {run_name} ({epoch_info})...")
    
    try:
        netG = Generator(Z_DIM, NUM_CLASSES, img_size=IMAGE_SIZE).to(DEVICE)
        netG.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        print("✅ Modèle chargé avec succès.")
    except Exception as e:
        print(f"❌ Erreur lors du chargement des poids : {e}")
        return

    netG.eval()
    
    combinations = list(itertools.product([0, 1], repeat=NUM_CLASSES))
    num_images = len(combinations)
    labels = torch.tensor(combinations).float().to(DEVICE)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"🎨 Génération de {num_grids} grilles différentes...")

    for i in range(num_grids):
        # On génère une nouvelle "personne" (vecteur Z) pour chaque grille
        fixed_z = torch.randn(1, Z_DIM, 1, 1).to(DEVICE).repeat(num_images, 1, 1, 1)
        
        with torch.no_grad():
            fake_images = netG(fixed_z, labels).detach().cpu()
        
        save_path = os.path.join(RESULTS_DIR, f"all_combinations_grid_{i+1}.png")
        
        # Sauvegarde AVEC annotations par image
        save_annotated_grid(fake_images, combinations, save_path)
        
        # Ajout du bandeau général
        attr_text = "Grid of 32 Combinations"
        add_banner_to_image(save_path, run_name, epoch_info, attr_text)
        
        print(f"   -> Grille {i+1} sauvegardée avec toutes les annotations : {save_path}")

    print(f"✅ Terminé.")

def generate_inference(weights_path, specific_attrs=None):
    
    if not os.path.exists(weights_path):
        print(f"❌ Erreur: Le fichier {weights_path} n'existe pas.")
        return

    run_name, epoch_info = get_model_info(weights_path)


    if specific_attrs is None:
        specific_attrs = (torch.rand(NUM_CLASSES) > 0.5).int().tolist()
    
    print(f"⏳ Chargement du modèle : {run_name} ({epoch_info})...")
    
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

    # Conversion des attributs en texte lisible pour le bandeau
    active_attr_names = [name for name, val in zip(ATTR_NAMES, specific_attrs) if val == 1]
    if not active_attr_names:
        attr_text_banner = "None"
    else:
        attr_text_banner = ", ".join(active_attr_names)

    print(f"🎨 Génération en cours pour : {attr_text_banner}")
    with torch.no_grad():
        fake_image = netG(z, labels).detach().cpu()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    attr_str = "_".join([f"{name[:4]}_{val}" for name, val in zip(ATTR_NAMES, specific_attrs)])
    save_filename = f"gen_{attr_str}.png"
    save_path = os.path.join(RESULTS_DIR, save_filename)

    vutils.save_image(fake_image, save_path, normalize=True, value_range=(-1, 1))
    
    # Ajout du bandeau
    add_banner_to_image(save_path, run_name, epoch_info, attr_text_banner)
    
    print(f"✅ Image sauvegardée : {save_path}")
