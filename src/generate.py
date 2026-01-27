"""
Generation Module
=================

This module handles inference using trained Generator models.
It supports:
1. Single image generation with specific attributes.
2. Grid generation showing all attribute combinations for a fixed latent vector.
"""

import os
import re
import itertools
from typing import List, Optional, Tuple

import torch
import torchvision.utils as vutils
from PIL import Image, ImageDraw, ImageFont

import config
from models import Generator

def get_model_info(weights_path: str) -> Tuple[str, str]:
    """
    Extracts run name and epoch info from the weights filename/path.
    """
    weights_path = os.path.abspath(weights_path)
    dirname = os.path.dirname(weights_path)
    run_name = os.path.basename(dirname)
    
    filename = os.path.basename(weights_path)
    if "best_fid" in filename:
        epoch = "Best FID"
    else:
        # Search for 'ep' followed by digits
        match = re.search(r'ep(\d+)', filename)
        if match:
            epoch = f"Epoch {match.group(1)}"
        else:
            epoch = filename
            
    return run_name, epoch

def add_banner_to_image(image_path: str, run_name: str, epoch: str, attr_text: str):
    """
    Adds a white banner with metadata to the top of the generated image.
    Upscales small images for better visibility.
    """
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"⚠️ Cannot open image to add banner: {e}")
        return

    # --- UPSCALING ---
    # Upscale if image is small (e.g., 128px) for visibility
    original_w, original_h = img.size
    target_min_width = 512
    
    if original_w < target_min_width:
        scale_factor = target_min_width / original_w
        new_w = int(original_w * scale_factor)
        new_h = int(original_h * scale_factor)
        # Use LANCZOS for high quality upscaling
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        print(f"🔍 Upscaling image: {original_w}x{original_h} -> {new_w}x{new_h}")

    width, height = img.size
    
    # Adjust font size based on width
    if width < 200:
        font_size = 9 
    else:
        font_size = max(14, int(width / 40))
    
    # Load Font
    font = None
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except:
        try:
            # Linux fallback
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            pass
            
    if font is None:
        font = ImageFont.load_default()

    # Temporary Draw to calculate text size
    draw_temp = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    
    def get_text_width(text, font):
        if hasattr(draw_temp, 'textlength'):
            return draw_temp.textlength(text, font=font)
        else:
            return draw_temp.textsize(text, font=font)[0]

    max_width = width - 10 # 5px margin
    lines_to_draw = []
    
    # Raw Info
    raw_lines = [f"Run: {run_name}", f"Model: {epoch}", f"Attrs: {attr_text}"]
    
    for raw_line in raw_lines:
        current_width = get_text_width(raw_line, font)
        if current_width <= max_width:
            lines_to_draw.append(raw_line)
        else:
            # Wrap text char by char if too long
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

    # Vertical spacing
    if font_size <= 10:
        line_height = 11
    else:
        line_height = font_size + 3

    banner_height = 10 + (len(lines_to_draw) * line_height) + 5
    
    # Create new image with banner
    new_img = Image.new('RGB', (width, height + banner_height), color='white')
    new_img.paste(img, (0, banner_height))
    
    # Draw Text
    draw = ImageDraw.Draw(new_img)
    y_text = 8
    for line in lines_to_draw:
        draw.text((5, y_text), line, fill='black', font=font)
        y_text += line_height
        
    new_img.save(image_path)
    print(f"✅ Banner added: {run_name} | {epoch}")

def save_annotated_grid(tensor_batch: torch.Tensor, combinations: List[Tuple[int, ...]], save_path: str):
    """
    Creates a grid where each image has its attributes written above it.
    """
    n_imgs = len(tensor_batch)
    cols = 8
    rows = (n_imgs + cols - 1) // cols
    
    # Dimensions (Upscaling for readability)
    cell_img_size = 256 
    header_height = 40
    cell_total_height = cell_img_size + header_height
    cell_total_width = cell_img_size
    
    # Full Grid Image
    grid_img = Image.new('RGB', (cols * cell_total_width, rows * cell_total_height), color='white')
    draw = ImageDraw.Draw(grid_img)
    
    # Font
    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    # Short naming for display
    short_names = {
            'Male': 'Male',
            'Blond_Hair': 'Blond',
            'Smiling': 'Smile',
            'Eyeglasses': 'Glasses',
            'Young': 'Young'
    }

    for idx in range(n_imgs):
        # 1. Process Image (Tensor -> PIL)
        t = tensor_batch[idx] # (3, 128, 128)
        # Denormalize (-1, 1) -> (0, 1)
        t = (t + 1) / 2.0
        t = t.clamp(0, 1)
        
        # Convert to numpy
        ndarr = t.mul(255).byte().permute(1, 2, 0).numpy()
        img = Image.fromarray(ndarr)
        
        # Upscale
        img = img.resize((cell_img_size, cell_img_size), Image.Resampling.LANCZOS)
        
        # 2. Get Active Attributes
        combo = combinations[idx]
        active = [short_names.get(config.ATTR_NAMES[i], config.ATTR_NAMES[i]) for i, val in enumerate(combo) if val == 1]
        
        if not active:
            label = "None"
        elif len(active) > 3:
            label = ", ".join(active[:3]) + "..."
        else:
            label = ", ".join(active)
            
        # 3. Paste into Grid
        r = idx // cols
        c = idx % cols
        x = c * cell_total_width
        y = r * cell_total_height
        
        # Center Text
        try:
            if hasattr(draw, 'textbbox'):
                bbox = draw.textbbox((0, 0), label, font=font)
                text_w = bbox[2] - bbox[0]
                text_x = x + (cell_total_width - text_w) / 2
            else:
                 text_x = x + 10
        except:
             text_x = x + 10

        draw.text((text_x, y + 10), label, fill='black', font=font)
        grid_img.paste(img, (x, y + header_height))
        
    grid_img.save(save_path)

def generate_all_combinations(weights_path: str, num_grids: int = 5):
    """
    Generates grids of images for all possible attribute combinations.
    """
    if not os.path.exists(weights_path):
        print(f"❌ Error: File {weights_path} does not exist.")
        return

    run_name, epoch_info = get_model_info(weights_path)
    print(f"⏳ Loading model: {run_name} ({epoch_info})...")
    
    try:
        netG = Generator(config.Z_DIM, config.NUM_CLASSES, img_size=config.IMAGE_SIZE).to(config.DEVICE)
        netG.load_state_dict(torch.load(weights_path, map_location=config.DEVICE))
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    netG.eval()
    
    combinations = list(itertools.product([0, 1], repeat=config.NUM_CLASSES))
    num_images = len(combinations)
    labels = torch.tensor(combinations).float().to(config.DEVICE)
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    print(f"🎨 Generating {num_grids} different grids...")

    for i in range(num_grids):
        # Generate new random "person" (latent Z) for each grid
        fixed_z = torch.randn(1, config.Z_DIM, 1, 1).to(config.DEVICE).repeat(num_images, 1, 1, 1)
        
        with torch.no_grad():
            fake_images = netG(fixed_z, labels).detach().cpu()
        
        save_path = os.path.join(config.RESULTS_DIR, f"all_combinations_grid_{i+1}.png")
        
        # Save WITH annotations 
        save_annotated_grid(fake_images, combinations, save_path)
        
        # Add banner
        attr_text = "Grid of 32 Combinations"
        add_banner_to_image(save_path, run_name, epoch_info, attr_text)
        
        print(f"   -> Grid {i+1} saved with annotations: {save_path}")

    print(f"✅ Finished.")

def generate_inference(weights_path: str, specific_attrs: Optional[List[int]] = None):
    """
    Generates a single image with specific attributes.
    """
    if not os.path.exists(weights_path):
        print(f"❌ Error: File {weights_path} does not exist.")
        return

    run_name, epoch_info = get_model_info(weights_path)

    if specific_attrs is None:
        # Default random (sanity check)
        specific_attrs = (torch.rand(config.NUM_CLASSES) > 0.5).int().tolist()
    
    print(f"⏳ Loading model: {run_name} ({epoch_info})...")
    
    try:
        netG = Generator(config.Z_DIM, config.NUM_CLASSES, img_size=config.IMAGE_SIZE).to(config.DEVICE)
        netG.load_state_dict(torch.load(weights_path, map_location=config.DEVICE))
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    netG.eval()

    z = torch.randn(1, config.Z_DIM, 1, 1).to(config.DEVICE)
    labels = torch.tensor([specific_attrs]).float().to(config.DEVICE)
    
    with torch.no_grad():
        fake_img = netG(z, labels).detach().cpu()
    
    # Save Image
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(config.RESULTS_DIR, f"generated_img.png")
    
    vutils.save_image(fake_img, save_path, normalize=True, value_range=(-1, 1))
    
    # Add Banner
    # Construct textual representation of attributes
    active_attr_names = [config.ATTR_NAMES[i] for i, val in enumerate(specific_attrs) if val == 1]
    attr_text = ", ".join(active_attr_names) if active_attr_names else "None"
    
    add_banner_to_image(save_path, run_name, epoch_info, attr_text)
    print(f"✅ Image saved to: {save_path}")


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
