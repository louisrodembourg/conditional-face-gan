import argparse
from train import train
from generate import generate_inference, generate_all_combinations
from config import ATTR_NAMES

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Projet AC-WGAN-GP - Deep Learning")
    
    # Choix du mode : train ou generate
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate', 'generate_grid'], 
                        help="Mode d'exécution : 'train', 'generate' (single), 'generate_grid' (all combos)")
    
    # Arguments spécifiques pour le mode 'generate'
    parser.add_argument('--weights', type=str, default=None, 
                        help="Chemin vers le fichier .pth du générateur (Obligatoire en mode generate)")
    
    parser.add_argument('--attr', type=int, nargs='+', default=None,
                        help=f"Liste des attributs (0 ou 1) dans l'ordre : {ATTR_NAMES}. Exemple: --attr 1 0 1 0 1")

    args = parser.parse_args()

    # --- LOGIQUE ---
    if args.mode == 'train':
        print("====================================")
        print("      DÉMARRAGE ENTRAÎNEMENT        ")
        print("====================================")
        train()
        
    elif args.mode == 'generate':
        print("====================================")
        print("        MODE GÉNÉRATION             ")
        print("====================================")
        
        if args.weights is None:
            print("❌ Erreur : Vous devez spécifier un fichier de poids avec --weights")
            print("Exemple : python main.py --mode generate --weights checkpoints/run1/netG.pth --attr 1 1 1 0 1")
        else:
            generate_inference(args.weights, args.attr)

    elif args.mode == 'generate_grid':
        print("====================================")
        print("     MODE GÉNÉRATION GRILLE         ")
        print("====================================")
        
        if args.weights is None:
            print("❌ Erreur : Vous devez spécifier un fichier de poids avec --weights")
        else:
            generate_all_combinations(args.weights, num_grids=5)