"""
Main Entry Point
================

This script serves as the primary entry point for the project.
It allows users to switch between training mode and generation/inference mode
via command-line arguments.

Usage:
    python src/main.py --mode train
    python src/main.py --mode generate --weights checkpoints/... --attr 1 0 1 0 1
"""

import argparse
import sys
import os

# Add the current directory to path to allow imports if run directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import train
from generate import generate_inference, generate_all_combinations
from config import ATTR_NAMES

def main():
    parser = argparse.ArgumentParser(description="AC-WGAN-GP Project - Deep Learning")
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate', 'generate_grid'], 
                        help="Execution mode: 'train', 'generate' (single inference), 'generate_grid' (all combinations)")
    
    # Arguments for generation
    parser.add_argument('--weights', type=str, default=None, 
                        help="Path to the Generator .pth weights file (Required for generation modes)")
    
    parser.add_argument('--attr', type=int, nargs='+', default=None,
                        help=f"List of attributes (0 or 1) in order: {ATTR_NAMES}. Example: --attr 1 0 1 0 1")

    args = parser.parse_args()

    # --- LOGIC ---
    if args.mode == 'train':
        print("====================================")
        print("      STARTING TRAINING             ")
        print("====================================")
        train()
        
    elif args.mode == 'generate':
        print("====================================")
        print("        GENERATION MODE             ")
        print("====================================")
        
        if args.weights is None:
            print("❌ Error: You must specify a weights file with --weights")
            print("Example: python src/main.py --mode generate --weights checkpoints/run1/netG.pth --attr 1 1 1 0 1")
        else:
            generate_inference(args.weights, args.attr)

    elif args.mode == 'generate_grid':
        print("====================================")
        print("     GRID GENERATION MODE           ")
        print("====================================")
        
        if args.weights is None:
            print("❌ Error: You must specify a weights file with --weights")
        else:
            generate_all_combinations(args.weights, num_grids=5)

if __name__ == '__main__':
    main()