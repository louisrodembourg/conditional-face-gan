# Project TODOs

## Data & Labels
- [ ] Augment and add 2 extra labels to training/generation pipeline
- [ ] Audit label definitions to align course exo_16 one-hot usage

## Architectures
- [ ] Compare course generator (exo_16_alt_G) vs project generator; choose backbone
- [ ] Compare course discriminator (exo_16_alt_D) vs project discriminator; choose backbone
- [ ] Switch one-hot conditioning wiring per exo_16 reference
- [ ] Experiment with AC-GAN auxiliary classifier head (see TPs)
- [ ] Try progressive growing: start low resolution then upscale stages
- [ ] Replace batch norm with instance norm or group norm in generator

## Losses
- [ ] Add Wasserstein loss with gradient penalty (WGAN-GP)
- [ ] Review gradient penalty term implementation details (lambda, norm)
- [ ] Add classification loss for conditional/AC-GAN setup

## Training Loop
- [ ] Compare training loops (course vs project) for update order and n_critic
- [ ] Increase batch size (measure GPU/VRAM and stability)
- [ ] Extend training to 50 epochs
- [ ] Implement decaying learning rate schedule during training
- [ ] Benchmark against fixed learning rate (no decay) for control

## Evaluation & Metrics
- [ ] Add FID computation to validate sample quality (per TP reference)
- [ ] Track per-class metrics if using AC-GAN (aux classifier accuracy)

## Experiment Management
- [ ] Add config file for hyperparams (batch size, lr, loss choices)
- [ ] Ensure reproducibility (set seeds, deterministic flags where feasible)
- [ ] Log experiments (loss curves, FID, sample grids) with clear run IDs
- [ ] Save model checkpoints per key milestones (e.g., every 5 epochs)
