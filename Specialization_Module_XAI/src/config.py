# config.py — Global configuration for reproducibility
import torch
import numpy as np
import random

SEED = 42
DEPTHS = [2, 3, 5]
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = './data'
CHECKPOINT_DIR = './checkpoints'
FIGURES_DIR = './figures'
N_CONCEPT_SAMPLES = 1000  # per concept, per class

# Fix all random seeds for reproducibility
def set_seeds(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
