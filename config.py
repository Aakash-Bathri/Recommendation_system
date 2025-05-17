import os

# Dataset configuration
DATA_ROOT = 'data/Reddit'
DATASET_NAME = 'Reddit'

# Data split configuration
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NEG_SAMPLING_RATIO = 1.0

# DeepWalk configuration
DEEPWALK_WINDOW = 5
DEEPWALK_WALK_LENGTH = 20
DEEPWALK_NUM_WALKS = 10
DEEPWALK_VECTOR_SIZE = 128
DEEPWALK_BATCH_SIZE = 5000  # Number of nodes to process at once

# Model configuration
HIDDEN_CHANNELS = 512
OUT_CHANNELS = 256
DECODER_HIDDEN_LAYERS = [512, 256]

# Training configuration
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 512
MAX_EPOCHS = 30
PATIENCE = 10
LR_FACTOR = 0.5
LR_PATIENCE = 5

# Output configuration
MODEL_SAVE_PATH = 'checkpoints'
PLOT_SAVE_PATH = 'plots'

# Create necessary directories
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(PLOT_SAVE_PATH, exist_ok=True)

# Device configuration
import torch
DEVICE = torch.device('cpu')
CUDA_VERSION = 'cpu' if not torch.cuda.is_available() else f'cu{torch.version.cuda.replace(".", "")}'