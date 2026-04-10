import torch
import os

# Define device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Directories and files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "data", "img")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "vqa_fusion_model.pth")

# Hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 40
VAL_FREQUENCY = 2  # Evaluate every `VAL_FREQUENCY` epochs

# Model Architecture Config
MAX_QUESTION_LEN = 14
NUM_ANSWERS = 1000
IMAGE_FEATURE_DIM = 2048
QUESTION_HIDDEN_DIM = 512
FUSION_HIDDEN_DIM = 1024
ATTENTION_NUM_HEADS = 8
DROPOUT_RATE = 0.5
USE_ATTENTION = False  # Set to True to use MultiHeadAttention Fusion
