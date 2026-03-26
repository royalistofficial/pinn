import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "data"

DEFAULT_TRI_AREA = 0.05
GAUSS_TRI_ORDER = 5
GAUSS_LINE_ORDER = 3
BOUNDARY_DENSITY_PTS = 100

PINN_ARCH = {
    "hidden": 8,
    "n_layers": 2,
    "degree": 3,
    "n_fourier": 2,
    "freq_min": 1.0,
    "freq_max": 2.0,
    "trainable_freqs": False,
}

PINN_MLP = {
    "hidden": 24,
    "n_blocks": 3,
}

MLP = False

LOSS_TYPE    = "mse"
DEFAULT_LR   = 1e-2
TRAIN_EPOCHS = 2000      
BC_PENALTY   = 10.0    

USE_CORNER = False

NTK_ANALYSIS_EVERY  = 200  
NTK_ANALYSIS_POINTS = 256

NTK_NODE_ORDER = "xy" 

FEM_TRI_AREA     = 0.005
FEM_REFINE_LEVELS = [0.1, 0.05, 0.01, 0.005]