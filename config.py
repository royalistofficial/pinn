import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "data"

DEFAULT_TRI_AREA = 0.05
GAUSS_TRI_ORDER = 5
GAUSS_LINE_ORDER = 7
BOUNDARY_DENSITY_PTS = 100

PINN_ARCH = {
    "hidden":        128, 
    "n_blocks":      3,    
    "n_fourier":     32,   
    "n_scales":      1,
    "freq_min":      0.1,
    "freq_max":      10.0,  
    "expansion":     1,
    "shortcut":      True,
    "activation":    "silu",
    "use_ntk_param": True,
    "trainable_freqs": True,
    "alpha_init":    1e-3,  
}

PINN_MLP = {
    "hidden": 64,
    "n_blocks": 3,
}

MLP = False

LOSS_TYPE    = "mse"
DEFAULT_LR   = 1e-2
TRAIN_EPOCHS = 1000      
BC_PENALTY   = 50.0    

USE_ADAPTIVE_FREQ    = True
ADAPTIVE_FREQ_POINTS = 512

USE_NTK_WEIGHTS     = True
NTK_WEIGHT_EVERY    = 100  
NTK_WEIGHT_POINTS   = 128   
NTK_WEIGHT_MOMENTUM = 0.9   

USE_CORNER_WEIGHTING = True
CORNER_WEIGHT_BETA   = 0.4

NTK_ANALYSIS_EVERY  = 200  
NTK_ANALYSIS_POINTS = 256

#   "original" / "xy" / "hilbert" / "spectral_K" / "spectral_KL"
NTK_NODE_ORDER = "xy" 

FEM_TRI_AREA     = 0.005
FEM_REFINE_LEVELS = [0.1, 0.05, 0.01, 0.005]