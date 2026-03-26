import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "data"

DEFAULT_TRI_AREA = 0.05
GAUSS_TRI_ORDER = 5
GAUSS_LINE_ORDER = 7
BOUNDARY_DENSITY_PTS = 100

PINN_ARCH = {
    "hidden": 32,           
    "n_blocks": 1,          
    "n_fourier": 4,         
    "n_scales": 1,          
    "freq_min": 0.1,
    "freq_max": 3.0,
    "expansion": 1,
    "shortcut": True,
    "activation": "silu",   
    "use_ntk_param": True,  
    "trainable_freqs": True,
}

PINN_MLP = {
    "hidden": 32,           
    "n_blocks": 2,
}

MLP = False

LOSS_TYPE = "mse"           
DEFAULT_LR = 1e-2
TRAIN_EPOCHS = 1000
BC_PENALTY = 200.0

USE_NTK_PRECOND = False     
NTK_PRECOND_EVERY = 50      
NTK_PRECOND_POINTS = 64    
NTK_PRECOND_REG = 1e-4      

USE_ADAPTIVE_FREQ = True    
ADAPTIVE_FREQ_POINTS = 256  

NTK_ANALYSIS_EVERY = 100    
NTK_ANALYSIS_POINTS = 256    

FEM_TRI_AREA = 0.005
FEM_REFINE_LEVELS = [0.1, 0.05, 0.01, 0.005]
