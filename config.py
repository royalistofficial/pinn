import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "data"

DEFAULT_TRI_AREA = 0.05      
GAUSS_TRI_ORDER = 5          
GAUSS_LINE_ORDER = 5         
BOUNDARY_DENSITY_PTS = 100   

DEFAULT_LR = 1e-3            
TRAIN_EPOCHS = 400          
BC_PENALTY = 10.0            

AUTO_BALANCE_ENABLED = False          
AUTO_BALANCE_METHOD = "ntk_trace"     
AUTO_BALANCE_UPDATE_EVERY = 100       
AUTO_BALANCE_MOMENTUM = 0.9           
AUTO_BALANCE_MIN_WEIGHT = 0.1         
AUTO_BALANCE_MAX_WEIGHT = 10.0        

NTK_ANALYSIS_EVERY = 200     
NTK_ANALYSIS_POINTS = 256    
NTK_NODE_ORDER = "xy"
