import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "data"

DEFAULT_TRI_AREA = 0.05      
GAUSS_TRI_ORDER = 5          
GAUSS_LINE_ORDER = 5         
BOUNDARY_DENSITY_PTS = 100   

ADAM_LR = 1e-4            
ADAM_EPOCHS = 500          

LBFGS_EPOCHS = 500          
LBFGS_LR = 1.0              
LBFGS_MAX_ITER = 20         
LBFGS_TOLERANCE_GRAD = 1e-7 
LBFGS_TOLERANCE_CHANGE = 1e-9 

BC_PENALTY = 10.0            

AUTO_BALANCE_ENABLED = True          
AUTO_BALANCE_MOMENTUM = 0.9           
AUTO_BALANCE_MIN_WEIGHT = 0.1         
AUTO_BALANCE_MAX_WEIGHT = 10.0        

NTK_ANALYSIS_EVERY = 100     
NTK_ANALYSIS_POINTS = 256
