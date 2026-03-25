import torch

# Базовые настройки
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "data"
CONSTANTS_DIR = "data_const"

# Настройки генерации сетки и квадратур
DEFAULT_TRI_AREA = 0.01
GAUSS_TRI_ORDER = 5
GAUSS_LINE_ORDER = 7
BOUNDARY_DENSITY_PTS = 100

# Настройки архитектуры PINN
CN_ALPHA = 0.01
CN_EXPANSION = 1

PINN_ARCH = {
    "hidden": 4,
    "n_blocks": 1,
    "n_fourier": 2,
    "n_scales": 2,
    "freq_min": 0.1,
    "freq_max": 2.0,
    "expansion": CN_EXPANSION,
    "shortcut": True,
    "alpha": CN_ALPHA,
    "trainable_freqs": True,
}

# --- Настройки обучения обычного PINN ---

# Флаг выбора функции потерь: 
# "mse" - стандартная среднеквадратичная ошибка (на основе точек)
# "norm" - использование L2-норм с интеграцией по Гауссу
LOSS_TYPE = "mse" 

# Скорость обучения и эпохи
DEFAULT_LR = 1e-2
TRAIN_EPOCHS = 5000 
BC_PENALTY = 200.0