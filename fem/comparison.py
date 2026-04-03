import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as mcolors

from config import DEVICE
from geometry.domains import make_domain
from geometry.mesher import Mesher
from problems.solutions import SOLUTIONS
from networks.pinn import PINN
from networks.configs import get_config
from fem.solver import FEMSolver
from functionals.operators import gradient  

def compare_pinn_and_fem(domain_name="square", solution_name="steep_peak", pinn_weights_path=None, net_config=None):
    print(f"--- Детальное сравнение МКЭ и PINN ---")
    print(f"Домен: {domain_name}, Задача: {solution_name}")
    
    # 1. Подготовка домена и точного решения
    domain = make_domain(domain_name)
    solution = SOLUTIONS[solution_name]()
    
    # 2. Генерация общей сетки для оценки
    mesher = Mesher(max_area=0.01) 
    mesh = mesher.build(domain)
    points = mesh["points"]
    triangles = mesh["triangles"]
    print(f"Узлов: {len(points)}, Элементов: {len(triangles)}")

    # 3. Решение через МКЭ (FEM)
    def f_func(x, y):
        xy_tensor = torch.tensor([[x, y]], dtype=torch.float32)
        return solution.rhs(xy_tensor).item()

    def bc_func(x, y):
        xy_tensor = torch.tensor([[x, y]], dtype=torch.float32)
        val = solution.eval(xy_tensor).item()
        return ('dirichlet', val)

    print("Решение через МКЭ...")
    fem_solver = FEMSolver(mesh, f_func, bc_func)
    u_fem = fem_solver.solve()

    # --- Вычисление градиентов МКЭ (усреднение по узлам) ---
    grad_u_fem = np.zeros((len(points), 2))
    node_counts = np.zeros(len(points))

    for e in triangles:
        coords = points[e]
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]
        
        area = 0.5 * abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
        if area < 1e-14:
            continue
            
        # Матрица производных функций формы
        b = np.array([y2 - y3, y3 - y1, y1 - y2])
        c = np.array([x3 - x2, x1 - x3, x2 - x1])
        B = np.vstack([b, c]) / (2 * area)
        
        # Градиент внутри элемента (постоянен)
        grad_e = B @ u_fem[e]
        
        # Добавляем вклад элемента в узлы (для гладкой визуализации)
        for i in range(3):
            grad_u_fem[e[i]] += grad_e
            node_counts[e[i]] += 1
            
    # Усредняем градиенты
    grad_u_fem /= np.clip(node_counts[:, None], 1, None)

    # 4. Решение через PINN
    print("Инициализация PINN...")
    if net_config is None:
        config = get_config("mlp", in_dim=2, out_dim=1, hidden_dim=64, n_layers=4)
    else:
        config = net_config
        
    pinn = PINN(config).to(DEVICE)
    
    if pinn_weights_path and os.path.exists(pinn_weights_path):
        pinn.load_state_dict(torch.load(pinn_weights_path, map_location=DEVICE))
        print(f"Веса PINN загружены из {pinn_weights_path}")
    else:
        print("ВНИМАНИЕ: Используется необученный PINN. Графики покажут ошибку до обучения.")

    pinn.eval()
    
    # 5. Инференс PINN и аналитического решения
    xy_tensor = torch.tensor(points, dtype=torch.float32, device=DEVICE, requires_grad=True)
    
    # Считаем u_pinn и градиенты PINN
    u_pinn_t = pinn(xy_tensor)
    grad_u_pinn_t = gradient(u_pinn_t, xy_tensor, create_graph=False)
    
    u_pinn = u_pinn_t.detach().cpu().numpy().flatten()
    grad_u_pinn = grad_u_pinn_t.detach().cpu().numpy()
    
    with torch.no_grad():
        u_exact = solution.eval(xy_tensor).cpu().numpy().flatten()
        grad_u_exact = solution.grad_vector(xy_tensor).cpu().numpy()

    # --- Вычисление абсолютных и энергетических ошибок ---
    # Абсолютная ошибка |u - u_pred|
    err_fem = np.abs(u_exact - u_fem)
    err_pinn = np.abs(u_exact - u_pinn)
    
    # Энергетическая ошибка: L2-норма разности градиентов |grad(u) - grad(u_pred)|
    energy_err_fem = np.sqrt(np.sum((grad_u_fem - grad_u_exact)**2, axis=1))
    energy_err_pinn = np.sqrt(np.sum((grad_u_pinn - grad_u_exact)**2, axis=1))

    # 6. Визуализация (Дашборд 2x4)
    tri_obj = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.patch.set_facecolor('white')
    
    def plot_field(ax, u, title, cmap='viridis', is_error=False):
        # Если это график ошибки, ограничим минимальное значение нулем
        vmin = 0.0 if is_error else np.min(u)
        vmax = np.max(u)
        if vmax <= vmin and is_error:
            vmax = vmin + 1e-8
            
        tc = ax.tripcolor(tri_obj, u, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.axis('off') # Убираем оси для чистоты графиков
        plt.colorbar(tc, ax=ax, fraction=0.046, pad=0.04)

    # Верхний ряд (Аналитика и МКЭ)
    plot_field(axes[0, 0], u_exact, "Точное решение", cmap='viridis')
    plot_field(axes[0, 1], u_fem, "Решение МКЭ", cmap='viridis')
    plot_field(axes[0, 2], err_fem, "Ошибка МКЭ\n$|u - u_{fem}|$", cmap='turbo', is_error=True)
    plot_field(axes[0, 3], energy_err_fem, "Энергетическая ошибка МКЭ\n$|\\nabla u - \\nabla u_{fem}|$", cmap='turbo', is_error=True)
    
    # Нижний ряд (Сетка и PINN)
    axes[1, 0].triplot(tri_obj, color='steelblue', linewidth=0.4, alpha=0.7)
    axes[1, 0].set_aspect('equal')
    axes[1, 0].set_title(f"Сетка ({len(triangles)} треуг.)", fontsize=12, fontweight='bold', pad=10)
    axes[1, 0].axis('off')
    
    plot_field(axes[1, 1], u_pinn, "Решение PINN", cmap='viridis')
    plot_field(axes[1, 2], err_pinn, "Ошибка PINN\n$|u - u_{pinn}|$", cmap='turbo', is_error=True)
    plot_field(axes[1, 3], energy_err_pinn, "Энергетическая ошибка PINN\n$|\\nabla u - \\nabla u_{pinn}|$", cmap='turbo', is_error=True)

    plt.tight_layout()
    os.makedirs("data", exist_ok=True)
    save_path = f"data/comparison_{domain_name}_{solution_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"График сравнения сохранен в {save_path}")
    
    plt.close(fig) 
    return save_path