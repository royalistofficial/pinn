import os
import glob
import threading
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk

# Импортируем модули проекта
import config
from geometry.domains import make_domain, DOMAIN_REGISTRY
from problems.solutions import SOLUTIONS
from geometry.mesher import Mesher
from geometry.quadrature import QuadratureBuilder
# Импортируем PRESET_CONFIGS для динамической подгрузки параметров
from networks.configs import NetworkConfig, PRESET_CONFIGS
from training.trainer import Trainer
import training.trainer as trainer_mod
from file_io.logger import FileLogger
from visualization.mesh_plots import plot_mesh

# Настройка темы
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class InteractiveCanvas(tk.Canvas):
    """Кастомный холст с поддержкой зума (колесико) и перемещения (ЛКМ)."""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg="#2b2b2b", highlightthickness=0, **kwargs)
        self.image = None
        self.tk_image = None
        self.scale = 1.0
        self.img_x = 0
        self.img_y = 0
        
        self.bind("<MouseWheel>", self.on_mousewheel)
        self.bind("<Button-4>", self.on_mousewheel)
        self.bind("<Button-5>", self.on_mousewheel)
        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<Configure>", self.on_resize)

    def load_image(self, path):
        if not os.path.exists(path): return
        self.image = Image.open(path)
        self.scale = 1.0
        self.fit_to_screen()
        self.redraw()

    def fit_to_screen(self, event=None):
        if not self.image: return
        w_canv, h_canv = self.winfo_width(), self.winfo_height()
        if w_canv < 10 or h_canv < 10: return
        scale_w, scale_h = w_canv / self.image.width, h_canv / self.image.height
        self.scale = min(scale_w, scale_h) * 0.95
        self.img_x, self.img_y = w_canv / 2, h_canv / 2

    def on_resize(self, event):
        if self.image and self.tk_image is None:
            self.fit_to_screen()
            self.redraw()

    def redraw(self):
        if not self.image: return
        new_w = max(1, int(self.image.width * self.scale))
        new_h = max(1, int(self.image.height * self.scale))
        resized = self.image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)
        self.delete("all")
        self.create_image(self.img_x, self.img_y, image=self.tk_image, anchor="center")

    def on_mousewheel(self, event):
        if not self.image: return
        scale_factor = 0.9 if (event.num == 5 or event.delta < 0) else 1.1
        self.img_x = event.x - (event.x - self.img_x) * scale_factor
        self.img_y = event.y - (event.y - self.img_y) * scale_factor
        self.scale *= scale_factor
        self.redraw()

    def on_press(self, event):
        self.start_x, self.start_y = event.x, event.y

    def on_drag(self, event):
        if not self.image: return
        self.img_x += event.x - self.start_x
        self.img_y += event.y - self.start_y
        self.start_x, self.start_y = event.x, event.y
        self.redraw()


class PINNApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("PINN Advanced Desktop Dashboard")
        self.geometry("1300x800")

        # Словар для хранения динамических переменных сети
        self.dynamic_net_vars = {}

        # --- ЛЕВАЯ ПАНЕЛЬ ---
        self.left_panel = ctk.CTkFrame(self, width=640, corner_radius=5)
        self.left_panel.pack(side="left", fill="y")
        self.left_panel.pack_propagate(False)

        ctk.CTkLabel(self.left_panel, text="Настройки PINN", font=ctk.CTkFont(size=25, weight="bold")).pack(pady=15, padx=20)
        self.scroll_settings = ctk.CTkScrollableFrame(self.left_panel, fg_color="transparent")
        self.scroll_settings.pack(fill="both", expand=True, padx=5, pady=5)

        # --- 1. ЗАДАЧА И ДОМЕН ---
        ctk.CTkLabel(self.scroll_settings, text="1. Физика", font=ctk.CTkFont(weight="bold", size=14), text_color="#3B82F6").pack(anchor="w", padx=10, pady=(10, 5))
        self.domain_var = ctk.StringVar(value=list(DOMAIN_REGISTRY.keys())[0])
        ctk.CTkOptionMenu(self.scroll_settings, values=list(DOMAIN_REGISTRY.keys()), variable=self.domain_var).pack(fill="x", padx=10, pady=5)
        self.solution_var = ctk.StringVar(value="steep_peak")
        ctk.CTkOptionMenu(self.scroll_settings, values=list(SOLUTIONS.keys()), variable=self.solution_var).pack(fill="x", padx=10, pady=5)

        # --- 2. ГЕНЕРАЦИЯ СЕТКИ И КВАДРАТУР (Для Обучения) ---
        ctk.CTkLabel(self.scroll_settings, text="2. Обучающая сетка", font=ctk.CTkFont(weight="bold", size=14), text_color="#3B82F6").pack(anchor="w", padx=10, pady=(15, 5))
        self.mesh_area = self.add_entry(self.scroll_settings, "Макс. площадь (max_area)", config.TRAIN_TRI_AREA)
        self.mesh_density = self.add_entry(self.scroll_settings, "Плотность граничных точек", config.TRAIN_BOUNDARY_DENSITY)
        self.mesh_lloyd = self.add_entry(self.scroll_settings, "Итерации сглаживания Ллойда", 3)
        self.quad_tri = self.add_entry(self.scroll_settings, "Порядок Гаусса (внутри, 1-6)", config.TRAIN_GAUSS_TRI_ORDER)
        self.quad_line = self.add_entry(self.scroll_settings, "Порядок Гаусса (граница)", config.TRAIN_GAUSS_LINE_ORDER)

        # --- 3. АРХИТЕКТУРА СЕТИ (ДИНАМИЧЕСКАЯ) ---
        ctk.CTkLabel(self.scroll_settings, text="3. Нейросеть", font=ctk.CTkFont(weight="bold", size=14), text_color="#3B82F6").pack(anchor="w", padx=10, pady=(15, 5))
        
        self.arch_var = ctk.StringVar(value="mlp")
        # При выборе новой архитектуры вызывается функция on_arch_change
        self.arch_menu = ctk.CTkOptionMenu(self.scroll_settings, values=list(PRESET_CONFIGS.keys()), variable=self.arch_var, command=self.on_arch_change)
        self.arch_menu.pack(fill="x", padx=10, pady=5)
        
        # Контейнер для динамических параметров выбранной сети
        self.net_params_frame = ctk.CTkFrame(self.scroll_settings, fg_color="transparent")
        self.net_params_frame.pack(fill="x", pady=5)
        
        # Инициализируем начальные поля (для MLP)
        self.on_arch_change("mlp")

        # --- 4. ОБУЧЕНИЕ ---
        ctk.CTkLabel(self.scroll_settings, text="4. Обучение", font=ctk.CTkFont(weight="bold", size=14), text_color="#3B82F6").pack(anchor="w", padx=10, pady=(15, 5))
        self.adam_epochs = self.add_entry(self.scroll_settings, "Эпохи Adam", config.ADAM_EPOCHS)
        self.adam_lr = self.add_entry(self.scroll_settings, "Learning Rate Adam", config.ADAM_LR)
        self.lbfgs_epochs = self.add_entry(self.scroll_settings, "Эпохи L-BFGS", config.LBFGS_EPOCHS)
        self.lbfgs_lr = self.add_entry(self.scroll_settings, "Learning Rate L-BFGS", config.LBFGS_LR)
        self.lbfgs_iter = self.add_entry(self.scroll_settings, "L-BFGS max_iter", config.LBFGS_MAX_ITER)
        self.early_stopping = self.add_entry(self.scroll_settings, "Терпение (Early Stopping)", 50)
        
        # --- 5. ФУНКЦИЯ ПОТЕРЬ И NTK ---
        ctk.CTkLabel(self.scroll_settings, text="5. Функция потерь", font=ctk.CTkFont(weight="bold", size=14), text_color="#3B82F6").pack(anchor="w", padx=10, pady=(15, 5))
        self.bc_penalty = self.add_entry(self.scroll_settings, "Штраф за граничные условия", config.BC_PENALTY)
        self.ntk_every = self.add_entry(self.scroll_settings, "Частота анализа NTK (эпохи)", config.NTK_ANALYSIS_EVERY)
        self.autobalance_var = ctk.BooleanVar(value=config.AUTO_BALANCE_ENABLED)
        ctk.CTkCheckBox(self.scroll_settings, text="Адаптивные веса (NTK)", variable=self.autobalance_var).pack(anchor="w", padx=10, pady=10)

        # ПАНЕЛЬ УПРАВЛЕНИЯ ВНИЗУ
        self.control_panel = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.control_panel.pack(fill="x", side="bottom", pady=10)
        self.status_label = ctk.CTkLabel(self.control_panel, text="Статус: Ожидание настройки", text_color="gray")
        self.status_label.pack(pady=(0, 10))
        self.run_btn = ctk.CTkButton(self.control_panel, text="Запустить обучение", height=40, font=ctk.CTkFont(weight="bold", size=14), command=self.start_training_thread, fg_color="#059669", hover_color="#047857")
        self.run_btn.pack(fill="x", padx=20, pady=(0, 10))

        # --- ПРАВАЯ ПАНЕЛЬ (ВКЛАДКИ ДЛЯ КАРТИНОК) ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=5, fg_color="transparent")
        self.main_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        self.tabview = ctk.CTkTabview(self.main_frame)
        self.tabview.pack(fill="both", expand=True)

        self.tabs = ["Сетка", "Метрики", "Поля", "NTK", "Галерея"]
        self.canvases = {}

        for tab_name in self.tabs:
            tab = self.tabview.add(tab_name)
            if tab_name == "Галерея":
                top_frame = ctk.CTkFrame(tab, fg_color="transparent")
                top_frame.pack(fill="x", pady=5)
                self.gallery_var = ctk.StringVar(value="Выберите график...")
                self.gallery_menu = ctk.CTkOptionMenu(top_frame, values=["Нет файлов"], variable=self.gallery_var, command=self.load_gallery_image)
                self.gallery_menu.pack(side="left", fill="x", expand=True, padx=(0, 10))
                ctk.CTkButton(top_frame, text="Обновить список", command=self.refresh_gallery_list, width=120).pack(side="right")
                
            canvas = InteractiveCanvas(tab)
            canvas.pack(fill="both", expand=True)
            self.canvases[tab_name] = canvas

        self.refresh_gallery_list()

    def _clear_data_folder(self):
        """Полностью очищает папку с результатами (data) при запуске приложения."""
        out_dir = config.OUTPUT_DIR
        if os.path.exists(out_dir):
            files = glob.glob(os.path.join(out_dir, "*"))
            for file_path in files:
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Не удалось удалить файл {file_path}: {e}")
        else:
            os.makedirs(out_dir, exist_ok=True)

    def add_entry(self, parent, label, default_val):
        lbl = ctk.CTkLabel(parent, text=label, font=ctk.CTkFont(size=12))
        lbl.pack(anchor="w", padx=10, pady=(5, 0))
        entry = ctk.CTkEntry(parent, height=28)
        entry.insert(0, str(default_val))
        entry.pack(fill="x", padx=10, pady=(0, 5))
        return entry

    def on_arch_change(self, arch_name):
        for widget in self.net_params_frame.winfo_children():
            widget.destroy()
            
        self.dynamic_net_vars.clear()
        
        preset = PRESET_CONFIGS[arch_name]
        
        self.dynamic_net_vars["hidden_dim"] = self.add_entry(self.net_params_frame, "Скрытых нейронов (hidden_dim)", preset.hidden_dim)
        self.dynamic_net_vars["n_layers"] = self.add_entry(self.net_params_frame, "Количество слоев (n_layers)", preset.n_layers)

        if arch_name == "mlp":
            lbl = ctk.CTkLabel(self.net_params_frame, text="Функция активации", font=ctk.CTkFont(size=12))
            lbl.pack(anchor="w", padx=10, pady=(5, 0))
            act_var = ctk.StringVar(value=preset.activation)
            menu = ctk.CTkOptionMenu(self.net_params_frame, values=["tanh", "gelu", "silu", "relu"], variable=act_var)
            menu.pack(fill="x", padx=10, pady=(0, 5))
            self.dynamic_net_vars["activation"] = act_var

        elif arch_name == "siren":
            self.dynamic_net_vars["siren_w0"] = self.add_entry(self.net_params_frame, "siren_w0 (частота)", preset.siren_w0)

        elif arch_name == "fourier":
            self.dynamic_net_vars["fourier_features"] = self.add_entry(self.net_params_frame, "fourier_features", preset.fourier_features)
            self.dynamic_net_vars["fourier_sigma"] = self.add_entry(self.net_params_frame, "fourier_sigma", preset.fourier_sigma)
            
            t_freqs = ctk.BooleanVar(value=preset.trainable_freqs)
            chk = ctk.CTkCheckBox(self.net_params_frame, text="Обучаемые частоты (trainable_freqs)", variable=t_freqs)
            chk.pack(anchor="w", padx=10, pady=10)
            self.dynamic_net_vars["trainable_freqs"] = t_freqs

        elif arch_name == "kan":
            self.dynamic_net_vars["kan_degree"] = self.add_entry(self.net_params_frame, "Степень полинома (kan_degree)", preset.kan_degree)

        elif arch_name == "pi-dbsn":
            self.dynamic_net_vars["dbsn_grid_size"] = self.add_entry(self.net_params_frame, "Размер сетки (dbsn_grid_size)", preset.dbsn_grid_size)
            self.dynamic_net_vars["dbsn_spline_order"] = self.add_entry(self.net_params_frame, "Порядок сплайна (dbsn_spline_order)", preset.dbsn_spline_order)

        elif arch_name == "rbf-kan":
            self.dynamic_net_vars["num_rbf_centers"] = self.add_entry(self.net_params_frame, "Центры RBF (num_rbf_centers)", preset.num_rbf_centers)

        elif arch_name == "wav-kan":
            self.dynamic_net_vars["num_wavelets"] = self.add_entry(self.net_params_frame, "Вейвлеты (num_wavelets)", preset.num_wavelets)

    def refresh_gallery_list(self):
        out_dir = config.OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)
        files = glob.glob(os.path.join(out_dir, "*.png"))
        file_names = sorted([os.path.basename(f) for f in files])
        
        if file_names:
            self.gallery_menu.configure(values=file_names)
            if self.gallery_var.get() in ["Выберите график...", "Нет файлов"]:
                self.gallery_var.set(file_names[-1])
                self.load_gallery_image(file_names[-1])
        else:
            self.gallery_menu.configure(values=["Нет файлов"])
            self.gallery_var.set("Нет файлов")

    def load_gallery_image(self, filename):
        if filename not in ["Нет файлов", "Выберите график..."]:
            path = os.path.join(config.OUTPUT_DIR, filename)
            self.update_image("Галерея", path)

    def start_training_thread(self):
        self.run_btn.configure(state="disabled", text="Обучение в процессе...")
        self.status_label.configure(text="Статус: Обучение...", text_color="orange")
        threading.Thread(target=self.run_pinn, daemon=True).start()

    def run_pinn(self):
        self._clear_data_folder()
        try:
            domain_name = self.domain_var.get()
            solution_name = self.solution_var.get()
            
            # --- СБОРКА ДИНАМИЧЕСКИХ ПАРАМЕТРОВ СЕТИ ---
            net_kwargs = {"architecture": self.arch_var.get()}
            for key, widget in self.dynamic_net_vars.items():
                if isinstance(widget, ctk.CTkEntry):
                    val_str = widget.get()
                    if '.' in val_str or 'e' in val_str.lower():
                        net_kwargs[key] = float(val_str)
                    else:
                        net_kwargs[key] = int(val_str)
                else:
                    net_kwargs[key] = widget.get()

            net_config = NetworkConfig(**net_kwargs)

            # --- ГЛОБАЛЬНЫЕ НАСТРОЙКИ ---
            trainer_mod.ADAM_EPOCHS = int(self.adam_epochs.get())
            trainer_mod.ADAM_LR = float(self.adam_lr.get())
            trainer_mod.LBFGS_EPOCHS = int(self.lbfgs_epochs.get())
            trainer_mod.LBFGS_LR = float(self.lbfgs_lr.get())
            trainer_mod.LBFGS_MAX_ITER = int(self.lbfgs_iter.get())
            trainer_mod.BC_PENALTY = float(self.bc_penalty.get())
            trainer_mod.AUTO_BALANCE_ENABLED = self.autobalance_var.get()
            trainer_mod.NTK_ANALYSIS_EVERY = int(self.ntk_every.get())
            patience_val = int(self.early_stopping.get())
            out_dir = config.OUTPUT_DIR

            self.status_label.configure(text="Статус: Генерация сеток (Обучение + Валидация)...")
            domain = make_domain(domain_name)
            solution = SOLUTIONS[solution_name]()
            
            # ==========================================================
            # 1. ГЕНЕРАЦИЯ ОБУЧАЮЩЕЙ СЕТКИ (TRAIN)
            # ==========================================================
            mesher_train = Mesher(
                max_area=float(self.mesh_area.get()), 
                lloyd_iters=int(self.mesh_lloyd.get()), 
                boundary_density=int(self.mesh_density.get())
            )
            mesh_train = mesher_train.build(domain)
            
            quad_builder_train = QuadratureBuilder(
                tri_order=int(self.quad_tri.get()), 
                line_order=int(self.quad_line.get()), 
                device=config.DEVICE
            )
            quad_train = quad_builder_train.build(mesh_train, domain)

            # ==========================================================
            # 2. ГЕНЕРАЦИЯ ВАЛИДАЦИОННОЙ СЕТКИ (EVAL)
            # ==========================================================
            mesher_eval = Mesher(
                max_area=config.EVAL_TRI_AREA, 
                lloyd_iters=int(self.mesh_lloyd.get()), 
                boundary_density=config.EVAL_BOUNDARY_DENSITY
            )
            mesh_eval = mesher_eval.build(domain)
            
            quad_builder_eval = QuadratureBuilder(
                tri_order=config.EVAL_GAUSS_TRI_ORDER, 
                line_order=config.EVAL_GAUSS_LINE_ORDER, 
                device=config.DEVICE
            )
            quad_eval = quad_builder_eval.build(mesh_eval, domain)

            # --- Отрисовка обучающей сетки ---
            mesh_train_path = os.path.join(out_dir, f"{domain_name}_mesh_train.png")
            plot_mesh(mesh_train, f"{domain_name} (Train)", domain.bc_type, mesh_train_path, quad=quad_train)
            
            # --- Отрисовка валидационной сетки ---
            mesh_eval_path = os.path.join(out_dir, f"{domain_name}_mesh_eval.png")
            plot_mesh(mesh_eval, f"{domain_name} (Eval)", domain.bc_type, mesh_eval_path, quad=quad_eval)

            # Выводим на экран обучающую по умолчанию
            self.after(0, self.update_image, "Сетка", mesh_train_path)

            self.status_label.configure(text="Статус: Оптимизация сети...")
            log_path = os.path.join(out_dir, f"gui_log_{domain_name}.txt")
            for f in glob.glob(os.path.join(out_dir, f"{domain_name}_fields_*.png")):
                os.remove(f)

            with FileLogger(log_path, also_print=True) as logger:
                trainer = Trainer(
                    domain=domain, 
                    quad=quad_train, 
                    eval_quad=quad_eval,
                    solution=solution, 
                    logger=logger, 
                    config=net_config
                )
                trainer.train(patience=patience_val)

            self.after(0, self.update_image, "Метрики", os.path.join(out_dir, f"{domain_name}_metrics.png"))
            self.after(0, self.update_image, "NTK", os.path.join(out_dir, "ntk_evolution_dashboard.png"))
            
            field_imgs = sorted(glob.glob(os.path.join(out_dir, f"{domain_name}_fields_*.png")))
            if field_imgs:
                self.after(0, self.update_image, "Поля", field_imgs[-1])

            self.after(0, self.training_finished, True)

        except Exception as e:
            print(f"Ошибка: {e}")
            import traceback
            traceback.print_exc()
            self.after(0, self.training_finished, False)

    def update_image(self, tab_name, img_path):
        if tab_name in self.canvases:
            self.canvases[tab_name].load_image(img_path)

    def training_finished(self, success):
        self.run_btn.configure(state="normal", text="Запустить обучение")
        self.refresh_gallery_list()
        if success:
            self.status_label.configure(text="Статус: Завершено (Успех)", text_color="#10B981")
        else:
            self.status_label.configure(text="Статус: Произошла ошибка", text_color="#EF4444")

if __name__ == "__main__":
    app = PINNApp()
    app.mainloop()