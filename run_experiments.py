import os
import shutil

# Импортируем модули проекта
import config
from geometry.domains import make_domain
from problems.solutions import SOLUTIONS
from geometry.mesher import Mesher
from geometry.quadrature import QuadratureBuilder
from networks.configs import PRESET_CONFIGS
from training.trainer import Trainer
from file_io.logger import FileLogger

# Импортируем модули, в которых нужно подменить OUTPUT_DIR для корректного сохранения
import training.trainer
import evaluation.callbacks

def set_output_directory(new_dir: str):
    """Обновляет директорию вывода во всех связанных модулях."""
    os.makedirs(new_dir, exist_ok=True)
    config.OUTPUT_DIR = new_dir
    training.trainer.OUTPUT_DIR = new_dir
    evaluation.callbacks.OUTPUT_DIR = new_dir

def run_automated_tests():
    # Определяем конфигурации тестов согласно заданию
    tests = [
        {
            "prefix": "test1",
            "domain_name": "square",
            "solution_name": "steep_peak"
        },
        {
            "prefix": "test2",
            "domain_name": "l_shape_mixed",
            "solution_name": "high_freq"
        }
    ]

    # Определяем архитектуры для тестирования (имя папки -> ключ в PRESET_CONFIGS)
    architectures = {
        "BSplineKAN": "kan",
        "wav_kan": "wav-kan",
        "cheby_kan": "cheby_kan"
    }

    base_data_dir = "data"
    os.makedirs(base_data_dir, exist_ok=True)

    for test in tests:
        test_prefix = test["prefix"]
        domain_name = test["domain_name"]
        solution_name = test["solution_name"]

        for arch_folder, arch_key in architectures.items():
            # Формируем имя папки, например: data/test1_BSplineKAN
            folder_name = f"{test_prefix}_{arch_folder}"
            output_dir = os.path.join(base_data_dir, folder_name)
            
            # Применяем новую директорию
            set_output_directory(output_dir)

            print("="*60)
            print(f"ЗАПУСК ТЕСТА: {folder_name}")
            print(f"Домен: {domain_name} | Решение: {solution_name} | Архитектура: {arch_key}")
            print("="*60)

            try:
                # 1. Инициализация домена и точного решения
                domain = make_domain(domain_name)
                solution = SOLUTIONS[solution_name]()

                # 2. Генерация обучающей сетки и квадратур
                mesher_train = Mesher(
                    max_area=config.TRAIN_TRI_AREA,
                    lloyd_iters=3,
                    boundary_density=config.TRAIN_BOUNDARY_DENSITY
                )
                mesh_train = mesher_train.build(domain)

                quad_builder_train = QuadratureBuilder(
                    tri_order=config.TRAIN_GAUSS_TRI_ORDER,
                    line_order=config.TRAIN_GAUSS_LINE_ORDER,
                    device=config.DEVICE
                )
                quad_train = quad_builder_train.build(mesh_train, domain)

                # 3. Генерация валидационной сетки и квадратур
                mesher_eval = Mesher(
                    max_area=config.EVAL_TRI_AREA,
                    lloyd_iters=3,
                    boundary_density=config.EVAL_BOUNDARY_DENSITY
                )
                mesh_eval = mesher_eval.build(domain)

                quad_builder_eval = QuadratureBuilder(
                    tri_order=config.EVAL_GAUSS_TRI_ORDER,
                    line_order=config.EVAL_GAUSS_LINE_ORDER,
                    device=config.DEVICE
                )
                quad_eval = quad_builder_eval.build(mesh_eval, domain)

                # 4. Настройка архитектуры сети
                net_config = PRESET_CONFIGS[arch_key]

                # 5. Определение файла для логирования
                log_path = os.path.join(output_dir, f"training_log_{folder_name}.txt")

                # 6. Инициализация и запуск обучения
                with FileLogger(log_path, also_print=True) as logger:
                    trainer = Trainer(
                        domain=domain,
                        quad=quad_train,
                        eval_quad=quad_eval,
                        solution=solution,
                        logger=logger,
                        config=net_config
                    )
                    
                    # Запускаем тренировку (настройки эпох берутся из config.py)
                    trainer.train(patience=50)
                    
                print(f"✅ Тест {folder_name} успешно завершён. Результаты сохранены в {output_dir}\n")

            except Exception as e:
                print(f"❌ Ошибка во время выполнения теста {folder_name}: {e}\n")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    run_automated_tests()