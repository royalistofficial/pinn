# PINN + FEM: Сравнительный анализ методов решения задачи Пуассона

Репозиторий содержит реализацию **Physics-Informed Neural Networks (PINN)** и **метода конечных элементов (МКЭ/FEM)** для решения задачи Пуассона `−Δu = f` с граничными условиями Дирихле и Неймана.

---

## Возможности

- **PINN** на PyTorch с Фурье-признаками, спектральной нормализацией и corner enrichment
- **МКЭ** (P1 линейные треугольные элементы) с поддержкой смешанных ГУ
- **Априорные оценки ошибки** МКЭ с учётом регулярности решения
- **Аналитическое вычисление NTK** (Neural Tangent Kernel) и предсказание динамики обучения
- **Сравнение PINN vs МКЭ**: точность, скорость сходимости, вычислительные затраты
- Модульная архитектура, полное покрытие тестами

---

## Структура

```
pinn_project/
├── config.py                  # Конфигурация
├── main.py                    # Точка входа
├── run_tests.py               # Запуск тестов
├── analyze_weights.py         # Анализ весов
├── requirements.txt
│
├── geometry/                  # Геометрия и сетки
│   ├── domains.py             # Описание областей
│   ├── mesher.py              # Генерация триангуляций
│   └── quadrature.py          # Квадратурные формулы
│
├── networks/                  # Нейронные сети
│   ├── activations.py         # Cn-гладкие активации
│   ├── blocks.py              # Фурье-блоки, ResNet
│   ├── corners.py             # Corner enrichment
│   ├── pinn.py                # Архитектура PINN
│   └── ntk_utils.py           # NTK: якобиан, спектр, предсказание
│
├── fem/                       # Метод конечных элементов
│   ├── solver.py              # FEM солвер (P1 элементы)
│   └── apriori_estimates.py   # Априорные оценки ошибки
│
├── comparison/                # Сравнение методов
│   └── compare.py             # PINN vs FEM
│
├── problems/                  # Постановки задач
│   └── solutions.py           # Аналитические решения
│
├── functionals/               # Функционалы
│   ├── integrals.py           # Интегрирование
│   ├── operators.py           # Градиент, лапласиан
│   ├── errors.py              # Метрики ошибок
│   └── losses.py              # Функции потерь
│
├── training/                  # Обучение
│   ├── data_module.py         # Данные и загрузчики
│   └── trainer.py             # Тренер PINN
│
├── evaluation/                # Оценка
│   ├── evaluator.py           # Оценка в процессе обучения
│   └── metrics_history.py     # История метрик
│
├── visualization/             # Визуализация
│   ├── field_plotter.py       # Графики полей
│   ├── field_evaluator.py     # Вычисление полей
│   └── metrics_plotter.py     # Графики метрик
│
├── file_io/
│   └── logger.py              # Логирование
│
└── tests/                     # Тесты
    ├── test_fem.py            # Тесты МКЭ и априорных оценок
    ├── test_ntk.py            # Тесты NTK
    ├── test_comparison.py     # Тесты сравнения
    ├── test_functionals.py
    ├── test_geometry_domains.py
    ├── test_geometry_mesher.py
    ├── test_geometry_quadrature.py
    ├── test_networks.py
    ├── test_problems.py
    └── test_training.py
```

---

## Априорные оценки

Для P1 элементов на задаче Пуассона:

| Область | Макс. угол ω | α = π/ω | ||u−u_h||_{H¹} | ||u−u_h||_{L²} |
|---------|-------------|---------|-----------------|-----------------|
| Квадрат | π/2         | 1.0     | O(h)            | O(h²)           |
| Круг    | π           | 1.0     | O(h)            | O(h²)           |
| L-shape | 3π/2        | 2/3     | O(h^{2/3})      | O(h^{4/3})      |

---

## NTK (Neural Tangent Kernel)

Реализована формула предсказания линеаризованной модели:

```
ŷ^(t) ≈ K_test · K⁻¹ · (I − exp(−η·K·t)) · y
```

где K = J·Jᵀ — NTK матрица Грама.

Модуль `ntk_utils.py` предоставляет:
- Вычисление якобиана и NTK матрицы
- Предсказание динамики обучения через NTK
- Анализ спектра (собственные значения, число обусловленности, эффективный ранг)

---

## Запуск

```bash
# Установка
pip install -r requirements.txt

# Полный запуск (PINN + FEM + сравнение)
python main.py

# Только тесты
python run_tests.py

# Анализ весов обученной модели
python analyze_weights.py
```

---

## Теория

- **PINN**: Raissi, Perdikaris, Karniadakis (2019)
- **NTK**: Jacot, Gabriel, Hongler (2018)
- **Априорные оценки МКЭ**: Céa, Aubin-Nitsche, Grisvard
