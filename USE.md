# USE.md — инструкция по запуску прикладных скриптов

Каталог `use_scripts/` содержит готовые сценарии прикладного использования библиотеки SKALA:
пакетный анализ сегментации, классификация шлифов (на GPU и CPU) и генерация
синтетического датасета аугментацией реальных образцов.

## Подготовка

Все команды выполняются из корня репозитория:

```bash
uv sync                       # установка зависимостей в .venv
git lfs pull                  # загрузка LFS-данных (data/, models/), если не скачаны при клонировании
```

Скрипты обращаются к относительным путям (`data/`, `models/`, `results/`),
поэтому запускать их нужно именно из корня репозитория. Результаты создаются
в папке `results/` (создается автоматически).

## Быстрый запуск через Makefile

| Команда | Скрипт | Назначение |
|---------|--------|------------|
| `make use1` | `batch_analysis_report` | Пакетные метрики сегментации + наложения масок |
| `make use2` | `classify_and_plot` | Классификация шлифов + диаграмма Шутова |
| `make use3` | `cpu_inference` | Классификация на CPU (без GPU) |
| `make use4` | `synthetic_data_gen` | Генерация синтетического датасета |

Эквивалентный запуск без Makefile (пример):

```bash
uv run -m use_scripts.batch_analysis_report
```

---

## 1. batch_analysis_report — пакетный анализ сегментации

**Что делает:** прогоняет Mask R-CNN на всех образцах из `data/segmentation_test`,
сводит предсказания трех модальностей (`_x`, `_xy`, `_y`) голосованием (2 из 3),
считает метрики сегментации против эталонных инстанс-масок `grains_*.raw`
и сохраняет наложения масок на изображения.

```bash
make use1
# или
uv run -m use_scripts.batch_analysis_report
```

**Входные данные (по умолчанию):**

| Параметр | Значение | Описание |
|----------|----------|----------|
| `samples_dir` | `data/segmentation_test` | Папки-образцы с `thin_*_{x,xy,y}__500x500.png` и `grains_*.raw` |
| `model_path` | `models/freeze_final_model_x_noise_gt_instance_checkpoint_epoch_10.pth` | Веса Mask R-CNN |

**Результаты:**

- `results/batch_report.csv` — метрики (tp, tn, fp, fn, accuracy, precision, recall, f1, iou) по каждому образцу;
- `results/plots/{имя_образца}_pred_vs_gt.png` — наложения: зеленый — TP, красный — FP, синий — FN;
- в консоль выводятся средние precision/recall/f1/iou.

**Примечания:** инференс выполняется на CUDA, если она доступна, иначе на CPU.
Параметры по умолчанию заданы в блоке `if __name__ == "__main__":` — для изменения
входов отредактируйте их или вызовите `batch_process_and_report(...)` из Python.

---

## 2. classify_and_plot — классификация и диаграмма Шутова

**Что делает:** классифицирует тестовые шлифы (RAW-файлы) моделью EfficientNet-B0,
строит гистограмму распределения по 12 классам (истинные vs предсказанные)
и считает метрики классификации (macro average).

```bash
make use2
# или
uv run -m use_scripts.classify_and_plot
```

**Входные данные (по умолчанию):**

| Параметр | Значение | Описание |
|----------|----------|----------|
| `classifier_weights` | `models/best_model_efficientnet_b0.pth` | Веса EfficientNet-B0 |
| `test_csv` | `data/clf_test_labels.csv` | CSV с колонками `img_path, class` |
| `img_dir` | `data/classification_test` | Папка с RAW-файлами шлифов |

**Результаты:**

- `results/clf_predictions.csv` — предсказанный и истинный класс для каждого шлифа;
- `results/shutov_diagram_distribution.png` — гистограмма распределения классов;
- `results/clf_metrics.csv` — accuracy, precision, recall, f1 (macro).

**Примечания:** инференс на CUDA, если доступна, иначе на CPU. Число классов
фиксировано (`NUM_CLASSES = 12`), сопоставление индексов с классами строится
по всему CSV.

---

## 3. cpu_inference — классификация принудительно на CPU

**Что делает:** то же, что и `classify_and_plot`, но инференс всегда выполняется
на CPU (даже при доступной CUDA) и обрабатывается не более `MAX_SAMPLES = 10`
шлифов — удобно для проверки окружения без GPU и оценки скорости.

```bash
make use3
# или
uv run -m use_scripts.cpu_inference
```

**Входные данные (по умолчанию):**

| Параметр | Значение | Описание |
|----------|----------|----------|
| `classifier_weights` | `models/best_model_efficientnet_b0.pth` | Веса EfficientNet-B0 |
| `test_csv` | `data/clf_test_labels.csv` | CSV с колонками `img_path, class` |
| `img_dir` | `data/classification_test` | Папка с RAW-файлами шлифов |
| `max_samples` | `10` | Лимит числа обрабатываемых шлифов (не более `MAX_SAMPLES`) |

**Результаты:**

- `results/cpu_clf_predictions.csv` — таблица предсказаний;
- в консоль: время инференса (полное и на один шлиф), построчная проверка
  `[+]`/`[-]` и итоговая accuracy.

---

## 4. synthetic_data_gen — генерация синтетического датасета

**Что делает:** берет случайные шлифы из `data/segmentation_test` и создает
аугментированные варианты. Геометрические искажения (повороты на 90°, отражения,
эластическая деформация) применяются синхронно ко всем трем модальностям
(`_x`, `_xy`, `_y`), инстанс-маске `grains_*.raw` и RAW-изображению `img_*.raw`;
фотометрические (шум, яркость/контраст) — только к PNG. Синтетические образцы
сохраняются в том же формате, что и исходные, поэтому результат можно
обработать скриптом `batch_analysis_report`.

```bash
make use4
# или
uv run -m use_scripts.synthetic_data_gen
```
