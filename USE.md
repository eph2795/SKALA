# Новые варианты использования и инструкции

## 1. Пакетный анализ и визуализация результатов для геологического отчета
**Сценарий:** После съемки серии шлифов (например, 50 образцов) необходимо автоматически обработать их, сохранить не только метрики, но и визуальные наложения масок для каждого образца в отчет.

**Пошаговая инструкция:**
1.  **Подготовка данных:** Организуйте файлы в папке `data/slices` по принципу: для каждого образца своя подпапка с тремя PNG-изображениями (`_x`, `_xy`, `_y`) и RAW-маской.
2.  **Создайте новый скрипт** `batch_analysis_report.py` в корне проекта.
3.  **Используйте этот код:**
    ```python
    import torch
    from pathlib import Path
    import pandas as pd
    from tqdm import tqdm
    from src.create_model import create_model
    from src.mask import get_one_mask
    from src.visual import visualize_segmentation # предположительно есть в visual.py

    def batch_process_and_report(samples_dir, model_path, output_csv, output_plots_dir):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_model("mask_rcnn")
        model.load_state_dict(torch.load(model_path)["model_state_dict"])
        model.to(device).eval()

        results = []
        output_plots_dir = Path(output_plots_dir)
        output_plots_dir.mkdir(exist_ok=True)

        for sample_folder in tqdm(list(Path(samples_dir).iterdir())):
            if not sample_folder.is_dir(): continue
            # Ищем файлы по шаблону (уточните под свои имена)
            img_x = next(sample_folder.glob("*_x.png"), None)
            img_xy = next(sample_folder.glob("*_xy.png"), None)
            img_y = next(sample_folder.glob("*_y.png"), None)
            mask_path = next(sample_folder.glob("*.raw"), None)

            if not all([img_x, img_xy, img_y, mask_path]): continue

            # Инференс и получение предсказанной маски
            pred_mask = get_one_mask(str(img_x), str(img_xy), str(img_y), str(mask_path), model_path)

            # Визуализация и сохранение (функцию нужно реализовать в visual.py или здесь)
            # plot_path = output_plots_dir / f"{sample_folder.name}_pred_vs_gt.png"
            # visualize_segmentation(img_x, pred_mask, mask_path, plot_path)

            # Расчет метрик (используя ваш код из metrics.py)
            # metrics = calc_segmentation_metrics(pred_mask, mask_path) # пример
            # metrics['sample_name'] = sample_folder.name
            # results.append(metrics)

        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"Отчет сохранен в {output_csv}")

    if __name__ == "__main__":
        batch_process_and_report(
            samples_dir="data/raw_samples", # ваша папка с образцами
            model_path="models/mask_rcnn_best.pth",
            output_csv="results/batch_report.csv",
            output_plots_dir="results/plots"
        )
    ```

## 2. Генерация синтетического датасета для дообучения модели
**Сценарий:** У вас есть несколько реальных шлифов, и вы хотите сгенерировать сотни вариаций (с поворотами, шумами, изменением яркости) для улучшения устойчивости модели к новым условиям съемки.

**Пошаговая инструкция:**
1.  Установите дополнительную библиотеку: `pip install albumentations`
2.  **Создайте скрипт** `synthetic_data_gen.py`
3.  **Используйте код для аугментации:**
    ```python
    import albumentations as A
    import cv2
    from pathlib import Path
    import numpy as np

    def generate_synthetic_dataset(source_dir, target_dir, num_variants=100):
        source_dir = Path(source_dir)
        target_dir = Path(target_dir)
        target_dir.mkdir(exist_ok=True)

        # Определяем пайплайн аугментаций
        transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2)
        ], additional_targets={'mask': 'mask'})

        for i in range(num_variants):
            # Берем случайный исходный срез
            orig_img_path = np.random.choice(list(source_dir.glob("*_x.png")))
            # Нужно найти соответствующие маску и др. модальности
            # ... логика поиска соответствующих файлов ...
            # image = cv2.imread(str(orig_img_path))
            # mask = np.fromfile(mask_path, dtype=np.uint8).reshape(500,500)

            # transformed = transform(image=image, mask=mask)
            # aug_img, aug_mask = transformed['image'], transformed['mask']

            # Сохраняем в target_dir с новым именем
            # ...
        print(f"Сгенерировано {num_variants} синтетических образцов в {target_dir}")

    # Пример вызова
    # generate_synthetic_dataset("data/real_slices", "data/synthetic_dataset")
    ```

## 3. Классификация зерен с выводом распределения по диаграмме Шутова
**Сценарий:** После сегментации каждого зерна нужно не только предсказать его класс, но и построить гистограмму распределения 12 классов для всего шлифа.

**Пошаговая инструкция:**
1.  Сначала получите маски отдельных зерен (инстанс-сегментация через `rcnn.py`).
2.  **Создайте скрипт** `classify_and_plot.py`
3.  **Реализуйте агрегацию:**
    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    from eval_classify import simple_predict
    from collections import Counter

    def classify_grains_and_plot(mask_rcnn_weights, classifier_weights, test_csv, img_dir):
        # Шаг 1: Получить предсказания для каждого зерна (ID, класс)
        predictions = simple_predict(
            model_path=classifier_weights,
            test_csv=test_csv, # CSV с колонками: grain_id, path_to_grain_mask
            img_dir=img_dir,
            output_path='temp_pred.csv'
        )
        pred_df = pd.read_csv('temp_pred.csv')
        class_counts = Counter(pred_df['pred_class'])

        # Шаг 2: Построить гистограмму
        class_names = [f"Class_{i}" for i in range(12)] # замените на имена из диаграммы Шутова
        counts = [class_counts.get(i, 0) for i in range(12)]

        plt.figure(figsize=(12,6))
        plt.bar(class_names, counts)
        plt.title("Распределение минеральных зёрен по типам (Диаграмма Шутова)")
        plt.ylabel("Количество зёрен")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("results/shutov_diagram_distribution.png")
        print("Гистограмма сохранена")

    # Пример вызова
    # classify_grains_and_plot("models/mask_rcnn.pth", "models/efficientnet.pth", "annotations.csv", "data/grain_masks/")
    ```

## 4. Сравнение двух моделей сегментации (Mask R-CNN vs U-Net) на тестовом наборе
**Сценарий:** Вы хотите независимо проверить утверждение из проекта, что Mask R-CNN лучше U-Net для ваших данных.

**Пошаговая инструкция:**
1.  Убедитесь, что обучены обе модели (`models/mask_rcnn.pth` и `models/unet.pth`).
2.  **Создайте скрипт** `compare_models.py`
3.  **Используйте существующие функции:**
    ```python
    from src.metrics import calc_masks
    from pathlib import Path

    test_folder = Path("data/test_slices")
    models = {
        "Mask R-CNN": "models/mask_rcnn.pth",
        "U-Net": "models/unet.pth"
    }
    comparison_results = {}

    for name, path in models.items():
        print(f"Оценка модели: {name}")
        # Ваша функция calc_masks должна возвращать метрики
        final_metrics, avg_metrics = calc_masks(folder=test_folder, model_path=path)
        comparison_results[name] = avg_metrics
        print(f"{name} -> IoU: {avg_metrics['iou']:.3f}, F1: {avg_metrics['f1']:.3f}\n")

    # Сравнение и сохранение результатов
    import json
    with open("results/models_comparison.json", "w") as f:
        json.dump(comparison_results, f, indent=4)
    print("Сравнение сохранено в results/models_comparison.json")
    ```

## 5. Использование CPU-инференса для полевых условий (без GPU)
**Сценарий:** У вас есть ноутбук без мощной видеокарты, но нужно обработать несколько шлифов прямо в полевой лаборатории.

**Пошаговая инструкция:**
1.  **Создайте скрипт** `cpu_inference.py`, явно указав устройство.
2.  **Адаптируйте код принудительно под CPU:**
    ```python
    import torch
    from pathlib import Path
    from src.mask import get_one_mask
    import time

    # Принудительно используем CPU, даже если CUDA доступен
    device = torch.device('cpu')

    # Модифицируем функцию get_one_mask или создаем обертку
    def predict_on_cpu(image_paths, mask_path, model_path):
        # Загружаем модель на CPU
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()

        start_time = time.time()
        # Здесь должен быть цикл инференса для каждого изображения
        # ... используя model и передавая данные на CPU
        elapsed = time.time() - start_time

        print(f"Инференс на CPU занял {elapsed:.2f} секунд")
        # return pred_mask

    # Пример вызова
    # predict_on_cpu(["img_x.png", "img_xy.png", "img_y.png"], "mask.raw", "models/model_cpu_compatible.pth")
    ```
    **Важно:** Перед этим сконвертируйте модель: `model.cpu()` и сохраните через `torch.save(model.state_dict(), "model_cpu.pth")`.

# Общие рекомендации по запуску
1.  **Установка зависимостей:** Выполните в корне репозитория: `pip install torch torchvision numpy pandas scikit-learn pillow tqdm matplotlib opencv-python albumentations`
2.  **Структура данных:** Строго соблюдайте ожидаемый библиотекой формат (размер 500×500, PNG для изображений, RAW для масок).
3.  **Пути:** Всегда используйте `pathlib.Path` для указания путей к файлам, это кроссплатформенно.
4.  **Тестирование:** Перед запуском пакетной обработки проверьте работу на 1-2 образцах, используя отладочную печать промежуточных размеров тензоров.

