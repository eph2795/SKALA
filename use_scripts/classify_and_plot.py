import pandas as pd
import matplotlib.pyplot as plt
from src.eval_classify import simple_predict
from collections import Counter


def classify_grains_and_plot(mask_rcnn_weights, classifier_weights, test_csv, img_dir):
    # Шаг 1: Получить предсказания для каждого зерна (ID, класс)
    simple_predict(
        model_path=classifier_weights,
        test_csv=test_csv,  # CSV с колонками: grain_id, path_to_grain_mask
        img_dir=img_dir,
        output_path="temp_pred.csv",
    )
    pred_df = pd.read_csv("temp_pred.csv")
    class_counts = Counter(pred_df["pred_class"])

    # Шаг 2: Построить гистограмму
    class_names = [f"Class_{i}" for i in range(12)]  # замените на имена из диаграммы Шутова
    counts = [class_counts.get(i, 0) for i in range(12)]

    plt.figure(figsize=(12, 6))
    plt.bar(class_names, counts)
    plt.title("Распределение минеральных зёрен по типам (Диаграмма Шутова)")
    plt.ylabel("Количество зёрен")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/shutov_diagram_distribution.png")
    print("Гистограмма сохранена")


# Пример вызова
# classify_grains_and_plot("models/mask_rcnn.pth", "models/efficientnet.pth", "annotations.csv", "data/grain_masks/")
