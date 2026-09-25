import json
from pathlib import Path

from src.metrics import calc_masks

test_folder = Path("data/test_slices")
models = {"Mask R-CNN": "models/mask_rcnn.pth", "U-Net": "models/unet.pth"}
comparison_results = {}

for name, path in models.items():
    print(f"Оценка модели: {name}")
    # Ваша функция calc_masks должна возвращать метрики
    final_metrics, avg_metrics = calc_masks(folder=test_folder, model_path=path)
    comparison_results[name] = avg_metrics
    print(f"{name} -> IoU: {avg_metrics['iou']:.3f}, F1: {avg_metrics['f1']:.3f}\n")

# Сравнение и сохранение результатов
with open("results/models_comparison.json", "w") as f:
    json.dump(comparison_results, f, indent=4)
print("Сравнение сохранено в results/models_comparison.json")
