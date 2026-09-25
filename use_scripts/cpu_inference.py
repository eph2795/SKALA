"""Классификация шлифов на CPU без использования GPU.

Скрипт принудительно выполняет инференс на CPU (даже если CUDA доступна)
и обрабатывает не более MAX_SAMPLES образцов из data/classification_test.

Запуск из корня репозитория:
    uv run -m use_scripts.cpu_inference
"""

import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.eval_classify import _read_raw_mask, load_model

MAX_SAMPLES = 10


def build_transform() -> transforms.Compose:
    """Преобразования, одинаковые с src/eval_classify.simple_predict."""
    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def select_samples(test_df: pd.DataFrame, img_dir: Path, max_samples: int) -> tuple[list[dict], dict[int, int]]:
    """Выбирает до max_samples строк с существующими RAW-файлами.

    Сопоставление индексов модели с классами строится по всему CSV: оно должно
    совпадать с обучающим (12 классов), даже если в выборке есть не все классы.
    """
    classes = sorted(test_df["class"].unique())
    idx_to_class = {idx: cls for idx, cls in enumerate(classes)}

    samples: list[dict] = []
    for _, row in test_df.iterrows():
        if len(samples) >= max_samples:
            break
        img_path = img_dir / str(row["img_path"]).lstrip("/")
        if img_path.is_file():
            samples.append({"img_path": img_path, "true_label": row["class"]})
        else:
            tqdm.write(f"Пропуск {img_path.name}: файл не найден")
    return samples, idx_to_class


def classify_on_cpu(
    classifier_weights: str | Path,
    test_csv: str | Path,
    img_dir: str | Path,
    output_path: str | Path = "results/cpu_clf_predictions.csv",
    max_samples: int = MAX_SAMPLES,
) -> pd.DataFrame:
    """Классифицирует до MAX_SAMPLES шлифов на CPU и сохраняет таблицу предсказаний."""
    device = torch.device("cpu")  # принудительно CPU, даже если CUDA доступна
    classifier_weights = Path(classifier_weights)
    test_csv = Path(test_csv)
    img_dir = Path(img_dir)
    output_path = Path(output_path)
    max_samples = min(max_samples, MAX_SAMPLES)

    if not classifier_weights.is_file():
        raise FileNotFoundError(f"Файл модели не найден: {classifier_weights}")
    if not test_csv.is_file():
        raise FileNotFoundError(f"CSV с метками не найден: {test_csv}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Папка с изображениями не найдена: {img_dir}")

    test_df = pd.read_csv(test_csv)
    samples, idx_to_class = select_samples(test_df, img_dir, max_samples)
    if not samples:
        print(f"В {img_dir} не найдено ни одного образца из {test_csv}, инференс не выполнен")
        return pd.DataFrame()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Инференс на устройстве: {device}, шлифов - {len(samples)}, классов - {len(idx_to_class)}")

    model = load_model(classifier_weights, "efficientnet_b0", len(idx_to_class), device)
    transform = build_transform()

    predictions: list[dict] = []
    start_time = time.time()
    with torch.no_grad():
        for sample in tqdm(samples, desc="Predicting on CPU"):
            image = Image.fromarray(_read_raw_mask(str(sample["img_path"]))).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)

            pred = model(input_tensor).argmax(1).item()
            predictions.append(
                {
                    "image_path": sample["img_path"].name,
                    "true_label": sample["true_label"],
                    "predicted_label": idx_to_class[pred],
                }
            )
    elapsed = time.time() - start_time

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(output_path, index=False)
    print(f"Предсказания сохранены в {output_path}")

    print(f"Инференс на CPU занял {elapsed:.2f} секунд ({elapsed / len(samples):.2f} сек на шлиф)")
    for _, row in pred_df.iterrows():
        mark = "+" if row["true_label"] == row["predicted_label"] else "-"
        print(f"[{mark}] {row['image_path']}: true={row['true_label']}, pred={row['predicted_label']}")

    accuracy = (pred_df["true_label"] == pred_df["predicted_label"]).mean()
    print(f"Accuracy на {len(pred_df)} шлифах: {accuracy:.2f}")
    return pred_df


if __name__ == "__main__":
    classify_on_cpu(
        classifier_weights="models/best_model_efficientnet_b0.pth",
        test_csv="data/clf_test_labels.csv",
        img_dir="data/classification_test",
        output_path="results/cpu_clf_predictions.csv",
    )
