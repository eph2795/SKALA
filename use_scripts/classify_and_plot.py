"""Классификация шлифов с выводом распределения по диаграмме Шутова.

Запуск из корня репозитория:
    uv run -m use_scripts.classify_and_plot
"""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torchvision import transforms
from tqdm import tqdm

from src.eval_classify import _read_raw_mask, load_model

NUM_CLASSES = 12


def build_transform() -> transforms.Compose:
    """Преобразования, одинаковые с src/eval_classify.simple_predict."""
    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict_classes(
    model: torch.nn.Module,
    test_df: pd.DataFrame,
    img_dir: Path,
    device: torch.device,
    transform: transforms.Compose,
) -> pd.DataFrame:
    """Предсказывает класс для каждого шлифа из тестового CSV и возвращает таблицу."""
    classes = sorted(test_df["class"].unique())
    idx_to_class = {idx: cls for idx, cls in enumerate(classes)}

    predictions: list[dict] = []
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
            img_path = img_dir / str(row["img_path"]).lstrip("/")
            image = Image.fromarray(_read_raw_mask(str(img_path))).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)

            pred = model(input_tensor).argmax(1).item()
            predictions.append(
                {"image_path": row["img_path"], "true_label": row["class"], "predicted_label": idx_to_class[pred]}
            )

    return pd.DataFrame(predictions)


def plot_class_distribution(pred_df: pd.DataFrame, plot_path: Path) -> None:
    """Строит гистограмму распределения 12 классов (предсказания и истинные метки)."""
    class_names = [f"Class_{idx:02d}" for idx in range(1, NUM_CLASSES + 1)]
    pred_counts = Counter(pred_df["predicted_label"])
    true_counts = Counter(pred_df["true_label"])
    pred_values = [pred_counts.get(idx, 0) for idx in range(1, NUM_CLASSES + 1)]
    true_values = [true_counts.get(idx, 0) for idx in range(1, NUM_CLASSES + 1)]

    positions = range(NUM_CLASSES)
    bar_width = 0.4
    plt.figure(figsize=(12, 6))
    plt.bar([pos - bar_width / 2 for pos in positions], true_values, width=bar_width, label="Истинные классы")
    plt.bar([pos + bar_width / 2 for pos in positions], pred_values, width=bar_width, label="Предсказанные классы")
    plt.xticks(list(positions), class_names, rotation=45)
    plt.title("Распределение шлифов по классам (Диаграмма Шутова)")
    plt.ylabel("Количество шлифов")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def calc_classification_metrics(pred_df: pd.DataFrame) -> dict[str, float]:
    """Считает метрики классификации: accuracy, precision, recall и F1 (macro average)."""
    true_labels = pred_df["true_label"]
    predicted_labels = pred_df["predicted_label"]
    return {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, average="macro", zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, average="macro", zero_division=0),
        "f1": f1_score(true_labels, predicted_labels, average="macro", zero_division=0),
    }


def classify_grains_and_plot(
    classifier_weights: str | Path,
    test_csv: str | Path,
    img_dir: str | Path,
    output_path: str | Path = "results/clf_predictions.csv",
    plot_path: str | Path = "results/shutov_diagram_distribution.png",
    metrics_path: str | Path = "results/clf_metrics.csv",
) -> pd.DataFrame:
    """Классифицирует тестовые шлифы, сохраняет предсказания, гистограмму классов и метрики."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_csv = Path(test_csv)
    img_dir = Path(img_dir)
    output_path = Path(output_path)
    plot_path = Path(plot_path)
    metrics_path = Path(metrics_path)

    if not test_csv.is_file():
        raise FileNotFoundError(f"CSV с метками не найден: {test_csv}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Папка с изображениями не найдена: {img_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(test_csv)
    classes = sorted(test_df["class"].unique())
    print(f"Инференс на устройстве: {device}, шлифов - {len(test_df)}, классов - {len(classes)}")

    model = load_model(classifier_weights, "efficientnet_b0", len(classes), device)
    pred_df = predict_classes(model, test_df, img_dir, device, build_transform())

    pred_df.to_csv(output_path, index=False)
    print(f"Предсказания сохранены в {output_path}")

    plot_class_distribution(pred_df, plot_path)
    print(f"Гистограмма сохранена в {plot_path}")

    metrics = calc_classification_metrics(pred_df)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Метрики сохранены в {metrics_path}")

    print("Метрики: " + ", ".join(f"{key}={value:.3f}" for key, value in metrics.items()))
    return pred_df


if __name__ == "__main__":
    classify_grains_and_plot(
        classifier_weights="models/best_model_efficientnet_b0.pth",
        test_csv="data/clf_test_labels.csv",
        img_dir="data/classification_test",
        output_path="results/clf_predictions.csv",
        plot_path="results/shutov_diagram_distribution.png",
        metrics_path="results/clf_metrics.csv",
    )
