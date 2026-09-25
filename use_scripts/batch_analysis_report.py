"""Пакетный анализ образцов: метрики сегментации и наложения масок для отчета.

Запуск из корня репозитория:
    uv run -m use_scripts.batch_analysis_report
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.create_model import create_model
from src.mask import _read_raw_mask
from src.visual import visualize_segmentation_diff

IMG_SIZE = (256, 256)
SCORE_THRESHOLD = 0.5
METRIC_KEYS = ["tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "f1", "iou"]


def build_transform() -> transforms.Compose:
    """Преобразование изображений, одинаковое с src/mask.py."""
    return transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_model(model_path: str | Path, device: torch.device) -> torch.nn.Module:
    """Загружает Mask R-CNN один раз и переносит его на доступное устройство."""
    model = create_model("mask_rcnn")
    pretrain = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(pretrain['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def find_sample_files(sample_folder: Path) -> dict[str, Path] | None:
    """Находит в папке образца изображения _x/_xy/_y и RAW-маску grains_*.raw.

    Поддерживаются имена вида ``thin_0_x__500x500.png`` (этот вариант
    приоритетен) и упрощенные вида ``sample_x.png``. Если чего-то не хватает,
    возвращает None.
    """
    files: dict[str, Path] = {}
    for key, kind in (("img_x", "_x"), ("img_xy", "_xy"), ("img_y", "_y")):
        for pattern in (f"*{kind}__*.png", f"*{kind}_*.png", f"*{kind}.png"):
            matches = sorted(sample_folder.glob(pattern))
            if matches:
                files[key] = matches[0]
                break
    for pattern in ("grains_*.raw", "*.raw"):
        matches = sorted(sample_folder.glob(pattern))
        if matches:
            files["mask"] = matches[0]
            break
    return files if len(files) == 4 else None


def predict_masks(
    model: torch.nn.Module,
    image_paths: list[Path],
    device: torch.device,
    transform: transforms.Compose,
) -> list[np.ndarray]:
    """Предсказывает бинарную маску для каждого изображения (_x, _xy, _y)."""
    masks = []
    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
            input_tensor = transform(image).float().unsqueeze(0).to(device)
            predictions = model(input_tensor)[0]

            binary_mask = np.zeros(IMG_SIZE, dtype=np.uint8)
            if "masks" in predictions and len(predictions["masks"]) > 0:
                instance_masks = predictions["masks"].cpu().numpy()
                scores = predictions["scores"].cpu().numpy()
                for instance_mask, score in zip(instance_masks, scores):
                    if score > SCORE_THRESHOLD:
                        binary_mask |= (instance_mask[0] > 0.5).astype(np.uint8)
            masks.append(binary_mask)
    return masks


def consensus_mask(masks: list[np.ndarray]) -> np.ndarray:
    """Сводит предсказанные маски в одну голосованием большинством (2 из 3)."""
    if not masks:
        raise ValueError("Список предсказанных масок пуст")
    threshold = len(masks) // 2 + 1
    stacked = np.sum(np.stack(masks, axis=0), axis=0)
    return (stacked >= threshold).astype(np.uint8)


def save_overlay_plot(image_path: Path, gt_mask: np.ndarray, pred_mask: np.ndarray, plot_path: Path) -> None:
    """Сохраняет наложение GT/Pred масок на исходное изображение.

    Зеленый - верно найденные объекты (TP), красный - ложные (FP),
    синий - пропущенные (FN).
    """
    image = np.array(Image.open(image_path).convert("RGB").resize(IMG_SIZE))
    overlay = image.copy()
    overlay[(gt_mask == 0) & (pred_mask == 1)] = (255, 0, 0)  # FP
    overlay[(gt_mask == 1) & (pred_mask == 0)] = (0, 0, 255)  # FN
    overlay[(gt_mask == 1) & (pred_mask == 1)] = (0, 255, 0)  # TP

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("Исходное изображение (_x)")
    axes[1].imshow(overlay)
    axes[1].set_title("Наложение: зеленый TP, красный FP, синий FN")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def batch_process_and_report(
    samples_dir: str | Path,
    model_path: str | Path,
    output_csv: str | Path,
    output_plots_dir: str | Path,
) -> pd.DataFrame:
    """Обрабатывает все образцы и сохраняет CSV с метриками и наложения масок."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples_dir = Path(samples_dir)
    output_csv = Path(output_csv)
    output_plots_dir = Path(output_plots_dir)

    if not samples_dir.is_dir():
        raise FileNotFoundError(f"Папка с образцами не найдена: {samples_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_plots_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(model_path, device)
    transform = build_transform()
    print(f"Инференс на устройстве: {device}")

    results: list[dict] = []
    sample_folders = sorted(folder for folder in samples_dir.iterdir() if folder.is_dir())
    for sample_folder in tqdm(sample_folders, desc=f"Обработка {samples_dir.name}"):
        files = find_sample_files(sample_folder)
        if files is None:
            tqdm.write(f"Пропуск {sample_folder.name}: не найдены PNG (_x/_xy/_y) или RAW-маска")
            continue

        image_paths = [files["img_x"], files["img_xy"], files["img_y"]]
        pred_mask = consensus_mask(predict_masks(model, image_paths, device, transform))
        gt_mask = _read_raw_mask(str(files["mask"]))

        metrics = visualize_segmentation_diff(gt_mask, pred_mask)
        metrics["sample_name"] = sample_folder.name
        results.append(metrics)

        plot_path = output_plots_dir / f"{sample_folder.name}_pred_vs_gt.png"
        save_overlay_plot(files["img_x"], gt_mask, pred_mask, plot_path)

    if not results:
        print(f"Образцы не найдены в {samples_dir}, отчет не сформирован")
        return pd.DataFrame()

    report = pd.DataFrame(results)[["sample_name", *METRIC_KEYS]]
    report.to_csv(output_csv, index=False)

    averaged = {key: report[key].mean() for key in ("precision", "recall", "f1", "iou")}
    print(f"Отчет сохранен в {output_csv}: образцов - {len(report)}")
    print("Средние метрики: " + ", ".join(f"{key}={value:.3f}" for key, value in averaged.items()))
    print(f"Наложения масок сохранены в {output_plots_dir}")
    return report


if __name__ == "__main__":
    batch_process_and_report(
        samples_dir="data/segmentation_test",
        model_path="models/freeze_final_model_x_noise_gt_instance_checkpoint_epoch_10.pth",
        output_csv="results/batch_report.csv",
        output_plots_dir="results/plots",
    )
