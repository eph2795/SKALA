"""Генерация синтетического датасета аугментацией реальных образцов.

Для каждого синтетического варианта берется случайный шлиф из data/segmentation_test
и аугментируется единым преобразованием: геометрические искажения (повороты на 90,
зеркальные отражения, эластическая деформация) применяются синхронно ко всем
модальностям (_x, _xy, _y) и маскам (инстанс-маска grains_*.raw, RAW-изображение
img_*.raw), фотометрические - только к PNG-изображениям.

Готовые образцы сохраняются в том же формате, что и исходные, поэтому их можно
использовать напрямую в use_scripts.batch_analysis_report.

Запуск из корня репозитория:
    uv run -m use_scripts.synthetic_data_gen
"""

import random
from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image
from tqdm import tqdm

IMG_SIZE = 500
NUM_VARIANTS = 100
RAW_DTYPES = {"img": np.uint8, "grains": np.uint32}

# Ключ -> суффикс модальности
PNG_KINDS = (("img_x", "_x"), ("img_xy", "_xy"), ("img_y", "_y"))


def build_transform() -> A.Compose:
    """Пайплайн аугментаций для мультимодальных данных.

    additional_targets типа "image" повторяют и геометрию, и фотометрию,
    тип "mask" - только геометрию с интерполяцией nearest (значения меток
    сохраняются точно).
    """
    return A.Compose(
        [
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussNoise(std_range=(0.05, 0.2), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        ],
        additional_targets={"image_xy": "image", "image_y": "image", "raw_img": "mask"},
    )


def read_raw(path: Path, dtype: np.dtype) -> np.ndarray:
    """Читает RAW-файл 500x500 с заданным типом элементов."""
    return np.fromfile(path, dtype=dtype).reshape(IMG_SIZE, IMG_SIZE)


def find_sample_files(sample_folder: Path) -> dict[str, Path] | None:
    """Находит в папке образца PNG (_x/_xy/_y) и RAW-файлы (img/grains).

    Шаблоны вида ``*_x__*.png`` и ``grains_*.raw`` не совпадают с
    "_noised"-версиями PNG, поэтому всегда берутся чистые модальности.
    Если какого-то файла нет, возвращает None.
    """
    files: dict[str, Path] = {}
    for key, kind in PNG_KINDS:
        matches = sorted(sample_folder.glob(f"*{kind}__*.png"))
        if matches:
            files[key] = matches[0]
    for key, pattern in (("raw_img", "img_*.raw"), ("mask", "grains_*.raw")):
        matches = sorted(sample_folder.glob(pattern))
        if matches:
            files[key] = matches[0]
    return files if len(files) == 5 else None


def load_sample(files: dict[str, Path]) -> dict[str, np.ndarray]:
    """Загружает все модальности и маски одного образца.

    Инстанс-маска uint32 переводится во float32: ElasticTransform использует
    cv2.remap, который не поддерживает uint32, а nearest-интерполяция
    сохраняет значения меток без изменений.
    """
    sample: dict[str, np.ndarray] = {key: np.array(Image.open(files[key]).convert("RGB")) for key, _ in PNG_KINDS}
    sample["raw_img"] = read_raw(files["raw_img"], RAW_DTYPES["img"])
    sample["mask"] = read_raw(files["mask"], RAW_DTYPES["grains"]).astype(np.float32)
    return sample


def save_sample(out_folder: Path, name: str, sample: dict[str, np.ndarray]) -> None:
    """Сохраняет синтетический образец в формате исходного датасета."""
    out_folder.mkdir(parents=True, exist_ok=True)
    for key, kind in PNG_KINDS:
        Image.fromarray(sample[key]).save(out_folder / f"thin_{name}{kind}__{IMG_SIZE}x{IMG_SIZE}.png")
    sample["raw_img"].tofile(out_folder / f"img_{name}__uint8__{IMG_SIZE}x{IMG_SIZE}.raw")
    grains = sample["mask"].astype(RAW_DTYPES["grains"])
    grains.tofile(out_folder / f"grains_{name}__uint32__{IMG_SIZE}x{IMG_SIZE}.raw")


def generate_synthetic_dataset(
    source_dir: str | Path,
    target_dir: str | Path,
    num_variants: int = 100,
    seed: int | None = None,
) -> list[Path]:
    """Генерирует num_variants аугментированных образцов из source_dir в target_dir.

    Возвращает список папок с созданными синтетическими образцами.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Папка с образцами не найдена: {source_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    transform = build_transform()

    sample_folders = sorted(folder for folder in source_dir.iterdir() if folder.is_dir())
    if not sample_folders:
        print(f"В {source_dir} не найдено ни одного образца, датасет не сформирован")
        return []

    created: list[Path] = []
    for i in tqdm(range(num_variants), desc=f"Генерация из {source_dir.name}"):
        files = None
        while files is None and sample_folders:
            files = find_sample_files(random.choice(sample_folders))
            if files is None:
                sample_folders.pop()
        if files is None:
            tqdm.write(f"Пропуск варианта {i}: в {source_dir} не осталось полных образцов")
            break

        original = load_sample(files)
        transformed = transform(
            image=original["img_x"],
            image_xy=original["img_xy"],
            image_y=original["img_y"],
            raw_img=original["raw_img"],
            mask=original["mask"],
        )
        sample = {
            "img_x": transformed["image"],
            "img_xy": transformed["image_xy"],
            "img_y": transformed["image_y"],
            "raw_img": transformed["raw_img"].astype(RAW_DTYPES["img"]),
            "mask": np.rint(transformed["mask"]).astype(RAW_DTYPES["grains"]),
        }

        name = f"aug_{i:04d}"
        out_folder = target_dir / name
        save_sample(out_folder, name, sample)
        created.append(out_folder)

    print(f"Сгенерировано {len(created)} синтетических образцов в {target_dir}")
    return created


if __name__ == "__main__":
    generate_synthetic_dataset(
        source_dir="data/segmentation_test",
        target_dir="results/synthetic_dataset",
        num_variants=NUM_VARIANTS,
        seed=0,
    )
