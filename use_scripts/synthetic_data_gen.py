import albumentations as A
from pathlib import Path
import numpy as np


def generate_synthetic_dataset(source_dir, target_dir, num_variants=100):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(exist_ok=True)

    # Определяем пайплайн аугментаций (используется в закомментированном примере ниже)
    _transform = A.Compose(
        [
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        ],
        additional_targets={"mask": "mask"},
    )

    for i in range(num_variants):
        # Берем случайный исходный срез
        _orig_img_path = np.random.choice(list(source_dir.glob("*_x.png")))
        # Нужно найти соответствующие маску и др. модальности
        # ... логика поиска соответствующих файлов ...
        # image = cv2.imread(str(_orig_img_path))
        # mask = np.fromfile(mask_path, dtype=np.uint8).reshape(500,500)

        # transformed = _transform(image=image, mask=mask)
        # aug_img, aug_mask = transformed['image'], transformed['mask']

        # Сохраняем в target_dir с новым именем
        # ...
    print(f"Сгенерировано {num_variants} синтетических образцов в {target_dir}")


# Пример вызова
# generate_synthetic_dataset("data/real_slices", "data/synthetic_dataset")
