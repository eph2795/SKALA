# Grain Segmentation & Classification Library

Библиотека для сегментации и классификации гранул на микроизображениях с использованием глубокого обучения.


## Установка

```bash
uv sync
```

## Структура проекта

```
src/
├── rcnn.py          # Mask R-CNN модель
├── unet.py          # U-Net модель  
├── mask.py          # Загрузка и обработка масок
├── metrics.py       # Метрики сегментации
├── visual.py        # Визуализация и расчет метрик
├── create_model.py  # Фабрика моделей
└── doubleconv.py    # Вспомогательные слои
```

## Использование

### Сегментация

```python
from src.metrics import calc_masks
from pathlib import Path

# Расчет метрик сегментации
final_metrics, averaged_metrics = calc_masks(
    folder=Path("data/slices"),
    model_path="models/model.pth"
)

print(averaged_metrics)
# {'precision': 0.85, 'recall': 0.82, 'iou': 0.73, 'f1': 0.83}
```

### Классификация

```python
from eval_classify import simple_predict

# Предсказание классов
results = simple_predict(
    model_path='model.pth',
    test_csv='annotations.csv',
    img_dir='/path/to/masks/',
    output_path='predictions.csv'
)
```

## Модели

### Mask R-CNN
- Детекция и сегментация гранул
- Модифицированные anchor boxes
- Замороженный backbone

### U-Net
- Семантическая сегментация
- Skip-connections
- Выход: бинарная маска

### Классификаторы
- ResNet50 (полная заморозка)
- EfficientNet-B0 (частичная заморозка)

## Метрики

### Сегментация
- **IoU** (Intersection over Union)
- **F1-score**
- **Precision / Recall**
- **Accuracy**

### Классификация
- **OTAR** (Operator True Accept Rate)
- **OPAR** (Operator Precision Accept Rate)
- **Precision / Recall** (macro average)

## Примеры

### Загрузка модели
```python
from src.create_model import create_model

model = create_model("mask_rcnn")
model.load_state_dict(torch.load("model.pth")["model_state_dict"])
model.eval()
```

### Инференс
```python
from src.mask import get_one_mask

masks = get_one_mask(
    image_1_path="image_x.png",
    image_2_path="image_xy.png", 
    image_3_path="image_y.png",
    mask_path="mask.raw",
    model_path="model.pth"
)
```

## Форматы данных

- **Изображения**: PNG (500×500)
- **Маски**: RAW (uint8 или uint32, 500×500)
- **Аннотации**: CSV (img_path, class)

## Требования

- Python 3.8+
- PyTorch 1.10+
- torchvision
- numpy, pandas
- scikit-learn
- Pillow, tqdm

## Лицензия
