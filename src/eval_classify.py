import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights
import pandas as pd
from PIL import Image

from tqdm import tqdm
import numpy as np

import numpy as np


def _read_raw_mask(mask_path: str) -> np.ndarray:
    """Читает RAW маску"""
    with open(mask_path, 'rb') as f:
        # if task == 'gt_instance':
        data = np.fromfile(f, dtype=np.uint8) # change 8 or 32
        # else:
        #     data = np.fromfile(f, dtype=np.uint8)
    mask = data.reshape((500,500))
    # mask = convert_to_binary_mask(mask)
    mask_pil = Image.fromarray(mask)
    mask_pil = mask_pil.resize((256, 256), Image.Resampling.NEAREST)
    mask = np.array(mask_pil)
    return mask


def create_model(model_name='resnet50', num_classes=12, freeze_backbone=True):
    """
    Создает модель с замороженным бэкбоном и новым классификатором
    
    Args:
        model_name: 'resnet50' или 'efficientnet_b0'
        num_classes: количество классов
        freeze_backbone: заморозить ли веса бэкбона
    
    Returns:
        model: подготовленная модель
    """
    if model_name == 'resnet50':
        # Загружаем предобученную ResNet50
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        if freeze_backbone:
            # Замораживаем все слои бэкбона
            for param in model.parameters():
                param.requires_grad = False
        
        # Заменяем последний fully connected слой
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        
    elif model_name == 'efficientnet_b0':
        # Загружаем предобученный EfficientNet-B0
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        for i, param in enumerate(model.parameters()):
            if i < int(len(list(model.parameters())) * 0.7):
                param.requires_grad = False
        
        # Заменяем последний классификатор
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

def load_model(model_path, model_name='efficientnet_b0', num_classes=12, device='cpu'):
    """
    Загружает сохраненную модель
    
    Args:
        model_path: путь к сохраненной модели (.pth)
        model_name: архитектура модели
        num_classes: количество классов
        device: устройство для инференса
    
    Returns:
        model: загруженная модель
    """
    # Создаем модель с той же архитектурой
    model = create_model(model_name, num_classes, freeze_backbone=False)
    model = model.to(device)
    
    # Загружаем веса
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def simple_predict(model_path, test_csv, img_dir, output_path='predictions.csv'):
    """
    Простая функция для получения предсказаний
    
    Args:
        model_path: путь к модели
        test_csv: путь к тестовому CSV
        img_dir: директория с изображениями
        output_path: куда сохранить результат
    """
    device = torch.device('cuda')
    
    # Загружаем данные
    test_df = pd.read_csv(test_csv)
    classes = sorted(test_df['class'].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    # Загружаем модель
    model = load_model(model_path, 'efficientnet_b0', len(classes), device)
    
    # Трансформации
    transform = transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Предсказания
    predictions = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Predicting'):
        img_path = img_dir+row['img_path']
        image = Image.fromarray(_read_raw_mask(img_path)).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
            pred = output.argmax(1).item()
        
        predictions.append({
            'image_path': row['img_path'],
            'true_label': row['class'],
            'predicted_label': idx_to_class[pred]
        })
    
    # Сохраняем
    results_df = pd.DataFrame(predictions)
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    return results_df


def main():
    # Использование
    results = simple_predict(
        model_path='models/best_model_efficientnet_b0.pth',
        test_csv='data/val_annotations.csv',
        img_dir='data/dataset_classification_v3',
    )


if __name__ == "__main__":
   main()
