from typing import Dict

import numpy as np


def visualize_segmentation_diff(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Dict[str, float]:

    diff_map = np.zeros_like(gt_mask, dtype=np.uint8)
    diff_map[(gt_mask == 0) & (pred_mask == 0)] = 0  # TN - фон
    diff_map[(gt_mask == 0) & (pred_mask == 1)] = 1  # FP - ложное срабатывание
    diff_map[(gt_mask == 1) & (pred_mask == 0)] = 2  # FN - пропуск объекта
    diff_map[(gt_mask == 1) & (pred_mask == 1)] = 3  # TP - верно найденный объект

    # Цвета для каждого типа
    # colors = ['black', 'red', 'blue', 'green']
    # labels = ['Фон (TN)', 'Ложные (FP)', 'Пропуски (FN)', 'Верные (TP)']

    # Считаем метрики
    tp = np.sum((gt_mask == 1) & (pred_mask == 1))
    tn = np.sum((gt_mask == 0) & (pred_mask == 0))
    fp = np.sum((gt_mask == 0) & (pred_mask == 1))
    fn = np.sum((gt_mask == 1) & (pred_mask == 0))

    # Метрики
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }
