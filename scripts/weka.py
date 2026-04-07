import numpy as np

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from PIL import Image
import json

from src.visual import visualize_segmentation_diff
from src.mask import _read_raw_mask, convert_to_binary_mask


def calc_weka_masks(
    folder: Path = Path("data/img_1__slices_D"),
):
    final_metrics = defaultdict(list)
    keys: list[str] = [
        "tp",
        "tn",
        "fp",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "iou",
    ]
    for f in tqdm(list(folder.iterdir())[:9]):
        mask_path = str(f / f"grains_{f.name}__uint32__500x500.raw")
        weka_mask = Image.open(f"data/WEKA/Classified_{f.name}.png").convert("RGB").resize((256, 256))
        weka_mask = convert_to_binary_mask(np.asarray(weka_mask))
        ground_truth = _read_raw_mask(mask_path)
        metrics = visualize_segmentation_diff(ground_truth, weka_mask)
        for key in keys:
            final_metrics[key].append(metrics[key])

    averaged_metrics = {
        key: sum(final_metrics[key]) / len(final_metrics[key]) for key in ["precision", "recall", "iou", "f1"]
    }

    return final_metrics, averaged_metrics


if __name__ == "__main__":
    final, average = calc_weka_masks()
    print(average)
    Path("result__weka_freeze.json").write_text(json.dumps(average, indent=2, ensure_ascii=False))
