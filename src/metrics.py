import numpy as np
from src.visual import visualize_segmentation_diff
from src.mask import _read_raw_mask, get_one_mask
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


def calc_masks(
    folder: Path = Path("data/img_1__slices_D"),
    model_path: str = "models/freeze_final_model_x_noise_gt_instance_checkpoint_epoch_10.pth",
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
    for f in tqdm(list(folder.iterdir())[:10]):
        image_1_path = str(f / f"thin_{f.name}_x__500x500.png")
        image_2_path = str(f / f"thin_{f.name}_xy__500x500.png")
        image_3_path = str(f / f"thin_{f.name}_y__500x500.png")
        mask_path = str(f / f"grains_{f.name}__uint32__500x500.raw")
        pred_list = get_one_mask(image_1_path, image_2_path, image_3_path, mask_path, model_path)
        consensus_mask = (pred_list[0] + pred_list[1] + pred_list[2] >= 2).astype(np.uint8)
        plt.imsave(f"results/{f.name}.png", consensus_mask, cmap="viridis")
        # plt.imshow(consensus_mask)
        ground_truth = _read_raw_mask(mask_path)
        metrics = visualize_segmentation_diff(ground_truth, consensus_mask)
        for key in keys:
            final_metrics[key].append(metrics[key])

    averaged_metrics = {
        key: sum(final_metrics[key]) / len(final_metrics[key]) for key in ["precision", "recall", "iou", "f1"]
    }

    return final_metrics, averaged_metrics
