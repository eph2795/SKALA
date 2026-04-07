from src.rcnn import MaskRCNN
from src.unet import UNet


def create_model(model_name):
    params = {"num_classes": 30}

    if model_name == "unet":
        params["num_classes"] = 2
        return UNet(**params)
    elif model_name == "mask_rcnn":
        return MaskRCNN(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")
