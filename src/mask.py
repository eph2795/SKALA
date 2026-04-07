import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from src.create_model import create_model


def get_one_mask(image_1_path, image_2_path, image_3_path, mask_path, model_path):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # или любой размер который ожидает модель
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # image_1_path = '/Users/masha/Desktop/softmatter_stage2/0/thin_0_x__500x500.png'
    # image_2_path = '/Users/masha/Desktop/softmatter_stage2/0/thin_0_xy__500x500.png'
    # image_3_path = '/Users/masha/Desktop/softmatter_stage2/0/thin_0_y__500x500.png'
    # mask_path = '/Users/masha/Desktop/softmatter_stage2/0/grains_0__uint32__500x500.raw'
    image_1 = Image.open(image_1_path).convert("RGB").resize((256, 256))
    image_2 = Image.open(image_2_path).convert("RGB").resize((256, 256))
    image_3 = Image.open(image_3_path).convert("RGB").resize((256, 256))

    # ground_truth = _read_raw_mask(mask_path)
    img_tensor_list = []
    for image in [image_1, image_2, image_3]:
        img_tensor_list.append(transform(image).float().unsqueeze(0))
    # img_tensor = transform(your_image).float().unsqueeze(0)
    # model_path = '/Users/masha/Desktop/freeze_final_model_x_noise_gt_instance_checkpoint_epoch_10.pth'
    device = torch.device("cuda")
    cpu = torch.device("cpu")
    model_name = "mask_rcnn"
    model = create_model(model_name)
    pretreain = torch.load(model_path, map_location="cpu")
    # pretreain = torch.load(model_path, map_location="cuda")
    model.load_state_dict(pretreain["model_state_dict"])
    model.to(device)
    model.eval()
    pred_list = {}
    with torch.no_grad():
        for j, img_tensor in enumerate(img_tensor_list):
            input_ = img_tensor.to(device)
            # output = model(img_tensor)
            output = model(input_)
            # output.to(cpu)

            if model_name == "mask_rcnn":
                # Mask R-CNN возвращает список словарей
                predictions = output[0]  # берем первый элемент батча

                # Создаем общую маску из всех объектов
                # pred_mask = np.zeros((256, 256), dtype=np.uint8)

                # print(predictions)
                if "masks" in predictions and len(predictions["masks"]) > 0:
                    masks = predictions["masks"].cpu().numpy()  # [N, 1, H, W]
                    scores = predictions["scores"].cpu().numpy()

                    H, W = masks.shape[2], masks.shape[3]
                    binary_mask = np.zeros((H, W), dtype=bool)

                    for i, (mask, score) in enumerate(zip(masks, scores), 1):
                        if score > 0.5:  # порог уверенности
                            # Бинаризуем маску
                            binary_mask |= mask[0] > 0.5
                            pred_list[j] = binary_mask.astype(np.uint8)
                            # plt.imshow(binary_mask.astype(np.uint8))
    return pred_list


def convert_to_binary_mask(mask_np):

    if len(mask_np.shape) == 3:
        if mask_np.shape[2] == 4:  # RGBA
            # Используем только RGB каналы, игнорируя альфа-канал
            gray_mask = np.mean(mask_np[:, :, :3], axis=2)
        elif mask_np.shape[2] == 3:  # RGB
            gray_mask = np.mean(mask_np, axis=2)
        else:
            # Для других форматов с 3+ каналами берем среднее по всем каналам
            gray_mask = np.mean(mask_np, axis=2)
    elif len(mask_np.shape) == 2:
        # Если маска уже в оттенках серого
        gray_mask = mask_np
    else:
        raise ValueError(f"Неожиданная размерность массива: {mask_np.shape}")

    # Нормализуем значения от 0 до 1
    if gray_mask.max() > 1:
        gray_mask = gray_mask.astype(np.float32) / 255.0

    # Создаем бинарную маску:
    # Все пиксели, которые не являются полностью черными (фоном), становятся объектами
    binary_mask = np.where(gray_mask > 0, 1, 0).astype(np.uint8)

    return binary_mask


def _read_raw_mask(mask_path: str) -> np.ndarray:
    """Читает RAW маску"""
    with open(mask_path, "rb") as f:
        # if task == 'gt_instance':
        data = np.fromfile(f, dtype=np.uint32)  # change 8 or 32
        # else:
        #     data = np.fromfile(f, dtype=np.uint8)
    mask = data.reshape((500, 500))
    mask = convert_to_binary_mask(mask)
    mask_pil = Image.fromarray(mask)
    mask_pil = mask_pil.resize((256, 256), Image.Resampling.NEAREST)
    mask = np.array(mask_pil)
    return mask
