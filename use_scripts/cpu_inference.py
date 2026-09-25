import torch
import time

# Принудительно используем CPU, даже если CUDA доступен
device = torch.device("cpu")


# Модифицируем функцию get_one_mask или создаем обертку
def predict_on_cpu(image_paths, mask_path, model_path):
    # Загружаем модель на CPU
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()

    start_time = time.time()
    # Здесь должен быть цикл инференса для каждого изображения
    # ... используя model и передавая данные на CPU
    elapsed = time.time() - start_time

    print(f"Инференс на CPU занял {elapsed:.2f} секунд")
    # return pred_mask


# Пример вызова
# predict_on_cpu(["img_x.png", "img_xy.png", "img_y.png"], "mask.raw", "models/model_cpu_compatible.pth")
