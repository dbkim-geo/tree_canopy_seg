import torch
import os
import numpy as np
from PIL import Image
from config import ModelType

@torch.no_grad()
def predict(model, dataloader, device, model_type: ModelType, save_dir: str, threshold=0.5):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    for i, batch in enumerate(dataloader):
        rgb = batch["rgb"].to(device)
        fake = batch["fake"].to(device)

        # 모델 분기
        if model_type == ModelType.DUAL:
            preds = model(rgb, fake)
        elif model_type == ModelType.BASE:
            preds = model(rgb)
        else:
            raise ValueError(f"[Predict] Unknown model_type: {model_type}")

        preds_bin = (preds > threshold).float()

        for b in range(preds_bin.shape[0]):
            pred_mask = preds_bin[b, 0].cpu().numpy() * 255
            pred_mask = Image.fromarray(pred_mask.astype(np.uint8))
            pred_mask.save(os.path.join(save_dir, f"pred_{i * preds_bin.shape[0] + b:04}.png"))
