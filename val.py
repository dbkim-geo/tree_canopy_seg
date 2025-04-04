import torch
from utils.metrics import dice_coef, iou_score
from config import ModelType

@torch.no_grad()
def validate(model, dataloader, loss_fn, device, model_type: ModelType):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0

    for batch in dataloader:
        rgb = batch["rgb"].to(device)
        fake = batch["fake"].to(device)
        label = batch["label"].to(device)

        # 모델 분기
        if model_type == ModelType.DUAL:
            preds = model(rgb, fake)
        elif model_type == ModelType.BASE:
            preds = model(rgb)
        else:
            raise ValueError(f"[Val] Unknown model_type: {model_type}")

        loss = loss_fn(preds, label)
        total_loss += loss.item()
        total_dice += dice_coef(preds, label)
        total_iou += iou_score(preds, label)

    n = len(dataloader)
    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "iou": total_iou / n
    }
